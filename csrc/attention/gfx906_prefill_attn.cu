// Handwritten gfx906 prefill attention scan kernel.
//
// Replaces the Triton slim scanner on the single-sequence plain-causal
// fp16-KV head-256/GQA6/800-page shape, writing the same fp32 split-KV
// segment workspace that reduce_segments combines. fp32 accumulation over
// fp16 inputs throughout (v_fma_mix_f32 selected per term), identical
// online-softmax semantics including the m_new == -inf guard.
//
// Kernel-level validation vs the Triton slim scanner on the tuning shapes
// (interleaved medians, single layer/chunk calls): 1.75-2.4x faster at
// 16k/64k/131k contexts with rel_rms vs an fp32 reference at the same
// 2.7e-4 level as Triton (summation-order-only differences).
//
// Structure: one 64-row q tile (BLOCK_Q=10 tokens x GQA6) resident in LDS;
// TILE=32 KV tiles staged through a padded (258-half) LDS buffer; QK rows
// split across JGROUP=16 lane groups with warp-shuffle stat reduction; PV
// interleaved d mapping (4-half2 stride) so consecutive threads hit
// consecutive banks; 512 threads = 4 waves/SIMD for latency hiding.
//
// KV staging is software-pipelined: each thread's 32B tile slice is loaded
// from global with 16B vector reads into registers one tile ahead and
// committed to LDS only after the barrier that frees the buffer, so DRAM/L2
// latency hides behind QK/PV instead of stalling a barrier-separated
// staging phase (counters showed the unpipelined kernel latency-bound at
// ~37% VALU with DRAM fetch worth only ~3ms of a 1025ms call).

#ifdef USE_ROCM

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>

#include <climits>

namespace {

constexpr int QHEADS = 24;
constexpr int KVHEADS = 4;
constexpr int GQA = 6;
constexpr int HEAD = 256;
constexpr int PAGE = 800;
constexpr int NSEG = 2;

// 16B staging granularity: a TILE x HEAD half tile is TILE * 32 uint4
// slices, spread over the block as u = tid + s * THREADS (row = u / 32).
constexpr int KV_ROW16 = HEAD / 8;

template <int TILE, int THREADS>
__device__ __forceinline__ void kv_load(uint4* pre, const __half* row0,
                                        int s1, int tid, int tbase,
                                        int valid) {
  // rows at tbase + r >= valid (masked-phase tail) stage as +0.0 halves,
  // bit-identical to the previous __floats2half2_rn(0, 0) fill
#pragma unroll
  for (int s = 0; s < TILE * KV_ROW16 / THREADS; ++s) {
    const int u = tid + s * THREADS;
    const int r = u / KV_ROW16;
    pre[s] = (tbase + r < valid)
                 ? *reinterpret_cast<const uint4*>(
                       row0 + (size_t)r * s1 + (u % KV_ROW16) * 8)
                 : make_uint4(0u, 0u, 0u, 0u);
  }
}

template <int TILE, int THREADS, int KSTRIDE>
__device__ __forceinline__ void kv_stage(const uint4* pre, __half* lds,
                                         int tid) {
  // the 258-half row stride keeps QK's 32-row reads bank-conflict-free but
  // is not a 16B multiple, so the commit stays at half2 granularity
#pragma unroll
  for (int s = 0; s < TILE * KV_ROW16 / THREADS; ++s) {
    const int u = tid + s * THREADS;
    __half2* dst = reinterpret_cast<__half2*>(lds) +
                   (u / KV_ROW16) * (KSTRIDE / 2) + (u % KV_ROW16) * 4;
    const __half2* src = reinterpret_cast<const __half2*>(&pre[s]);
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[e] = src[e];
  }
}

// V is consumed as row pairs: Vt2[m][d] packs (V[2m][d], V[2m+1][d]) so PV
// runs on v_dot2_f32_f16 with an fp32 accumulator. The pair partner (the
// adjacent KV row, same 16B column slice) always lives in lane tid^32 of
// the same wave, so the transpose is a shuffle plus byte permutes -- no
// LDS round trip and no extra buffer (Vt2 reuses the K staging region,
// which is dead after the QK barrier).
template <int TILE, int THREADS>
__device__ __forceinline__ void kv_stage_vt(const uint4* pre, __half* lds,
                                            int tid) {
#pragma unroll
  for (int s = 0; s < TILE * KV_ROW16 / THREADS; ++s) {
    const int u = tid + s * THREADS;
    const int r = u / KV_ROW16;
    const int c16 = u % KV_ROW16;
    const bool even = (r & 1) == 0;
    // a 16B slice holds 8 halves; word i covers d's c16*8+2i and c16*8+2i+1
    // and expands to two half2 units (even row's half low, odd row's high),
    // so a Vt2 row keeps HEAD half2 units and word i stores at half2 offset
    // c16*8 + 2i
    __half2* dst =
        reinterpret_cast<__half2*>(lds) + (r >> 1) * HEAD + c16 * 8;
    const unsigned* w = reinterpret_cast<const unsigned*>(&pre[s]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned mine = w[i];
      const unsigned other = __shfl_xor(mine, 32);
      const unsigned x = even ? mine : other;
      const unsigned y = even ? other : mine;
      reinterpret_cast<uint2*>(dst)[i] =
          make_uint2(__byte_perm(x, y, 0x5410), __byte_perm(x, y, 0x7632));
    }
  }
}

__device__ __forceinline__ const __half* kv_row(
    const __half* base, int s0, int s1, int s2,
    const int* __restrict__ bt, int kvh, int tbase, int nb_tbl, int nb_cache,
    int* dbg) {
  // all KV load addresses must stay inside the cache tensor by
  // construction: a stale/torn block-table or seq_lens entry (upstream
  // write racing this kernel) must degrade to wrong data, never to a wild
  // address. Violations are counted in dbg[0]/dbg[1] for the dispatch to
  // read; the clamps are no-ops on well-formed inputs.
  //
  // the column bound and the page-id bound are different numbers: the
  // block table is sized by the longest sequence while page ids index the
  // whole pool, so a table column may legitimately hold an id past the
  // table width
  int page = tbase / PAGE;
  if ((unsigned)page >= (unsigned)nb_tbl) {
    if (dbg) atomicAdd(dbg, 1);
    page = 0;
  }
  int pbi = bt[page];
  if ((unsigned)pbi >= (unsigned)nb_cache) {
    if (dbg) atomicAdd(dbg + 1, 1);
    pbi = 0;
  }
  int tin = tbase - page * PAGE;
  if (tin >= PAGE) {
    if (dbg) atomicAdd(dbg, 1);
    tin = 0;
  }
  return base + (size_t)pbi * s0 + (size_t)tin * s1 + (size_t)kvh * s2;
}

// v_dot2_f32_f16: both fp16 products are computed exactly (11x11-bit
// mantissas fit in fp32) and accumulated in fp32 -- the same error class
// as the per-term fma chain it replaces, one instruction instead of two
__device__ __forceinline__ float dot2_f32_f16(const __half2 a,
                                              const __half2 b,
                                              const float c) {
#if defined(__clang__) && __has_builtin(__builtin_amdgcn_fdot2)
  return __builtin_amdgcn_fdot2(a, b, c, false);
#else
  union {
    __half2 h;
    unsigned u;
  } ua, ub;
  ua.h = a;
  ub.h = b;
  float r;
  __asm__("v_dot2_f32_f16 %0, %1, %2, %3"
          : "=v"(r)
          : "v"(ua.u), "v"(ub.u), "v"(c));
  return r;
#endif
}

template <int BM, int TILE, int THREADS>
__global__ void __launch_bounds__(THREADS)
gfx906_prefill_attn_kernel(const __half* __restrict__ q,  // [q_len, 24, 256]
                           const __half* __restrict__ kc,  // [nb, 800, 4, 256]
                           const __half* __restrict__ vc,  // [nb, 800, 4, 256]
                           const int* __restrict__ block_table,
                           const int* __restrict__ seq_lens,  // [1]
                           float* __restrict__ segm_out,   // [rows, 24, NSEG, 256]
                           float* __restrict__ segm_max,   // [rows, 24, NSEG]
                           float* __restrict__ segm_expsum,  // [.., NSEG]
                           const float scale, const int q_len,
                           const int nb_tbl, const int nb_cache, const int k_s0,
                           const int k_s1, const int k_s2, const int v_s0,
                           const int v_s1, const int v_s2, int* dbg) {
  constexpr int BLOCK_Q = BM / GQA;
  constexpr int JGROUP = TILE / 2;                // QK lanes per row
  constexpr int QK_ROWS = BM * JGROUP / THREADS;  // QK rows per thread
  constexpr int PV_T = THREADS / BM;              // PV threads per row
  constexpr int PV_D = HEAD / PV_T;               // PV d-slice

  const int qb = blockIdx.x;
  const int kvh = blockIdx.y;
  const int segm = blockIdx.z;
  const int tid = threadIdx.x;
  const int seq_len = seq_lens[0];
  if (qb * BLOCK_Q >= q_len) return;
  const int tps = (seq_len + NSEG * TILE - 1) / (NSEG * TILE);
  if (segm * tps * TILE >= seq_len) {
    // empty segment: the fp32 segment workspace is reused across layers,
    // so stale state must be invalidated here or reduce_segments merges
    // the previous layer's segment into this one (short prompts only:
    // seq_len <= NSEG * TILE leaves higher segments empty)
    const int i_pv_e = tid / PV_T;
    const int tok_e = qb * BLOCK_Q + i_pv_e / GQA;
    if (tok_e < q_len && tid % PV_T == 0) {
      const int head_e = kvh * GQA + i_pv_e % GQA;
      const size_t base = ((size_t)tok_e * QHEADS + head_e) * NSEG + segm;
      segm_max[base] = -INFINITY;
      segm_expsum[base] = 0.f;
    }
    return;
  }

  const int q_lo = qb * BLOCK_Q;
  const int context = seq_len - q_len;
  const int max_prefix =
      min(context + q_lo + (BM - 1) / GQA + 1, seq_len);
  const int num_tiles = (max_prefix + TILE - 1) / TILE;
  const int seg_lo = segm * tps;
  const int seg_hi = min((segm + 1) * tps, num_tiles);
  const int full_end = (context + q_lo) / TILE;

  // +2-half row padding: row starts land on distinct banks (a 512B stride
  // is a multiple of the 128B LDS bank period and would 4-way conflict)
  constexpr int QSTRIDE = HEAD + 2;
  constexpr int KSTRIDE = HEAD + 2;
  __shared__ __half Q_lds[BM * QSTRIDE];
  __shared__ __align__(16) __half KV_lds[TILE * KSTRIDE];
  __shared__ __half2 P2_lds[BM][TILE / 2 + 1];
  __shared__ float M_lds[BM], L_lds[BM], alpha_lds[BM];

  // stage Q (invalid rows read as 0) and init softmax state
  for (int idx = tid; idx < BM * (HEAD / 2); idx += THREADS) {
    const int m = idx / (HEAD / 2), d2 = idx % (HEAD / 2);
    const int tok = q_lo + m / GQA;
    __half2 val;
    if (tok < q_len) {
      const int head = kvh * GQA + m % GQA;
      // clamp the load address: the guard branch above must not rely on
      // the compiler keeping the load predicated
      const int tk = min(tok, q_len - 1);
      val = *reinterpret_cast<const __half2*>(
          q + ((size_t)tk * QHEADS + head) * HEAD + d2 * 2);
    } else {
      val = __floats2half2_rn(0.f, 0.f);
    }
    reinterpret_cast<__half2*>(Q_lds)
        [(idx / (HEAD / 2)) * (QSTRIDE / 2) + idx % (HEAD / 2)] = val;
  }
  for (int m = tid; m < BM; m += THREADS) {
    M_lds[m] = -INFINITY;
    L_lds[m] = 1.0f;
  }
  __syncthreads();

  // persistent fp32 accumulator: row i_pv, d in the thread's PV_D slice
  const int i_pv = tid / PV_T;
  float acc[PV_D];
#pragma unroll
  for (int d = 0; d < PV_D; ++d) acc[d] = 0.f;

  // QK mapping: rows [i_qk0, +QK_ROWS) x cols {j_qk0, j_qk0+1}
  const int jh = tid % JGROUP;
  const int i_qk0 = (tid / JGROUP) * QK_ROWS;
  const int j_qk0 = jh * 2;

  const int a_end = min(seg_hi, full_end);
  const int b_beg = max(seg_lo, full_end);

  // per-thread prefetch slices for the next K and V tiles. The single KV
  // buffer forces a K(j) -> V(j) -> K(j+1) handoff, so the pipeline keeps
  // four barriers per tile: [softmax done] [V visible] [PV done] [K visible]
  constexpr int SPT = TILE * KV_ROW16 / THREADS;
  uint4 pre_k[SPT], pre_v[SPT];

  // phase A prologue: K(seg_lo) into LDS, V(seg_lo) in flight
  if (a_end > seg_lo) {
    const int t0 = seg_lo * TILE;
    kv_load<TILE, THREADS>(pre_k, kv_row(kc, k_s0, k_s1, k_s2, block_table, kvh, t0, nb_tbl, nb_cache, dbg),
                           k_s1, tid, t0, INT_MAX);
    kv_stage<TILE, THREADS, KSTRIDE>(pre_k, KV_lds, tid);
    __syncthreads();
    kv_load<TILE, THREADS>(pre_v, kv_row(vc, v_s0, v_s1, v_s2, block_table, kvh, t0, nb_tbl, nb_cache, dbg),
                           v_s1, tid, t0, INT_MAX);
  }

  for (int j = seg_lo; j < a_end; ++j) {
    const int tbase = j * TILE;
    const int tnext = tbase + TILE;
    const bool more = tnext < a_end * TILE;

    // K(j+1) in flight across the whole QK/softmax/PV cycle
    if (more)
      kv_load<TILE, THREADS>(pre_k,
                             kv_row(kc, k_s0, k_s1, k_s2, block_table, kvh, tnext, nb_tbl, nb_cache, dbg),
                             k_s1, tid, tnext, INT_MAX);

    // QK dot (fp32 acc over fp16 inputs)
    float s[QK_ROWS][2];
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) s[a][0] = s[a][1] = 0.f;
    {
      const __half2* K2 = reinterpret_cast<const __half2*>(KV_lds);
      const __half2* Q2 = reinterpret_cast<const __half2*>(Q_lds);
#pragma unroll 8
      for (int d2 = 0; d2 < HEAD / 2; ++d2) {
        const __half2 k0 = K2[j_qk0 * (KSTRIDE / 2) + d2];
        const __half2 k1 = K2[(j_qk0 + 1) * (KSTRIDE / 2) + d2];
#pragma unroll
        for (int a = 0; a < QK_ROWS; ++a) {
          const __half2 qv = Q2[(i_qk0 + a) * (QSTRIDE / 2) + d2];
          s[a][0] = dot2_f32_f16(qv, k0, s[a][0]);
          s[a][1] = dot2_f32_f16(qv, k1, s[a][1]);
        }
      }
    }
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) {
      s[a][0] *= scale;
      s[a][1] *= scale;
    }

    // row stats (fold old M; reduce across the JGROUP lanes)
    float m_new[QK_ROWS], l_loc[QK_ROWS];
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) {
      m_new[a] = fmaxf(M_lds[i_qk0 + a], fmaxf(s[a][0], s[a][1]));
      l_loc[a] = 0.f;
    }
#pragma unroll
    for (int off = JGROUP / 2; off > 0; off >>= 1) {
#pragma unroll
      for (int a = 0; a < QK_ROWS; ++a)
        m_new[a] = fmaxf(m_new[a], __shfl_xor(m_new[a], off));
    }
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a)
      if (m_new[a] == -INFINITY) m_new[a] = 0.f;  // Triton's guard
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) {
      const float p0 = __expf(s[a][0] - m_new[a]);
      const float p1 = __expf(s[a][1] - m_new[a]);
      P2_lds[i_qk0 + a][jh] = __floats2half2_rn(p0, p1);
      l_loc[a] = p0 + p1;
    }
#pragma unroll
    for (int off = JGROUP / 2; off > 0; off >>= 1) {
#pragma unroll
      for (int a = 0; a < QK_ROWS; ++a)
        l_loc[a] += __shfl_xor(l_loc[a], off);
    }
    if (jh == 0) {
#pragma unroll
      for (int a = 0; a < QK_ROWS; ++a) {
        const float m_old = M_lds[i_qk0 + a];
        const float alpha = __expf(m_old - m_new[a]);
        L_lds[i_qk0 + a] = L_lds[i_qk0 + a] * alpha + l_loc[a];
        alpha_lds[i_qk0 + a] = alpha;
        M_lds[i_qk0 + a] = m_new[a];
      }
    }
    __syncthreads();
    kv_stage_vt<TILE, THREADS>(pre_v, KV_lds, tid);
    // V(j+1) in flight across PV plus the next tile's QK/softmax
    if (more)
      kv_load<TILE, THREADS>(pre_v,
                             kv_row(vc, v_s0, v_s1, v_s2, block_table, kvh, tnext, nb_tbl, nb_cache, dbg),
                             v_s1, tid, tnext, INT_MAX);
    __syncthreads();

    // PV: acc = acc*alpha + P16 @ V (fp32 acc)
    const float alpha = alpha_lds[i_pv];
#pragma unroll
    for (int d = 0; d < PV_D; ++d) acc[d] *= alpha;
    const __half2* P2r = P2_lds[i_pv];
    const int dq = tid % PV_T;
    for (int m = 0; m < TILE / 2; ++m) {
      const __half2 pp = P2r[m];
      const uint2* Vt4 = reinterpret_cast<const uint2*>(KV_lds) +
                         m * (HEAD / 2);
#pragma unroll
      for (int d2 = 0; d2 < PV_D / 2; ++d2) {
        // d mapping matches the epilogue: acc[2*d2 + e] owns d
        // 2*dq + 2*PV_T*d2 + e; one uint2 covers the adjacent pair
        const uint2 v01 = Vt4[dq + PV_T * d2];
        acc[2 * d2] = dot2_f32_f16(
            pp, *reinterpret_cast<const __half2*>(&v01.x), acc[2 * d2]);
        acc[2 * d2 + 1] = dot2_f32_f16(
            pp, *reinterpret_cast<const __half2*>(&v01.y), acc[2 * d2 + 1]);
      }
    }
    __syncthreads();
    if (more) kv_stage<TILE, THREADS, KSTRIDE>(pre_k, KV_lds, tid);
    __syncthreads();
  }

  // phase B prologue: K(b_beg) into LDS, V(b_beg) in flight
  if (b_beg < seg_hi) {
    const int t0 = b_beg * TILE;
    kv_load<TILE, THREADS>(pre_k, kv_row(kc, k_s0, k_s1, k_s2, block_table, kvh, t0, nb_tbl, nb_cache, dbg),
                           k_s1, tid, t0, max_prefix);
    kv_stage<TILE, THREADS, KSTRIDE>(pre_k, KV_lds, tid);
    __syncthreads();
    kv_load<TILE, THREADS>(pre_v, kv_row(vc, v_s0, v_s1, v_s2, block_table, kvh, t0, nb_tbl, nb_cache, dbg),
                           v_s1, tid, t0, max_prefix);
  }

  for (int j = b_beg; j < seg_hi; ++j) {
    const int tbase = j * TILE;
    const int tnext = tbase + TILE;
    const bool more = tnext < seg_hi * TILE;

    if (more)
      kv_load<TILE, THREADS>(pre_k,
                             kv_row(kc, k_s0, k_s1, k_s2, block_table, kvh, tnext, nb_tbl, nb_cache, dbg),
                             k_s1, tid, tnext, max_prefix);
    // QK dot + causal/query masks
    float s[QK_ROWS][2];
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) s[a][0] = s[a][1] = 0.f;
    {
      const __half2* K2 = reinterpret_cast<const __half2*>(KV_lds);
      const __half2* Q2 = reinterpret_cast<const __half2*>(Q_lds);
#pragma unroll 8
      for (int d2 = 0; d2 < HEAD / 2; ++d2) {
        const __half2 k0 = K2[j_qk0 * (KSTRIDE / 2) + d2];
        const __half2 k1 = K2[(j_qk0 + 1) * (KSTRIDE / 2) + d2];
#pragma unroll
        for (int a = 0; a < QK_ROWS; ++a) {
          const __half2 qv = Q2[(i_qk0 + a) * (QSTRIDE / 2) + d2];
          s[a][0] = dot2_f32_f16(qv, k0, s[a][0]);
          s[a][1] = dot2_f32_f16(qv, k1, s[a][1]);
        }
      }
    }
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) {
      const int i = i_qk0 + a;
      const int tok = q_lo + i / GQA;
      const int abs_pos = context + tok;
      s[a][0] = (tok < q_len && tbase + j_qk0 <= abs_pos) ? s[a][0] * scale
                                                          : -INFINITY;
      s[a][1] = (tok < q_len && tbase + j_qk0 + 1 <= abs_pos) ? s[a][1] * scale
                                                              : -INFINITY;
    }

    float m_new[QK_ROWS], l_loc[QK_ROWS];
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) {
      m_new[a] = fmaxf(M_lds[i_qk0 + a], fmaxf(s[a][0], s[a][1]));
      l_loc[a] = 0.f;
    }
#pragma unroll
    for (int off = JGROUP / 2; off > 0; off >>= 1) {
#pragma unroll
      for (int a = 0; a < QK_ROWS; ++a)
        m_new[a] = fmaxf(m_new[a], __shfl_xor(m_new[a], off));
    }
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a)
      if (m_new[a] == -INFINITY) m_new[a] = 0.f;
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) {
      const float p0 = __expf(s[a][0] - m_new[a]);
      const float p1 = __expf(s[a][1] - m_new[a]);
      P2_lds[i_qk0 + a][jh] = __floats2half2_rn(p0, p1);
      l_loc[a] = p0 + p1;
    }
#pragma unroll
    for (int off = JGROUP / 2; off > 0; off >>= 1) {
#pragma unroll
      for (int a = 0; a < QK_ROWS; ++a)
        l_loc[a] += __shfl_xor(l_loc[a], off);
    }
    if (jh == 0) {
#pragma unroll
      for (int a = 0; a < QK_ROWS; ++a) {
        const float m_old = M_lds[i_qk0 + a];
        const float alpha = __expf(m_old - m_new[a]);
        L_lds[i_qk0 + a] = L_lds[i_qk0 + a] * alpha + l_loc[a];
        alpha_lds[i_qk0 + a] = alpha;
        M_lds[i_qk0 + a] = m_new[a];
      }
    }
    __syncthreads();
    kv_stage_vt<TILE, THREADS>(pre_v, KV_lds, tid);
    if (more)
      kv_load<TILE, THREADS>(pre_v,
                             kv_row(vc, v_s0, v_s1, v_s2, block_table, kvh, tnext, nb_tbl, nb_cache, dbg),
                             v_s1, tid, tnext, max_prefix);
    __syncthreads();

    const float alpha = alpha_lds[i_pv];
#pragma unroll
    for (int d = 0; d < PV_D; ++d) acc[d] *= alpha;
    const __half2* P2r = P2_lds[i_pv];
    const int dq = tid % PV_T;
    for (int m = 0; m < TILE / 2; ++m) {
      const __half2 pp = P2r[m];
      const uint2* Vt4 = reinterpret_cast<const uint2*>(KV_lds) +
                         m * (HEAD / 2);
#pragma unroll
      for (int d2 = 0; d2 < PV_D / 2; ++d2) {
        const uint2 v01 = Vt4[dq + PV_T * d2];
        acc[2 * d2] = dot2_f32_f16(
            pp, *reinterpret_cast<const __half2*>(&v01.x), acc[2 * d2]);
        acc[2 * d2 + 1] = dot2_f32_f16(
            pp, *reinterpret_cast<const __half2*>(&v01.y), acc[2 * d2 + 1]);
      }
    }
    __syncthreads();
    if (more) kv_stage<TILE, THREADS, KSTRIDE>(pre_k, KV_lds, tid);
    __syncthreads();
  }

  // epilogue into the fp32 segment workspace
  const int tok = q_lo + i_pv / GQA;
  if (tok < q_len) {
    const int head = kvh * GQA + i_pv % GQA;
    float* dst =
        segm_out + (((size_t)tok * QHEADS + head) * NSEG + segm) * HEAD;
    const int dq = tid % PV_T;
#pragma unroll
    for (int d2 = 0; d2 < PV_D / 2; ++d2) {
      const int ddx = 2 * dq + 2 * PV_T * d2;  // matches the PV mapping
      dst[ddx] = acc[2 * d2];
      dst[ddx + 1] = acc[2 * d2 + 1];
    }
    if (tid % PV_T == 0) {
      const size_t base = ((size_t)tok * QHEADS + head) * NSEG + segm;
      segm_max[base] = M_lds[i_pv];
      segm_expsum[base] = L_lds[i_pv];
    }
  }
}

}  // namespace

void gfx906_prefill_attn_scan(torch::Tensor q, torch::Tensor key_cache,
                              torch::Tensor value_cache,
                              torch::Tensor block_table,
                              torch::Tensor seq_lens,
                              torch::Tensor segm_output,
                              torch::Tensor segm_max,
                              torch::Tensor segm_expsum, double scale,
                              int64_t q_len, int64_t k_s0, int64_t k_s1,
                              int64_t k_s2, int64_t v_s0, int64_t v_s1,
                              int64_t v_s2, torch::Tensor dbg) {
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == at::kHalf, "kv cache must be fp16");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == QHEADS && q.size(2) == HEAD,
              "bad q shape");
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.size(1) == PAGE &&
                  key_cache.size(2) == KVHEADS && key_cache.size(3) == HEAD,
              "bad key cache layout");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(),
              "bad value cache layout");
  // K/V arrive as strided views of the paged cache (e.g. the interleaved
  // halves of a fused K/V buffer); only the head dim must be dense
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "kv cache head dim must be dense");
  // staging reads K/V rows with 16B vector loads: every row stride (in
  // halves) must be a multiple of 8 and the bases 16B-aligned (both the
  // contiguous and the fused/interleaved serve layouts satisfy this)
  TORCH_CHECK((k_s0 % 8 == 0) && (k_s1 % 8 == 0) && (k_s2 % 8 == 0) &&
                  (v_s0 % 8 == 0) && (v_s1 % 8 == 0) && (v_s2 % 8 == 0) &&
                  (reinterpret_cast<uintptr_t>(key_cache.data_ptr()) % 16 == 0) &&
                  (reinterpret_cast<uintptr_t>(value_cache.data_ptr()) % 16 == 0),
              "kv cache strides must be 16B-stageable");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.is_contiguous(),
              "bad block table");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.numel() == 1,
              "single-sequence only");
  TORCH_CHECK(dbg.scalar_type() == at::kInt && dbg.numel() >= 2,
              "dbg counter buffer");

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();


  constexpr int BM = 64, TILE = 32, TH = 512;
  constexpr int BLOCK_Q = BM / GQA;
  dim3 grid((q_len + BLOCK_Q - 1) / BLOCK_Q, KVHEADS, NSEG);
  gfx906_prefill_attn_kernel<BM, TILE, TH>
      <<<grid, dim3(TH), 0, stream>>>(
          reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
          block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
          segm_output.data_ptr<float>(), segm_max.data_ptr<float>(),
          segm_expsum.data_ptr<float>(), (float)scale, (int)q_len,
          (int)block_table.size(1), (int)key_cache.size(0), (int)k_s0, (int)k_s1, (int)k_s2, (int)v_s0, (int)v_s1, (int)v_s2,
          dbg.data_ptr<int>());
}

#endif  // USE_ROCM
