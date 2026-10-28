// Handwritten gfx906 prefill attention scan kernel.
//
// Replaces the Triton slim scanner on the single-sequence plain-causal
// fp16-KV head-256/GQA6 shape (runtime page size), writing the same
// fp32 split-KV segment workspace that reduce_segments combines. fp32 accumulation over
// fp16 inputs throughout (v_dot2_f32_f16: exact fp16 products, fp32
// accumulate), identical online-softmax semantics including the
// m_new == -inf guard.
//
// Kernel-level validation vs the Triton slim scanner on the tuning shapes
// (interleaved medians, single layer/chunk calls): 1.75-2.4x faster at
// 16k/64k/131k contexts with rel_rms vs an fp32 reference at the same
// 2.7e-4 level as Triton (summation-order-only differences).
//
// Structure: one 64-row q tile (BLOCK_Q=10 tokens x GQA6) resident in LDS;
// TILE=32 KV tiles staged through a padded (258-half) LDS buffer; QK is a
// 4x4 (row, col) block per thread on one of 4 interleaved d quarters (the
// quarters fold with two shuffles, the row max spans the 8 col blocks); V
// is transposed at staging into row pairs so PV runs dot2 over 4-row x
// 8-d thread blocks (strided d keeps the wave bank-conflict-free); 512
// threads = 4 waves/SIMD for latency hiding.
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
    int page_sz, int* dbg) {
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
  int page = tbase / page_sz;
  if ((unsigned)page >= (unsigned)nb_tbl) {
    if (dbg) atomicAdd(dbg, 1);
    page = 0;
  }
  int pbi = bt[page];
  if ((unsigned)pbi >= (unsigned)nb_cache) {
    if (dbg) atomicAdd(dbg + 1, 1);
    pbi = 0;
  }
  int tin = tbase - page * page_sz;
  if (tin >= page_sz) {
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
                           const float scale, const int q_len, const int bs,
                           const int nb_tbl, const int nb_cache, const int k_s0,
                           const int k_s1, const int k_s2, const int v_s0,
                           const int v_s1, const int v_s2, int* dbg) {
  constexpr int BLOCK_Q = BM / GQA;
  // QK tiling: a thread owns a QK_R x QK_C (row, col) block on one of QK_DB
  // interleaved d quarters, so every K/Q half2 is fetched once per row/col
  // block instead of once per row; the quarter lanes fold with two shuffles
  constexpr int QK_R = 4;
  constexpr int QK_C = 4;
  constexpr int QK_DB = 4;
  static_assert(BM / QK_R * (TILE / QK_C) * QK_DB == THREADS,
                "QK tiling mismatch");
  // PV tiling: a thread owns PV_ROWS rows x PV_DS d's, so a Vt2 half2 is
  // fetched once per row block instead of once per row
  constexpr int PV_ROWS = 4;
  constexpr int PV_DS = 8;
  constexpr int PV_NDB = HEAD / PV_DS;            // d phases = 32
  static_assert(BM / PV_ROWS * PV_NDB == THREADS, "PV tiling mismatch");

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
    const int rb_e = tid / PV_NDB;
    if (tid % PV_NDB == 0) {
#pragma unroll
      for (int r = 0; r < PV_ROWS; ++r) {
        const int i = rb_e * PV_ROWS + r;
        const int tok_e = qb * BLOCK_Q + i / GQA;
        if (tok_e < q_len) {
          const int head_e = kvh * GQA + i % GQA;
          const size_t base = ((size_t)tok_e * QHEADS + head_e) * NSEG + segm;
          segm_max[base] = -INFINITY;
          segm_expsum[base] = 0.f;
        }
      }
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

  // persistent fp32 accumulator: rows [r0, r0+PV_ROWS), d = db + 32*j
  const int r0 = (tid / PV_NDB) * PV_ROWS;
  const int db = tid % PV_NDB;
  float acc[PV_ROWS][PV_DS];
#pragma unroll
  for (int r = 0; r < PV_ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PV_DS; ++j) acc[r][j] = 0.f;

  // QK mapping: rows [i_qk0, +QK_R) x cols [j_qk0, +QK_C) x the interleaved
  // d quarter {qk_db, +QK_DB, ...}; the quarter lanes of a block are the 4
  // adjacent threads, the col blocks of a row sit 4 lanes apart
  const int qk_db = tid % QK_DB;
  const int qk_cb = (tid / QK_DB) % (TILE / QK_C);
  const int i_qk0 = (tid / (QK_DB * TILE / QK_C)) * QK_R;
  const int j_qk0 = qk_cb * QK_C;

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
    kv_load<TILE, THREADS>(pre_k, kv_row(kc, k_s0, k_s1, k_s2, block_table, kvh, t0, nb_tbl, nb_cache, bs, dbg),
                           k_s1, tid, t0, INT_MAX);
    kv_stage<TILE, THREADS, KSTRIDE>(pre_k, KV_lds, tid);
    __syncthreads();
    kv_load<TILE, THREADS>(pre_v, kv_row(vc, v_s0, v_s1, v_s2, block_table, kvh, t0, nb_tbl, nb_cache, bs, dbg),
                           v_s1, tid, t0, INT_MAX);
  }

  for (int j = seg_lo; j < a_end; ++j) {
    const int tbase = j * TILE;
    const int tnext = tbase + TILE;
    const bool more = tnext < a_end * TILE;

    // K(j+1) in flight across the whole QK/softmax/PV cycle
    if (more)
      kv_load<TILE, THREADS>(pre_k,
                             kv_row(kc, k_s0, k_s1, k_s2, block_table, kvh, tnext, nb_tbl, nb_cache, bs, dbg),
                             k_s1, tid, tnext, INT_MAX);

    // QK dot (fp32 acc over fp16 inputs): 4x4 (row, col) block on the
    // thread's interleaved d quarter; K and Q half2 fetches drop ~3x vs the
    // per-row mapping
    float s[QK_R][QK_C];
#pragma unroll
    for (int r = 0; r < QK_R; ++r)
#pragma unroll
      for (int c = 0; c < QK_C; ++c) s[r][c] = 0.f;
    {
      const __half2* K2 = reinterpret_cast<const __half2*>(KV_lds);
      const __half2* Q2 = reinterpret_cast<const __half2*>(Q_lds);
#pragma unroll 4
      for (int i = 0; i < (HEAD / 2) / QK_DB; ++i) {
        const int d2 = qk_db + QK_DB * i;
        __half2 qv[QK_R], kv[QK_C];
#pragma unroll
        for (int r = 0; r < QK_R; ++r)
          qv[r] = Q2[(i_qk0 + r) * (QSTRIDE / 2) + d2];
#pragma unroll
        for (int c = 0; c < QK_C; ++c)
          kv[c] = K2[(j_qk0 + c) * (KSTRIDE / 2) + d2];
#pragma unroll
        for (int r = 0; r < QK_R; ++r)
#pragma unroll
          for (int c = 0; c < QK_C; ++c)
            s[r][c] = dot2_f32_f16(qv[r], kv[c], s[r][c]);
      }
    }
    // fold the d quarters of the block; all 4 lanes end with the full sum
#pragma unroll
    for (int r = 0; r < QK_R; ++r)
#pragma unroll
      for (int c = 0; c < QK_C; ++c) {
        s[r][c] += __shfl_xor(s[r][c], 1);
        s[r][c] += __shfl_xor(s[r][c], 2);
      }
    if (qk_db == 0) {
#pragma unroll
      for (int r = 0; r < QK_R; ++r)
#pragma unroll
        for (int c = 0; c < QK_C; ++c) s[r][c] *= scale;

      // row stats on the summed lanes; the max spans the 8 col blocks four
      // lanes apart (shfl partners share qk_db == 0)
      float m_new[QK_R], l_loc[QK_R];
#pragma unroll
      for (int r = 0; r < QK_R; ++r) {
        m_new[r] = M_lds[i_qk0 + r];
#pragma unroll
        for (int c = 0; c < QK_C; ++c) m_new[r] = fmaxf(m_new[r], s[r][c]);
        l_loc[r] = 0.f;
      }
#pragma unroll
      for (int off = QK_DB; off < (TILE / QK_C) * QK_DB; off <<= 1) {
#pragma unroll
        for (int r = 0; r < QK_R; ++r)
          m_new[r] = fmaxf(m_new[r], __shfl_xor(m_new[r], off));
      }
#pragma unroll
      for (int r = 0; r < QK_R; ++r)
        if (m_new[r] == -INFINITY) m_new[r] = 0.f;  // Triton's guard
#pragma unroll
      for (int r = 0; r < QK_R; ++r) {
#pragma unroll
        for (int c = 0; c < QK_C; c += 2) {
          const float p0 = __expf(s[r][c] - m_new[r]);
          const float p1 = __expf(s[r][c + 1] - m_new[r]);
          P2_lds[i_qk0 + r][j_qk0 / 2 + c / 2] = __floats2half2_rn(p0, p1);
          l_loc[r] += p0 + p1;
        }
      }
#pragma unroll
      for (int off = QK_DB; off < (TILE / QK_C) * QK_DB; off <<= 1) {
#pragma unroll
        for (int r = 0; r < QK_R; ++r)
          l_loc[r] += __shfl_xor(l_loc[r], off);
      }
      if (qk_cb == 0) {
#pragma unroll
        for (int r = 0; r < QK_R; ++r) {
          const float m_old = M_lds[i_qk0 + r];
          const float alpha = __expf(m_old - m_new[r]);
          L_lds[i_qk0 + r] = L_lds[i_qk0 + r] * alpha + l_loc[r];
          alpha_lds[i_qk0 + r] = alpha;
          M_lds[i_qk0 + r] = m_new[r];
        }
      }
    }
    __syncthreads();
    kv_stage_vt<TILE, THREADS>(pre_v, KV_lds, tid);
    // V(j+1) in flight across PV plus the next tile's QK/softmax
    if (more)
      kv_load<TILE, THREADS>(pre_v,
                             kv_row(vc, v_s0, v_s1, v_s2, block_table, kvh, tnext, nb_tbl, nb_cache, bs, dbg),
                             v_s1, tid, tnext, INT_MAX);
    __syncthreads();

    // PV: acc = acc*alpha + P16 @ V (fp32 acc)
#pragma unroll
    for (int r = 0; r < PV_ROWS; ++r) {
      const float alpha = alpha_lds[r0 + r];
#pragma unroll
      for (int j = 0; j < PV_DS; ++j) acc[r][j] *= alpha;
    }
    // each Vt2 half2 feeds PV_ROWS rows at once (LDS bytes drop ~2.8x);
    // the d stride 32 spreads the 32 d phases of a wave over distinct banks
    const __half2* Vt2 = reinterpret_cast<const __half2*>(KV_lds);
#pragma unroll 4
    for (int m = 0; m < TILE / 2; ++m) {
      __half2 pp[PV_ROWS];
#pragma unroll
      for (int r = 0; r < PV_ROWS; ++r) pp[r] = P2_lds[r0 + r][m];
      const __half2* Vm = Vt2 + m * HEAD + db;
#pragma unroll
      for (int j = 0; j < PV_DS; ++j) {
        const __half2 v = Vm[32 * j];
#pragma unroll
        for (int r = 0; r < PV_ROWS; ++r)
          acc[r][j] = dot2_f32_f16(pp[r], v, acc[r][j]);
      }
    }
    __syncthreads();
    if (more) kv_stage<TILE, THREADS, KSTRIDE>(pre_k, KV_lds, tid);
    __syncthreads();
  }

  // phase B prologue: K(b_beg) into LDS, V(b_beg) in flight
  if (b_beg < seg_hi) {
    const int t0 = b_beg * TILE;
    kv_load<TILE, THREADS>(pre_k, kv_row(kc, k_s0, k_s1, k_s2, block_table, kvh, t0, nb_tbl, nb_cache, bs, dbg),
                           k_s1, tid, t0, max_prefix);
    kv_stage<TILE, THREADS, KSTRIDE>(pre_k, KV_lds, tid);
    __syncthreads();
    kv_load<TILE, THREADS>(pre_v, kv_row(vc, v_s0, v_s1, v_s2, block_table, kvh, t0, nb_tbl, nb_cache, bs, dbg),
                           v_s1, tid, t0, max_prefix);
  }

  for (int j = b_beg; j < seg_hi; ++j) {
    const int tbase = j * TILE;
    const int tnext = tbase + TILE;
    const bool more = tnext < seg_hi * TILE;

    if (more)
      kv_load<TILE, THREADS>(pre_k,
                             kv_row(kc, k_s0, k_s1, k_s2, block_table, kvh, tnext, nb_tbl, nb_cache, bs, dbg),
                             k_s1, tid, tnext, max_prefix);
    // QK dot + causal/query masks (same 4x4 block / d-quarter mapping)
    float s[QK_R][QK_C];
#pragma unroll
    for (int r = 0; r < QK_R; ++r)
#pragma unroll
      for (int c = 0; c < QK_C; ++c) s[r][c] = 0.f;
    {
      const __half2* K2 = reinterpret_cast<const __half2*>(KV_lds);
      const __half2* Q2 = reinterpret_cast<const __half2*>(Q_lds);
#pragma unroll 4
      for (int i = 0; i < (HEAD / 2) / QK_DB; ++i) {
        const int d2 = qk_db + QK_DB * i;
        __half2 qv[QK_R], kv[QK_C];
#pragma unroll
        for (int r = 0; r < QK_R; ++r)
          qv[r] = Q2[(i_qk0 + r) * (QSTRIDE / 2) + d2];
#pragma unroll
        for (int c = 0; c < QK_C; ++c)
          kv[c] = K2[(j_qk0 + c) * (KSTRIDE / 2) + d2];
#pragma unroll
        for (int r = 0; r < QK_R; ++r)
#pragma unroll
          for (int c = 0; c < QK_C; ++c)
            s[r][c] = dot2_f32_f16(qv[r], kv[c], s[r][c]);
      }
    }
#pragma unroll
    for (int r = 0; r < QK_R; ++r)
#pragma unroll
      for (int c = 0; c < QK_C; ++c) {
        s[r][c] += __shfl_xor(s[r][c], 1);
        s[r][c] += __shfl_xor(s[r][c], 2);
      }
    if (qk_db == 0) {
#pragma unroll
      for (int r = 0; r < QK_R; ++r) {
        const int tok = q_lo + (i_qk0 + r) / GQA;
        const int abs_pos = context + tok;
#pragma unroll
        for (int c = 0; c < QK_C; ++c)
          s[r][c] = (tok < q_len && tbase + j_qk0 + c <= abs_pos)
                        ? s[r][c] * scale
                        : -INFINITY;
      }

      float m_new[QK_R], l_loc[QK_R];
#pragma unroll
      for (int r = 0; r < QK_R; ++r) {
        m_new[r] = M_lds[i_qk0 + r];
#pragma unroll
        for (int c = 0; c < QK_C; ++c) m_new[r] = fmaxf(m_new[r], s[r][c]);
        l_loc[r] = 0.f;
      }
#pragma unroll
      for (int off = QK_DB; off < (TILE / QK_C) * QK_DB; off <<= 1) {
#pragma unroll
        for (int r = 0; r < QK_R; ++r)
          m_new[r] = fmaxf(m_new[r], __shfl_xor(m_new[r], off));
      }
#pragma unroll
      for (int r = 0; r < QK_R; ++r)
        if (m_new[r] == -INFINITY) m_new[r] = 0.f;  // Triton's guard
#pragma unroll
      for (int r = 0; r < QK_R; ++r) {
#pragma unroll
        for (int c = 0; c < QK_C; c += 2) {
          const float p0 = __expf(s[r][c] - m_new[r]);
          const float p1 = __expf(s[r][c + 1] - m_new[r]);
          P2_lds[i_qk0 + r][j_qk0 / 2 + c / 2] = __floats2half2_rn(p0, p1);
          l_loc[r] += p0 + p1;
        }
      }
#pragma unroll
      for (int off = QK_DB; off < (TILE / QK_C) * QK_DB; off <<= 1) {
#pragma unroll
        for (int r = 0; r < QK_R; ++r)
          l_loc[r] += __shfl_xor(l_loc[r], off);
      }
      if (qk_cb == 0) {
#pragma unroll
        for (int r = 0; r < QK_R; ++r) {
          const float m_old = M_lds[i_qk0 + r];
          const float alpha = __expf(m_old - m_new[r]);
          L_lds[i_qk0 + r] = L_lds[i_qk0 + r] * alpha + l_loc[r];
          alpha_lds[i_qk0 + r] = alpha;
          M_lds[i_qk0 + r] = m_new[r];
        }
      }
    }
    __syncthreads();
    kv_stage_vt<TILE, THREADS>(pre_v, KV_lds, tid);
    if (more)
      kv_load<TILE, THREADS>(pre_v,
                             kv_row(vc, v_s0, v_s1, v_s2, block_table, kvh, tnext, nb_tbl, nb_cache, bs, dbg),
                             v_s1, tid, tnext, max_prefix);
    __syncthreads();

#pragma unroll
    for (int r = 0; r < PV_ROWS; ++r) {
      const float alpha = alpha_lds[r0 + r];
#pragma unroll
      for (int j = 0; j < PV_DS; ++j) acc[r][j] *= alpha;
    }
    const __half2* Vt2 = reinterpret_cast<const __half2*>(KV_lds);
#pragma unroll 4
    for (int m = 0; m < TILE / 2; ++m) {
      __half2 pp[PV_ROWS];
#pragma unroll
      for (int r = 0; r < PV_ROWS; ++r) pp[r] = P2_lds[r0 + r][m];
      const __half2* Vm = Vt2 + m * HEAD + db;
#pragma unroll
      for (int j = 0; j < PV_DS; ++j) {
        const __half2 v = Vm[32 * j];
#pragma unroll
        for (int r = 0; r < PV_ROWS; ++r)
          acc[r][j] = dot2_f32_f16(pp[r], v, acc[r][j]);
      }
    }
    __syncthreads();
    if (more) kv_stage<TILE, THREADS, KSTRIDE>(pre_k, KV_lds, tid);
    __syncthreads();
  }

  // epilogue into the fp32 segment workspace
#pragma unroll
  for (int r = 0; r < PV_ROWS; ++r) {
    const int i = r0 + r;
    const int tok = q_lo + i / GQA;
    if (tok < q_len) {
      const int head = kvh * GQA + i % GQA;
      float* dst =
          segm_out + (((size_t)tok * QHEADS + head) * NSEG + segm) * HEAD;
#pragma unroll
      for (int j = 0; j < PV_DS; ++j) dst[db + 32 * j] = acc[r][j];
      if (db == 0) {
        const size_t base = ((size_t)tok * QHEADS + head) * NSEG + segm;
        segm_max[base] = M_lds[i];
        segm_expsum[base] = L_lds[i];
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Handwritten gfx906 speculative-decode attention scan kernel.
//
// Serves the spec-decode shape (verify + draft catch-up forwards): one query
// token per 6-row GQA block, q_len tokens in flight, single sequence, fp16
// KV on the head-256/GQA6/800-page shape. Writes the same fp32 split-KV
// segment workspace the Triton path produces, in exactly the layout
// reduce_segments recomputes (tiles_per_segment = cdiv(seq_len,
// nseg * 16); inactive segments early-return without touching the rows
// that reduce masks out, tokens whose causal bound ends inside the segment
// store the untouched M=-inf/L=1/acc=0 state so stale rows never merge).
//
// The Triton scanner for this shape is launch-config-bound at ~35-90 GB/s
// of KV scan: the block-table gather defeats its software pipelining and
// every stage counts an extra LDS round trip. Here the block is 2 warps;
// each lane owns 8 dims of the token's 6 query rows, streams K/V rows as
// one 16B load each with a one-token K prefetch, folds the QK partials
// with a 5-step butterfly, and keeps a private online-softmax state over
// its half of the segment's tokens. The two warp states merge through 6KB
// of LDS in the epilogue. No LDS staging of KV: at 6 rows the arithmetic
// intensity cannot pay for it, and per-lane uint4 slices keep enough loads
// in flight through occupancy (4+ resident 64-thread blocks per CU).
// ---------------------------------------------------------------------------

namespace {

constexpr int SPEC_TILE = 16;  // layout tile; must equal reduce_segments' TILE_SIZE

// runtime-block-size variant of kv_row for the spec scanner: the engine's
// paged cache is not fixed to 800-token pages (e.g. 832 in production),
// and the spec kernel's per-token row loads have no page-size constraint
__device__ __forceinline__ const __half* kv_row_bs(
    const __half* base, int s0, int s1, int s2, const int* __restrict__ bt,
    int kvh, int tbase, int bs, int nb_tbl, int nb_cache, int* dbg) {
  int page = tbase / bs;
  if ((unsigned)page >= (unsigned)nb_tbl) {
    if (dbg) atomicAdd(dbg, 1);
    page = 0;
  }
  int pbi = bt[page];
  if ((unsigned)pbi >= (unsigned)nb_cache) {
    if (dbg) atomicAdd(dbg + 1, 1);
    pbi = 0;
  }
  int tin = tbase - page * bs;
  if (tin >= bs) {
    if (dbg) atomicAdd(dbg, 1);
    tin = 0;
  }
  return base + (size_t)pbi * s0 + (size_t)tin * s1 + (size_t)kvh * s2;
}

__global__ void __launch_bounds__(64)
gfx906_spec_attn_kernel(const __half* __restrict__ q,  // [q_len, 24, 256]
                        const __half* __restrict__ kc,
                        const __half* __restrict__ vc,
                        const int* __restrict__ block_table,
                        const int* __restrict__ seq_lens,  // [1]
                        float* __restrict__ segm_out,   // [q_len, 24, nseg, 256]
                        float* __restrict__ segm_max,   // [q_len, 24, nseg]
                        float* __restrict__ segm_expsum,
                        const float scale, const int q_len, const int nseg,
                        const int bs, const int nb_tbl, const int nb_cache,
                        const int k_s0, const int k_s1, const int k_s2,
                        const int v_s0, const int v_s1, const int v_s2,
                        int* dbg) {
  constexpr int NWARP = 2;
  const int tok = blockIdx.x;
  const int kvh = blockIdx.y;
  const int segm = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const int seq_len = seq_lens[0];
  // segment partition must match reduce_segments exactly
  const int tps = (seq_len + nseg * SPEC_TILE - 1) / (nseg * SPEC_TILE);
  const int seg_lo = segm * tps * SPEC_TILE;
  if (seg_lo >= seq_len) return;  // inactive; reduce masks the segment out
  const int seg_hi = min((segm + 1) * tps * SPEC_TILE, seq_len);
  // this token's causal bound; negative during seq_lens=1 graph capture,
  // which just empties every range (the launch is what gets recorded)
  const int context = seq_len - q_len;
  const int hi = min(min(context + tok + 1, seq_len), seg_hi);

  // 6 query rows of this token for kv head kvh, 8 dims per lane
  __half2 qr[GQA][4];
#pragma unroll
  for (int r = 0; r < GQA; ++r) {
    const uint4 v4 = *reinterpret_cast<const uint4*>(
        q + ((size_t)tok * QHEADS + kvh * GQA + r) * HEAD + lane * 8);
    qr[r][0] = reinterpret_cast<const __half2&>(v4.x);
    qr[r][1] = reinterpret_cast<const __half2&>(v4.y);
    qr[r][2] = reinterpret_cast<const __half2&>(v4.z);
    qr[r][3] = reinterpret_cast<const __half2&>(v4.w);
  }

  float M[GQA], L[GQA], acc[GQA][8];
#pragma unroll
  for (int r = 0; r < GQA; ++r) {
    M[r] = -INFINITY;
    L[r] = 1.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[r][j] = 0.f;
  }

  // warp-private scan: the NWARP warps take complementary tokens of
  // [seg_lo, hi). Both K and V of the next warp-local token stay in flight
  // in registers across a full QK/softmax/PV cycle, so each lane keeps
  // four 16B loads outstanding and the DRAM latency hides behind compute.
  int t = seg_lo + warp;
  uint4 k16, v16;
  bool have = t < hi;
  if (have) {
    const __half* krow =
        kv_row_bs(kc, k_s0, k_s1, k_s2, block_table, kvh, t, bs, nb_tbl, nb_cache, dbg);
    k16 = *reinterpret_cast<const uint4*>(krow + lane * 8);
    const __half* vrow =
        kv_row_bs(vc, v_s0, v_s1, v_s2, block_table, kvh, t, bs, nb_tbl, nb_cache, dbg);
    v16 = *reinterpret_cast<const uint4*>(vrow + lane * 8);
  }
  while (have) {
    const int tn = t + NWARP;
    const bool have_next = tn < hi;
    uint4 kn16, vn16;
    if (have_next) {
      const __half* knrow =
          kv_row_bs(kc, k_s0, k_s1, k_s2, block_table, kvh, tn, bs, nb_tbl, nb_cache, dbg);
      kn16 = *reinterpret_cast<const uint4*>(knrow + lane * 8);
      const __half* vnrow =
          kv_row_bs(vc, v_s0, v_s1, v_s2, block_table, kvh, tn, bs, nb_tbl, nb_cache, dbg);
      vn16 = *reinterpret_cast<const uint4*>(vnrow + lane * 8);
    }

    {
      const __half2* kh2 = reinterpret_cast<const __half2*>(&k16);
      float s[GQA];
#pragma unroll
      for (int r = 0; r < GQA; ++r) {
        s[r] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          s[r] = dot2_f32_f16(qr[r][j], kh2[j], s[r]);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int r = 0; r < GQA; ++r) s[r] += __shfl_xor(s[r], off);

      // softmax update: identical on every lane of the warp
      const __half2* vh2 = reinterpret_cast<const __half2*>(&v16);
#pragma unroll
      for (int r = 0; r < GQA; ++r) {
        const float score = scale * s[r];
        float m_new = fmaxf(M[r], score);
        if (m_new == -INFINITY) m_new = 0.f;  // Triton's guard
        const float alpha = __expf(M[r] - m_new);
        const float p = __expf(score - m_new);
        L[r] = L[r] * alpha + p;
        M[r] = m_new;
        const float2 f0 = __half22float2(vh2[0]);
        const float2 f1 = __half22float2(vh2[1]);
        const float2 f2 = __half22float2(vh2[2]);
        const float2 f3 = __half22float2(vh2[3]);
        acc[r][0] = acc[r][0] * alpha + p * f0.x;
        acc[r][1] = acc[r][1] * alpha + p * f0.y;
        acc[r][2] = acc[r][2] * alpha + p * f1.x;
        acc[r][3] = acc[r][3] * alpha + p * f1.y;
        acc[r][4] = acc[r][4] * alpha + p * f2.x;
        acc[r][5] = acc[r][5] * alpha + p * f2.y;
        acc[r][6] = acc[r][6] * alpha + p * f3.x;
        acc[r][7] = acc[r][7] * alpha + p * f3.y;
      }
    }

    k16 = kn16;
    v16 = vn16;
    t = tn;
    have = have_next;
  }

  // merge the two warp states through LDS: warp 1 posts its partial state,
  // warp 0 folds it into its own and writes the segment row
  __shared__ float sm_M[GQA], sm_L[GQA];
  __shared__ float sm_acc[GQA][HEAD];
  if (warp == 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < GQA; ++r) {
        sm_M[r] = M[r];
        sm_L[r] = L[r];
      }
    }
#pragma unroll
    for (int r = 0; r < GQA; ++r) {
#pragma unroll
      for (int j = 0; j < 8; ++j) sm_acc[r][lane * 8 + j] = acc[r][j];
    }
  }
  __syncthreads();
  if (warp != 0) return;

#pragma unroll
  for (int r = 0; r < GQA; ++r) {
    const int head = kvh * GQA + r;
    const size_t base = ((size_t)tok * QHEADS + head) * nseg + segm;
    float* dst = segm_out + base * HEAD;
    const float mm = fmaxf(M[r], sm_M[r]);
    if (mm == -INFINITY) {
      // nothing scanned inside this token's causal range: post the
      // untouched state exactly as the Triton scanner stores it
      segm_max[base] = -INFINITY;
      segm_expsum[base] = 1.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[lane * 8 + j] = 0.f;
      continue;
    }
    const float a1 = __expf(sm_M[r] - mm);
    const float a0 = __expf(M[r] - mm);
    segm_max[base] = mm;
    segm_expsum[base] = L[r] * a0 + sm_L[r] * a1;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      dst[lane * 8 + j] = acc[r][j] * a0 + sm_acc[r][lane * 8 + j] * a1;
  }
}

}  // namespace

void gfx906_spec_attn_scan(torch::Tensor q, torch::Tensor key_cache,
                           torch::Tensor value_cache, torch::Tensor block_table,
                           torch::Tensor seq_lens, torch::Tensor segm_output,
                           torch::Tensor segm_max, torch::Tensor segm_expsum,
                           double scale, int64_t q_len, int64_t nseg,
                           int64_t bs, int64_t k_s0, int64_t k_s1,
                           int64_t k_s2, int64_t v_s0, int64_t v_s1,
                           int64_t v_s2, torch::Tensor dbg) {
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == at::kHalf, "kv cache must be fp16");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == QHEADS && q.size(2) == HEAD,
              "bad q shape");
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.size(2) == KVHEADS &&
                  key_cache.size(3) == HEAD,
              "bad key cache layout");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(),
              "bad value cache layout");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "kv cache head dim must be dense");
  // per-lane 16B row slices need 16B-multiple strides and bases
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

  dim3 grid((unsigned)q_len, KVHEADS, (unsigned)nseg);
  gfx906_spec_attn_kernel<<<grid, dim3(64), 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
      segm_output.data_ptr<float>(), segm_max.data_ptr<float>(),
      segm_expsum.data_ptr<float>(), (float)scale, (int)q_len, (int)nseg,
      (int)bs, (int)block_table.size(1), (int)key_cache.size(0), (int)k_s0, (int)k_s1,
      (int)k_s2, (int)v_s0, (int)v_s1, (int)v_s2, dbg.data_ptr<int>());
}

void gfx906_prefill_attn_scan(torch::Tensor q, torch::Tensor key_cache,
                              torch::Tensor value_cache,
                              torch::Tensor block_table,
                              torch::Tensor seq_lens,
                              torch::Tensor segm_output,
                              torch::Tensor segm_max,
                              torch::Tensor segm_expsum, double scale,
                              int64_t q_len, int64_t bs, int64_t k_s0,
                              int64_t k_s1, int64_t k_s2, int64_t v_s0,
                              int64_t v_s1, int64_t v_s2, torch::Tensor dbg) {
  constexpr int BM = 64, TILE = 32, TH = 512;
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == at::kHalf, "kv cache must be fp16");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == QHEADS && q.size(2) == HEAD,
              "bad q shape");
  // the paged cache block size is a runtime parameter (800- and 832-token
  // pages are both in service); a 32-token KV tile must never straddle a page
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.size(1) == bs &&
                  key_cache.size(2) == KVHEADS && key_cache.size(3) == HEAD,
              "bad key cache layout");
  TORCH_CHECK(bs > 0 && bs % TILE == 0,
              "cache block size must be a positive multiple of the 32-token "
              "KV tile");
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

  constexpr int BLOCK_Q = BM / GQA;
  dim3 grid((q_len + BLOCK_Q - 1) / BLOCK_Q, KVHEADS, NSEG);
  gfx906_prefill_attn_kernel<BM, TILE, TH>
      <<<grid, dim3(TH), 0, stream>>>(
          reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
          block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
          segm_output.data_ptr<float>(), segm_max.data_ptr<float>(),
          segm_expsum.data_ptr<float>(), (float)scale, (int)q_len, (int)bs,
          (int)block_table.size(1), (int)key_cache.size(0), (int)k_s0, (int)k_s1, (int)k_s2, (int)v_s0, (int)v_s1, (int)v_s2,
          dbg.data_ptr<int>());
}

#endif  // USE_ROCM
