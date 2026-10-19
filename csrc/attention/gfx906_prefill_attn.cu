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

#ifdef USE_ROCM

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>

namespace {

constexpr int QHEADS = 24;
constexpr int KVHEADS = 4;
constexpr int GQA = 6;
constexpr int HEAD = 256;
constexpr int PAGE = 800;
constexpr int NSEG = 2;

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
                           const float scale, const int q_len, const int k_s0,
                           const int k_s1, const int k_s2, const int v_s0,
                           const int v_s1, const int v_s2) {
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
  constexpr int PSTRIDE = TILE + 2;
  __shared__ __half Q_lds[BM * QSTRIDE];
  __shared__ __half KV_lds[TILE * KSTRIDE];
  __shared__ __half P_lds[BM * PSTRIDE];
  __shared__ float M_lds[BM], L_lds[BM], alpha_lds[BM];

  // stage Q (invalid rows read as 0) and init softmax state
  for (int idx = tid; idx < BM * (HEAD / 2); idx += THREADS) {
    const int m = idx / (HEAD / 2), d2 = idx % (HEAD / 2);
    const int tok = q_lo + m / GQA;
    __half2 val;
    if (tok < q_len) {
      const int head = kvh * GQA + m % GQA;
      val = *reinterpret_cast<const __half2*>(
          q + ((size_t)tok * QHEADS + head) * HEAD + d2 * 2);
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

  for (int j = seg_lo; j < a_end; ++j) {
    // stage K tile (unmasked)
    const int tbase = j * TILE;
    const int page = tbase / PAGE;
    const int pbi = block_table[page];
    const __half* krow = kc + (size_t)pbi * k_s0 +
                         (size_t)(tbase - page * PAGE) * k_s1 +
                         (size_t)kvh * k_s2;
    for (int idx = tid; idx < TILE * (HEAD / 2); idx += THREADS) {
      const int r = idx / (HEAD / 2);
      reinterpret_cast<__half2*>(KV_lds)
          [r * (KSTRIDE / 2) + idx % (HEAD / 2)] =
              reinterpret_cast<const __half2*>(
                  krow + (size_t)r * k_s1)[idx % (HEAD / 2)];
    }
    __syncthreads();

    // QK dot (fp32 acc over fp16 inputs)
    float s[QK_ROWS][2];
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) s[a][0] = s[a][1] = 0.f;
    {
      const __half2* K2 = reinterpret_cast<const __half2*>(KV_lds);
      const __half2* Q2 = reinterpret_cast<const __half2*>(Q_lds);
#pragma unroll 8
      for (int d2 = 0; d2 < HEAD / 2; ++d2) {
        const float2 k0 = __half22float2(K2[j_qk0 * (KSTRIDE / 2) + d2]);
        const float2 k1 = __half22float2(K2[(j_qk0 + 1) * (KSTRIDE / 2) + d2]);
#pragma unroll
        for (int a = 0; a < QK_ROWS; ++a) {
          const float2 qv =
              __half22float2(Q2[(i_qk0 + a) * (QSTRIDE / 2) + d2]);
          // single-term statements so every fp-contract fold selects
          // v_fma_mix_f32 (a two-term chain leaves the first as mul+add)
          s[a][0] += qv.x * k0.x;
          s[a][0] += qv.y * k0.y;
          s[a][1] += qv.x * k1.x;
          s[a][1] += qv.y * k1.y;
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
      P_lds[(i_qk0 + a) * PSTRIDE + j_qk0] = __float2half(p0);
      P_lds[(i_qk0 + a) * PSTRIDE + j_qk0 + 1] = __float2half(p1);
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

    // stage V tile into the same LDS buffer
    const __half* vrow = vc + (size_t)pbi * v_s0 +
                         (size_t)(tbase - page * PAGE) * v_s1 +
                         (size_t)kvh * v_s2;
    for (int idx = tid; idx < TILE * (HEAD / 2); idx += THREADS) {
      const int r = idx / (HEAD / 2);
      reinterpret_cast<__half2*>(KV_lds)
          [r * (KSTRIDE / 2) + idx % (HEAD / 2)] =
              reinterpret_cast<const __half2*>(
                  vrow + (size_t)r * v_s1)[idx % (HEAD / 2)];
    }
    __syncthreads();

    // PV: acc = acc*alpha + P16 @ V (fp32 acc)
    const float alpha = alpha_lds[i_pv];
#pragma unroll
    for (int d = 0; d < PV_D; ++d) acc[d] *= alpha;
    for (int r = 0; r < TILE; ++r) {
      // interleaved d mapping (4-half2 stride): consecutive threads of a
      // row hit consecutive banks instead of one bank 4 ways
      const float p = __half2float(P_lds[i_pv * PSTRIDE + r]);
      const __half2* V2 = reinterpret_cast<const __half2*>(KV_lds) +
                          r * (KSTRIDE / 2) + (tid % PV_T);
#pragma unroll
      for (int d2 = 0; d2 < PV_D / 2; ++d2) {
        // interleaved d mapping: thread dq owns h2 {dq + PV_T*d2}; a fixed
        // stride of 4 only covers half the row when PV_T != 4
        const float2 vv = __half22float2(V2[PV_T * d2]);
        acc[2 * d2] += p * vv.x;
        acc[2 * d2 + 1] += p * vv.y;
      }
    }
    __syncthreads();
  }

  for (int j = b_beg; j < seg_hi; ++j) {
    // stage K tile (masked by max_prefix)
    const int tbase = j * TILE;
    const int page = tbase / PAGE;
    const int pbi = block_table[page];
    const __half* krow = kc + (size_t)pbi * k_s0 +
                         (size_t)(tbase - page * PAGE) * k_s1 +
                         (size_t)kvh * k_s2;
    for (int idx = tid; idx < TILE * (HEAD / 2); idx += THREADS) {
      const int r = idx / (HEAD / 2);
      if (tbase + r < max_prefix) {
        reinterpret_cast<__half2*>(KV_lds)
            [r * (KSTRIDE / 2) + idx % (HEAD / 2)] =
                reinterpret_cast<const __half2*>(
                    krow + (size_t)r * k_s1)[idx % (HEAD / 2)];
      } else {
        // must use the padded index: a raw flat idx aliases real rows
        reinterpret_cast<__half2*>(KV_lds)
            [r * (KSTRIDE / 2) + idx % (HEAD / 2)] =
                __floats2half2_rn(0.f, 0.f);
      }
    }
    __syncthreads();

    // QK dot + causal/query masks
    float s[QK_ROWS][2];
#pragma unroll
    for (int a = 0; a < QK_ROWS; ++a) s[a][0] = s[a][1] = 0.f;
    {
      const __half2* K2 = reinterpret_cast<const __half2*>(KV_lds);
      const __half2* Q2 = reinterpret_cast<const __half2*>(Q_lds);
#pragma unroll 8
      for (int d2 = 0; d2 < HEAD / 2; ++d2) {
        const float2 k0 = __half22float2(K2[j_qk0 * (KSTRIDE / 2) + d2]);
        const float2 k1 = __half22float2(K2[(j_qk0 + 1) * (KSTRIDE / 2) + d2]);
#pragma unroll
        for (int a = 0; a < QK_ROWS; ++a) {
          const float2 qv =
              __half22float2(Q2[(i_qk0 + a) * (QSTRIDE / 2) + d2]);
          s[a][0] += qv.x * k0.x;
          s[a][0] += qv.y * k0.y;
          s[a][1] += qv.x * k1.x;
          s[a][1] += qv.y * k1.y;
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
      P_lds[(i_qk0 + a) * PSTRIDE + j_qk0] = __float2half(p0);
      P_lds[(i_qk0 + a) * PSTRIDE + j_qk0 + 1] = __float2half(p1);
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

    const __half* vrow = vc + (size_t)pbi * v_s0 +
                         (size_t)(tbase - page * PAGE) * v_s1 +
                         (size_t)kvh * v_s2;
    for (int idx = tid; idx < TILE * (HEAD / 2); idx += THREADS) {
      const int r = idx / (HEAD / 2);
      if (tbase + r < max_prefix) {
        reinterpret_cast<__half2*>(KV_lds)
            [r * (KSTRIDE / 2) + idx % (HEAD / 2)] =
                reinterpret_cast<const __half2*>(
                    vrow + (size_t)r * v_s1)[idx % (HEAD / 2)];
      } else {
        reinterpret_cast<__half2*>(KV_lds)
            [r * (KSTRIDE / 2) + idx % (HEAD / 2)] =
                __floats2half2_rn(0.f, 0.f);
      }
    }
    __syncthreads();

    const float alpha = alpha_lds[i_pv];
#pragma unroll
    for (int d = 0; d < PV_D; ++d) acc[d] *= alpha;
    for (int r = 0; r < TILE; ++r) {
      const float p = __half2float(P_lds[i_pv * PSTRIDE + r]);
      const __half2* V2 = reinterpret_cast<const __half2*>(KV_lds) +
                          r * (KSTRIDE / 2) + (tid % PV_T);
#pragma unroll
      for (int d2 = 0; d2 < PV_D / 2; ++d2) {
        // interleaved d mapping: thread dq owns h2 {dq + PV_T*d2}; a fixed
        // stride of 4 only covers half the row when PV_T != 4
        const float2 vv = __half22float2(V2[PV_T * d2]);
        acc[2 * d2] += p * vv.x;
        acc[2 * d2 + 1] += p * vv.y;
      }
    }
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
                              int64_t v_s2) {
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
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.is_contiguous(),
              "bad block table");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.numel() == 1,
              "single-sequence only");

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
          (int)k_s0, (int)k_s1, (int)k_s2, (int)v_s0, (int)v_s1, (int)v_s2);
}

#endif  // USE_ROCM
