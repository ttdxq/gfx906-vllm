#pragma once

#if defined(USE_ROCM)
// Preserve the format-specific dot math while testing activation locality and
// the number of independent rows scheduled together.
template <typename scalar_t, int QK, int QI, typename block_t, int VDR,
          vec_dot_q_cuda_t DOT, int ROWS, bool SHARED_X>
static __global__ __launch_bounds__(ROWS * 64) void mul_mat_vec_q_schedule(
    const block_t* __restrict__ weights, const block_q8_1* __restrict__ x,
    scalar_t* __restrict__ out, int cols, int rows, int vecs, int stride) {
  const int padded_blocks = ((cols + 511) / 512 * 512) / QK8_1;
  extern __shared__ int staged[];
  if constexpr (SHARED_X) {
    for (int i = threadIdx.x; i < padded_blocks * 9; i += blockDim.x)
      staged[i] = ((const int*)x)[i];
    __syncthreads();
    x = (const block_q8_1*)staged;
  }
  const int row = blockIdx.x * ROWS + threadIdx.x / 64;
  if (row >= rows) return;
  const int lane = threadIdx.x & 63;
  const int nb = cols / QK;
  const int q = VDR * (lane % (QI / VDR));
  float sums[4] = {};
  for (int b = lane / (QI / VDR); b < nb; b += VDR * 64 / QI) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      if (v >= vecs) break;
      sums[v] += DOT(weights + (size_t)row * nb + b,
                     x + (size_t)v * padded_blocks + b * (QK / QK8_1), q);
    }
  }
#pragma unroll
  for (int mask = 32; mask > 0; mask >>= 1) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      if (v >= vecs) break;
      sums[v] += VLLM_SHFL_XOR_SYNC(sums[v], mask);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      if (v >= vecs) break;
      out[v * stride + row] = sums[v];
    }
  }
}

template <typename scalar_t, int QK, int QI, typename block_t, int VDR,
          vec_dot_q_cuda_t DOT>
static void launch_mmvq_schedule(const void* w, const void* x, scalar_t* out,
    int cols, int rows, int vecs, int stride, cudaStream_t stream,
    int row_tile, bool shared_x) {
  const int bytes = ((cols + 511) / 512 * 512) / 32 * 36;
#define SCHEDULE_LAUNCH(ROWS_, SHARED_) \
  mul_mat_vec_q_schedule<scalar_t, QK, QI, block_t, VDR, DOT, ROWS_, SHARED_> \
    <<<(rows + ROWS_ - 1) / ROWS_, ROWS_ * 64, SHARED_ ? bytes : 0, stream>>>( \
        (const block_t*)w, (const block_q8_1*)x, out, cols, rows, vecs, stride)
  if (row_tile == 8) {
    if (shared_x) { SCHEDULE_LAUNCH(8, true); }
    else { SCHEDULE_LAUNCH(8, false); }
  } else {
    if (shared_x) { SCHEDULE_LAUNCH(4, true); }
    else { SCHEDULE_LAUNCH(4, false); }
  }
#undef SCHEDULE_LAUNCH
}
#endif
