#include <cstdint>

/* Adapted from ./csrc/quantization/gguf/mmq.cuh
   based on ./vllm/model_executor/layers/fused_moe/fused_moe.py */
template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_mul_mat_cuda_t vec_dot, int storage_qk = qk>
static __device__ __forceinline__ void moe_q(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k) {
  const int blocks_per_row_x = ncols_x / storage_qk;
  const int blocks_per_col_y = nrows_y / QK8_1;
  const int blocks_per_warp = WARP_SIZE_GGUF / qi * (qk / storage_qk);

  const int ncols_dst = ncols_y * top_k;

  const auto row_dst_0 = blockIdx.x * mmq_y;
  const int& row_x_0 = row_dst_0;

  const auto col_dst_0 = blockIdx.y * mmq_x;

  if (col_dst_0 >= num_tokens_post_padded[0]) return;
  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx < 0) return;

  int token_offs[mmq_x / nwarps];
  for (int i = 0; i < mmq_x; i += nwarps) {
    token_offs[i / nwarps] = sorted_token_ids[col_dst_0 + threadIdx.y + i];
  }

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_1* y = (const block_q8_1*)(vy);

  int* tile_x_ql = nullptr;
  half2* tile_x_dm = nullptr;
  int* tile_x_qh = nullptr;
  int* tile_x_sc = nullptr;

  allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

  __shared__ int tile_y_qs[mmq_x * WARP_SIZE_GGUF];
  __shared__ half2 tile_y_ds[mmq_x * WARP_SIZE_GGUF / QI8_1];

  float sum[mmq_y / WARP_SIZE_GGUF][mmq_x / nwarps] = {{0.0f}};

  for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {
    load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
               tile_x_qh, tile_x_sc, threadIdx.y, nrows_x - row_x_0 - 1,
               threadIdx.x, blocks_per_row_x, blocks_per_row_x - ib0);

    const int n_per_r = ((storage_qk * blocks_per_warp) / qr);
#pragma unroll
    for (int ir = 0; ir < qr && ib0 * storage_qk + ir * n_per_r < ncols_x; ++ir) {
      const auto kqs = ir * WARP_SIZE_GGUF + threadIdx.x;
      const int kbxd = kqs / QI8_1;

#pragma unroll
      for (int i = 0; i < mmq_x; i += nwarps) {
        const int col_y_eff = token_offs[i / nwarps] / top_k;
        const int block_x = ib0 * (storage_qk / QK8_1) + kbxd;
        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
          const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + block_x];
          const int index_y =
              (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
          tile_y_qs[index_y] =
              get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
        }
      }

      if constexpr (mmq_x > 8) {
        // Wider token tiles need scales for every group of nwarps tokens.
        const auto kby = threadIdx.x % (WARP_SIZE_GGUF / QI8_1);
        const int block_x =
            ib0 * (storage_qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby;
        for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
          const int ids =
              ids0 + threadIdx.y * QI8_1 +
              threadIdx.x / (WARP_SIZE_GGUF / QI8_1);
          const int col_y_eff =
              sorted_token_ids[col_dst_0 + ids] / top_k;
          if (col_y_eff >= ncols_y || block_x >= blocks_per_col_y) {
            continue;
          }
          const half2* dsi_src =
              &y[col_y_eff * blocks_per_col_y + block_x].ds;
          half2* dsi_dst =
              &tile_y_ds[ids * (WARP_SIZE_GGUF / QI8_1) + kby];
          if (need_sum) {
            *dsi_dst = *dsi_src;
          } else {
            float* dfi_dst = (float*)dsi_dst;
            *dfi_dst = __low2float(*dsi_src);
          }
        }
      } else if (threadIdx.x < n_per_r / QK8_1) {
        const auto kby = threadIdx.x % (WARP_SIZE_GGUF / QI8_1);
        const int col_y_eff = token_offs[0] / top_k;
        const int block_x =
            ib0 * (storage_qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby;
        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
          const half2* dsi_src = &y[col_y_eff * blocks_per_col_y + block_x].ds;
          half2* dsi_dst =
              &tile_y_ds[threadIdx.y * (WARP_SIZE_GGUF / QI8_1) + kby];
          if (need_sum) {
            *dsi_dst = *dsi_src;
          } else {
            float* dfi_dst = (float*)dsi_dst;
            *dfi_dst = __low2float(*dsi_src);
          }
        }
      }
      __syncthreads();

      // #pragma unroll // unrolling this loop causes too much register pressure
      for (int k = ir * WARP_SIZE_GGUF / qr; k < (ir + 1) * WARP_SIZE_GGUF / qr;
           k += vdr) {
#pragma unroll
        for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
          for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            sum[i / WARP_SIZE_GGUF][j / nwarps] +=
                vec_dot(tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs,
                        tile_y_ds, threadIdx.x + i, threadIdx.y + j, k);
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int j = 0; j < mmq_x; j += nwarps) {
    const int col_dst = token_offs[j / nwarps];
    if (col_dst >= ncols_dst) {
      return;
    }

#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
      const auto row_dst = row_dst_0 + threadIdx.x + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE_GGUF][j / nwarps];
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_0 8
  #define MOE_Y_Q4_0 128
  #define NWARPS_Q4_0 8
#else
  #define MOE_X_Q4_0 4
  #define MOE_Y_Q4_0 32
  #define NWARPS_Q4_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_0, 2)
#endif
    moe_q4_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q4_0;
  const int mmq_y = MOE_Y_Q4_0;
  const int nwarps = NWARPS_Q4_0;

  moe_q<scalar_t, QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
        VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MOE_X_Q4_0;
  int mmq_y = MOE_Y_Q4_0;
  int nwarps = NWARPS_Q4_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_1 8
  #define MOE_Y_Q4_1 128
  #define NWARPS_Q4_1 8
#else
  #define MOE_X_Q4_1 4
  #define MOE_Y_Q4_1 32
  #define NWARPS_Q4_1 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_1, 2)
#endif
    moe_q4_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q4_1;
  const int mmq_y = MOE_Y_Q4_1;
  const int nwarps = NWARPS_Q4_1;

  moe_q<scalar_t, QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_1<mmq_y>, load_tiles_q4_1<mmq_y, nwarps, need_check>,
        VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MOE_X_Q4_1;
  int mmq_y = MOE_Y_Q4_1;
  int nwarps = NWARPS_Q4_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_0 8
  #define MOE_Y_Q5_0 128
  #define NWARPS_Q5_0 8
#else
  #define MOE_X_Q5_0 4
  #define MOE_Y_Q5_0 32
  #define NWARPS_Q5_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_0, 2)
#endif
    moe_q5_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q5_0;
  const int mmq_y = MOE_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  moe_q<scalar_t, QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_0<mmq_y>, load_tiles_q5_0<mmq_y, nwarps, need_check>,
        VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q5_0;
  const int mmq_y = MOE_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_1 8
  #define MOE_Y_Q5_1 128
  #define NWARPS_Q5_1 8
#else
  #define MOE_X_Q5_1 4
  #define MOE_Y_Q5_1 32
  #define NWARPS_Q5_1 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_1, 2)
#endif
    moe_q5_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q5_1;
  const int mmq_y = MOE_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  moe_q<scalar_t, QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_1<mmq_y>, load_tiles_q5_1<mmq_y, nwarps, need_check>,
        VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q5_1;
  const int mmq_y = MOE_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q8_0 8
  #define MOE_Y_Q8_0 128
  #define NWARPS_Q8_0 8
#else
  #define MOE_X_Q8_0 4
  #define MOE_Y_Q8_0 32
  #define NWARPS_Q8_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q8_0, 2)
#endif
    moe_q8_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q8_0;
  const int mmq_y = MOE_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  moe_q<scalar_t, QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q8_0<mmq_y>, load_tiles_q8_0<mmq_y, nwarps, need_check>,
        VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q8_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q8_0;
  const int mmq_y = MOE_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q2_K 8
  #define MOE_Y_Q2_K 128
  #define NWARPS_Q2_K 8
#else
  #define MOE_X_Q2_K 4
  #define MOE_Y_Q2_K 32
  #define NWARPS_Q2_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q2_K, 2)
#endif
    moe_q2_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q2_K;
  const int mmq_y = MOE_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  moe_q<scalar_t, QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q2_K<mmq_y>, load_tiles_q2_K<mmq_y, nwarps, need_check>,
        VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q2_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q2_K;
  const int mmq_y = MOE_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

// Q2_0 (GGML type 42): 64-weight blocks with one f16 scale. Same tile shape
// as the other flat quants; the dot product carries the (code-1) offset.
#if defined(USE_ROCM)
  #define MOE_X_Q2_0 8
  #define MOE_Y_Q2_0 128
  #define NWARPS_Q2_0 8
#else
  #define MOE_X_Q2_0 4
  #define MOE_Y_Q2_0 32
  #define NWARPS_Q2_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q2_0, 2)
#endif
    moe_q2_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
             const int* __restrict__ expert_ids,
             const int* __restrict__ num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q2_0;
  const int mmq_y = MOE_Y_Q2_0;
  const int nwarps = NWARPS_Q2_0;

  moe_q<scalar_t, QK2_0, QR2_0, QI2_0_MMQ, false, block_q2_0, mmq_x, mmq_y,
        nwarps, allocate_tiles_q2_0<mmq_y>,
        load_tiles_q2_0<mmq_y, nwarps, need_check>, VDR_Q2_0_Q8_1_MMQ,
        vec_dot_q2_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q2_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q2_0;
  const int mmq_y = MOE_Y_Q2_0;
  const int nwarps = NWARPS_Q2_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q2_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q2_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q3_K 8
  #define MOE_Y_Q3_K 128
  #define NWARPS_Q3_K 8
#else
  #define MOE_X_Q3_K 4
  #define MOE_Y_Q3_K 32
  #define NWARPS_Q3_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q3_K, 2)
#endif
    moe_q3_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {

  const int mmq_x = MOE_X_Q3_K;
  const int mmq_y = MOE_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  moe_q<scalar_t, QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q3_K<mmq_y>, load_tiles_q3_K<mmq_y, nwarps, need_check>,
        VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}
template <typename scalar_t>
static void ggml_moe_q3_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q3_K;
  const int mmq_y = MOE_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_K 8
  #define MOE_Y_Q4_K 128
  #define NWARPS_Q4_K 8
#else
  #define MOE_X_Q4_K 4
  #define MOE_Y_Q4_K 32
  #define NWARPS_Q4_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_K, 2)
#endif
    moe_q4_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q4_K;
  const int mmq_y = MOE_Y_Q4_K;
  const int nwarps = NWARPS_Q4_K;

  moe_q<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_K<mmq_y>, load_tiles_q4_K<mmq_y, nwarps, need_check>,
        VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q4_K;
  const int mmq_y = MOE_Y_Q4_K;
  const int nwarps = NWARPS_Q4_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_K 8
  #define MOE_Y_Q5_K 128
  #define NWARPS_Q5_K 8
#else
  #define MOE_X_Q5_K 4
  #define MOE_Y_Q5_K 32
  #define NWARPS_Q5_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_K, 2)
#endif
    moe_q5_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q5_K;
  const int mmq_y = MOE_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  moe_q<scalar_t, QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_K<mmq_y>, load_tiles_q5_K<mmq_y, nwarps, need_check>,
        VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q5_K;
  const int mmq_y = MOE_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q6_K 8
  #define MOE_Y_Q6_K 128
  #define NWARPS_Q6_K 8
#else
  #define MOE_X_Q6_K 4
  #define MOE_Y_Q6_K 32
  #define NWARPS_Q6_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q6_K, 2)
#endif
    moe_q6_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q6_K;
  const int mmq_y = MOE_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  moe_q<scalar_t, QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q6_K<mmq_y>, load_tiles_q6_K<mmq_y, nwarps, need_check>,
        VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q6_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q6_K;
  const int mmq_y = MOE_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

// IQ2_XS/IQ3_S/IQ4_NL use the normalized 256-weight i-quant tile loaders from
// mmq.cuh.  Keeping the same expert/token grouping as the K-quant kernels
// avoids the one-warp-per-token MMVQ fallback during long prefill batches.
#if defined(USE_ROCM)
  #define MOE_X_IQ 8
  #define MOE_Y_IQ 128
  #define NWARPS_IQ 8
#else
  #define MOE_X_IQ 4
  #define MOE_Y_IQ 32
  #define NWARPS_IQ 4
#endif

// IQ4_NL rows may end after any 32-weight block (e.g. K=320 with TP2).
// Expand eight raw blocks per tile without rounding the stored row stride.
template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_moe_iq4_nl(
    const void* __restrict__ vx, int* __restrict__ x_ql,
    half2* __restrict__ x_dm, int* __restrict__ x_qh,
    int* __restrict__ x_sc, const int& i_offset, const int& i_max,
    const int& k, const int& blocks_per_row, const int& blocks_left) {
  const block_iq4_nl* bx0 = (const block_iq4_nl*)vx;
  const int sub = k >> 2, m = k & 3;
  float* x_scf = (float*)x_sc;
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    const int i = need_check ? min(i0 + i_offset, i_max) : i0 + i_offset;
    const block_iq4_nl* bxi = bx0 + i * blocks_per_row;
    int lo = 0, hi = 0;
    if (sub < blocks_left) {
      const uint32_t pw = get_int_from_uint8(bxi[sub].qs, m);
      lo = iq4xs_lookup_word(pw & 0x0F0F0F0F);
      hi = iq4xs_lookup_word((pw >> 4) & 0x0F0F0F0F);
    }
    x_ql[i * (2 * WARP_SIZE_GGUF + 1) + 8 * sub + m] = lo;
    x_ql[i * (2 * WARP_SIZE_GGUF + 1) + 8 * sub + m + 4] = hi;
    if (k < 8) {
      x_scf[i * 8 + k] = k < blocks_left ? __half2float(bxi[k].d) : 0.0f;
    }
  }
}

template <typename scalar_t, bool need_check, int TILE_ROWS = MOE_Y_IQ>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF * NWARPS_IQ, 2)
#endif
    moe_iq4_nl(const void* __restrict__ vx, const void* __restrict__ vy,
               scalar_t* __restrict__ dst, const int* sorted_token_ids,
               const int* expert_ids, const int* num_tokens_post_padded,
               const int exp_stride, const int ncols_x, const int nrows_x,
               const int ncols_y, const int nrows_y, const int nrows_dst,
               const int top_k) {
  moe_q<scalar_t, QK_K, QR4_XS, QI4_XS, false, block_iq4_nl,
        MOE_X_IQ, TILE_ROWS, NWARPS_IQ,
        allocate_tiles_iq<TILE_ROWS, 2 * WARP_SIZE_GGUF + 1, 8>,
        load_tiles_moe_iq4_nl<TILE_ROWS, NWARPS_IQ, need_check>,
        VDR_IQ_Q8_1_MMQ, vec_dot_iq_v4_mul_mat, QK4_NL>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_iq4_nl_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y,
    const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const char* tile_env = std::getenv("VLLM_GGUF_MOE_IQ4_NL_TILE_ROWS");
  int default_tile_rows = MOE_Y_IQ;
#ifdef USE_ROCM
  if (std::string(at::cuda::getCurrentDeviceProperties()->gcnArchName)
          .find("gfx906") == 0) {
    default_tile_rows = 32;
  }
#endif
  const int tile_rows = tile_env == nullptr ? default_tile_rows : std::atoi(tile_env);
  const dim3 block_dims(WARP_SIZE_GGUF, NWARPS_IQ, 1);
#define LAUNCH_IQ_TILE(ROWS_) \
  do { \
    const dim3 grid((nrows_x + ROWS_ - 1) / ROWS_, \
                    tokens_post_padded / MOE_X_IQ, 1); \
    if (nrows_x % ROWS_ == 0) { \
      moe_iq4_nl<scalar_t, false, ROWS_><<<grid, block_dims, 0, stream>>>( \
          w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded, \
          exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k); \
    } else { \
      moe_iq4_nl<scalar_t, true, ROWS_><<<grid, block_dims, 0, stream>>>( \
          w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded, \
          exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k); \
    } \
  } while (0)
  if (tile_rows == 32) {
    LAUNCH_IQ_TILE(32);
  } else if (tile_rows == 64) {
    LAUNCH_IQ_TILE(64);
  } else {
    LAUNCH_IQ_TILE(MOE_Y_IQ);
  }
#undef LAUNCH_IQ_TILE
}

template <typename scalar_t, bool need_check, int TILE_ROWS = MOE_Y_IQ>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF * NWARPS_IQ, 2)
#endif
    moe_iq3_s(const void* __restrict__ vx, const void* __restrict__ vy,
              scalar_t* __restrict__ dst, const int* sorted_token_ids,
              const int* expert_ids, const int* num_tokens_post_padded,
              const int exp_stride, const int ncols_x, const int nrows_x,
              const int ncols_y, const int nrows_y, const int nrows_dst,
              const int top_k) {
  moe_q<scalar_t, QK_K, QR4_XS, QI4_XS, false, block_iq3_s,
        MOE_X_IQ, TILE_ROWS, NWARPS_IQ,
        allocate_tiles_iq<TILE_ROWS, 2 * WARP_SIZE_GGUF + 1, 8>,
        load_tiles_iq3_s<TILE_ROWS, NWARPS_IQ, need_check>,
        VDR_IQ_Q8_1_MMQ, vec_dot_iq_v4_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_iq3_s_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y,
    const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const char* tile_env = std::getenv("VLLM_GGUF_MOE_IQ3_S_TILE_ROWS");
  int default_tile_rows = MOE_Y_IQ;
#ifdef USE_ROCM
  if (std::string(at::cuda::getCurrentDeviceProperties()->gcnArchName)
          .find("gfx906") == 0) {
    default_tile_rows = 32;
  }
#endif
  const int tile_rows = tile_env == nullptr ? default_tile_rows : std::atoi(tile_env);
  const dim3 block_dims(WARP_SIZE_GGUF, NWARPS_IQ, 1);
#define LAUNCH_IQ_TILE(ROWS_) \
  do { \
    const dim3 grid((nrows_x + ROWS_ - 1) / ROWS_, \
                    tokens_post_padded / MOE_X_IQ, 1); \
    if (nrows_x % ROWS_ == 0) { \
      moe_iq3_s<scalar_t, false, ROWS_><<<grid, block_dims, 0, stream>>>( \
          w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded, \
          exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k); \
    } else { \
      moe_iq3_s<scalar_t, true, ROWS_><<<grid, block_dims, 0, stream>>>( \
          w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded, \
          exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k); \
    } \
  } while (0)
  if (tile_rows == 32) {
    LAUNCH_IQ_TILE(32);
  } else if (tile_rows == 64) {
    LAUNCH_IQ_TILE(64);
  } else {
    LAUNCH_IQ_TILE(MOE_Y_IQ);
  }
#undef LAUNCH_IQ_TILE
}

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF * NWARPS_IQ, 2)
#endif
    moe_iq2_xs(const void* __restrict__ vx, const void* __restrict__ vy,
               scalar_t* __restrict__ dst, const int* sorted_token_ids,
               const int* expert_ids, const int* num_tokens_post_padded,
               const int exp_stride, const int ncols_x, const int nrows_x,
               const int ncols_y, const int nrows_y, const int nrows_dst,
               const int top_k) {
  moe_q<scalar_t, QK_K, QR4_XS, QI4_XS, false, block_iq2_xs,
        MOE_X_IQ, MOE_Y_IQ, NWARPS_IQ,
        allocate_tiles_iq<MOE_Y_IQ, 2 * WARP_SIZE_GGUF + 1, 16>,
        load_tiles_iq2_xs<MOE_Y_IQ, NWARPS_IQ, need_check>,
        VDR_IQ_Q8_1_MMQ, vec_dot_iq_v8_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_iq2_xs_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y,
    const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const dim3 block_nums((nrows_x + MOE_Y_IQ - 1) / MOE_Y_IQ,
                        tokens_post_padded / MOE_X_IQ, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, NWARPS_IQ, 1);
  if (nrows_x % MOE_Y_IQ == 0) {
    moe_iq2_xs<scalar_t, false><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    moe_iq2_xs<scalar_t, true><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

// The remaining normalized i-quant families (IQ3_XXS v4, IQ2_S v8 and the
// IQ4_XS tile pair) use the same expert/token grouping so long prefill
// batches avoid the per-token MMVQ fallback on them too.
template <typename scalar_t, bool need_check, int TILE_ROWS = MOE_Y_IQ>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF * NWARPS_IQ, 2)
#endif
    moe_iq3_xxs(const void* __restrict__ vx, const void* __restrict__ vy,
                scalar_t* __restrict__ dst, const int* sorted_token_ids,
                const int* expert_ids, const int* num_tokens_post_padded,
                const int exp_stride, const int ncols_x, const int nrows_x,
                const int ncols_y, const int nrows_y, const int nrows_dst,
                const int top_k) {
  moe_q<scalar_t, QK_K, QR4_XS, QI4_XS, false, block_iq3_xxs,
        MOE_X_IQ, TILE_ROWS, NWARPS_IQ,
        allocate_tiles_iq<TILE_ROWS, 2 * WARP_SIZE_GGUF + 1, 8>,
        load_tiles_iq3_xxs<TILE_ROWS, NWARPS_IQ, need_check>,
        VDR_IQ_Q8_1_MMQ, vec_dot_iq_v4_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_iq3_xxs_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const char* tile_env = std::getenv("VLLM_GGUF_MOE_IQ3_XXS_TILE_ROWS");
  int default_tile_rows = MOE_Y_IQ;
#ifdef USE_ROCM
  if (std::string(at::cuda::getCurrentDeviceProperties()->gcnArchName)
          .find("gfx906") == 0) {
    default_tile_rows = 32;
  }
#endif
  const int tile_rows = tile_env == nullptr ? default_tile_rows : std::atoi(tile_env);
  const dim3 block_dims(WARP_SIZE_GGUF, NWARPS_IQ, 1);
#define LAUNCH_IQ_TILE(ROWS_) \
  do { \
    const dim3 grid((nrows_x + ROWS_ - 1) / ROWS_, \
                    tokens_post_padded / MOE_X_IQ, 1); \
    if (nrows_x % ROWS_ == 0) { \
      moe_iq3_xxs<scalar_t, false, ROWS_><<<grid, block_dims, 0, stream>>>( \
          w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded, \
          exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k); \
    } else { \
      moe_iq3_xxs<scalar_t, true, ROWS_><<<grid, block_dims, 0, stream>>>( \
          w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded, \
          exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k); \
    } \
  } while (0)
  if (tile_rows == 32) {
    LAUNCH_IQ_TILE(32);
  } else if (tile_rows == 64) {
    LAUNCH_IQ_TILE(64);
  } else {
    LAUNCH_IQ_TILE(MOE_Y_IQ);
  }
#undef LAUNCH_IQ_TILE
}

template <typename scalar_t, bool need_check, int TILE_ROWS = MOE_Y_IQ>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF * NWARPS_IQ, 2)
#endif
    moe_iq2_s(const void* __restrict__ vx, const void* __restrict__ vy,
              scalar_t* __restrict__ dst, const int* sorted_token_ids,
              const int* expert_ids, const int* num_tokens_post_padded,
              const int exp_stride, const int ncols_x, const int nrows_x,
              const int ncols_y, const int nrows_y, const int nrows_dst,
              const int top_k) {
  moe_q<scalar_t, QK_K, QR4_XS, QI4_XS, false, block_iq2_s,
        MOE_X_IQ, TILE_ROWS, NWARPS_IQ,
        allocate_tiles_iq<TILE_ROWS, 2 * WARP_SIZE_GGUF + 1, 16>,
        load_tiles_iq2_s<TILE_ROWS, NWARPS_IQ, need_check>,
        VDR_IQ_Q8_1_MMQ, vec_dot_iq_v8_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_iq2_s_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const char* tile_env = std::getenv("VLLM_GGUF_MOE_IQ2_S_TILE_ROWS");
  int default_tile_rows = MOE_Y_IQ;
#ifdef USE_ROCM
  if (std::string(at::cuda::getCurrentDeviceProperties()->gcnArchName)
          .find("gfx906") == 0) {
    default_tile_rows = 32;
  }
#endif
  const int tile_rows = tile_env == nullptr ? default_tile_rows : std::atoi(tile_env);
  const dim3 block_dims(WARP_SIZE_GGUF, NWARPS_IQ, 1);
#define LAUNCH_IQ_TILE(ROWS_) \
  do { \
    const dim3 grid((nrows_x + ROWS_ - 1) / ROWS_, \
                    tokens_post_padded / MOE_X_IQ, 1); \
    if (nrows_x % ROWS_ == 0) { \
      moe_iq2_s<scalar_t, false, ROWS_><<<grid, block_dims, 0, stream>>>( \
          w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded, \
          exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k); \
    } else { \
      moe_iq2_s<scalar_t, true, ROWS_><<<grid, block_dims, 0, stream>>>( \
          w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded, \
          exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k); \
    } \
  } while (0)
  if (tile_rows == 32) {
    LAUNCH_IQ_TILE(32);
  } else if (tile_rows == 64) {
    LAUNCH_IQ_TILE(64);
  } else {
    LAUNCH_IQ_TILE(MOE_Y_IQ);
  }
#undef LAUNCH_IQ_TILE
}

template <typename scalar_t, bool need_check, int TILE_ROWS = MOE_Y_IQ>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF * NWARPS_IQ, 2)
#endif
    moe_iq4_xs(const void* __restrict__ vx, const void* __restrict__ vy,
               scalar_t* __restrict__ dst, const int* sorted_token_ids,
               const int* expert_ids, const int* num_tokens_post_padded,
               const int exp_stride, const int ncols_x, const int nrows_x,
               const int ncols_y, const int nrows_y, const int nrows_dst,
               const int top_k) {
  moe_q<scalar_t, QK_K, QR4_XS, QI4_XS, false, block_iq4_xs,
        MOE_X_IQ, TILE_ROWS, NWARPS_IQ,
        allocate_tiles_iq4_xs<TILE_ROWS>,
        load_tiles_iq4_xs<TILE_ROWS, NWARPS_IQ, need_check>,
        VDR_IQ4_XS_Q8_1_MMQ, vec_dot_iq4_xs_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_iq4_xs_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y,
    const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const dim3 block_nums((nrows_x + MOE_Y_IQ - 1) / MOE_Y_IQ,
                        tokens_post_padded / MOE_X_IQ, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, NWARPS_IQ, 1);
  if (nrows_x % MOE_Y_IQ == 0) {
    moe_iq4_xs<scalar_t, false><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    moe_iq4_xs<scalar_t, true><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}
