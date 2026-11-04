// copied and adapted from
// https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int LANES = 0>
static __global__ void moe_vec_q(const void* __restrict__ vx,
                                 const void* __restrict__ vy,
                                 scalar_t* __restrict__ dst,
                                 const int* topk_ids, const int topk,
                                 const int ncols, const int nrows,
                                 const int token_stride) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;

  const auto token = blockIdx.z / topk;
  const auto expert = (topk_ids)[blockIdx.z];

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int row_lanes = LANES > 0 ? LANES : WARP_SIZE;
  const int blocks_per_warp = vdr * row_lanes / qi;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = ((const block_q_t*)vx) + expert * nrows * blocks_per_row;
  const block_q8_1* y =
      (const block_q8_1*)(((const int*)vy) + token * token_stride);

  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row;
       i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i;  // x block index

    const int iby = i * (qk / QK8_1);  // y block index that aligns with ibx

    const int iqs =
        vdr *
        (threadIdx.x %
         (qi / vdr));  // x block quant index when casting the quants to int

    tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = row_lanes / 2; mask > 0; mask >>= 1) {
    tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
  }

  if (threadIdx.x == 0) {
    dst[blockIdx.z * nrows + row] = tmp;
  }
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
static bool try_moe_vec_small_k(
    const void* vx, const void* vy, scalar_t* dst, const int* topk_ids,
    int top_k, int tokens, int ncols, int nrows, int token_stride,
    cudaStream_t stream) {
#ifdef USE_ROCM
  const char* enabled = std::getenv("VLLM_GGUF_MOE_SMALL_K_SUBWARP");
  const int active_lanes = (ncols / qk) * (qi / vdr);
  if (enabled == nullptr || enabled[0] != '1' || qk != 32 || ncols <= 0 ||
      ncols % qk != 0 || active_lanes > 32 ||
      std::string(at::cuda::getCurrentDeviceProperties()->gcnArchName)
              .find("gfx906") != 0) {
    return false;
  }
  // Keep 128 threads per CTA while packing independent short rows into
  // subwarps. Each row fits in one iteration, so the discarded lanes in
  // the 64-lane reference contributed only zeros to its reduction.
  const int lanes = active_lanes <= 16 ? 16 : 32;
  const int rows = 128 / lanes;
  const dim3 grid((nrows + rows - 1) / rows, 1, tokens * top_k);
  const dim3 block(lanes, rows, 1);
  if (lanes == 16) {
    moe_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 16>
        <<<grid, block, 0, stream>>>(
            vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
  } else {
    moe_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 32>
        <<<grid, block, 0, stream>>>(
            vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
  }
  return true;
#else
  return false;
#endif
}

template <typename scalar_t>
static void moe_vec_q4_0_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
            vec_dot_q4_0_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q4_1_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
            vec_dot_q4_1_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q5_0_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
            vec_dot_q5_0_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q5_1_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
            vec_dot_q5_1_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q8_0_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  if (try_moe_vec_small_k<scalar_t, QK8_0, QI8_0, block_q8_0,
                          VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
          vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride,
          stream)) {
    return;
  }
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
            vec_dot_q8_0_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q2_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
            vec_dot_q2_K_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q3_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
            vec_dot_q3_K_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q4_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
            vec_dot_q4_K_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q5_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
            vec_dot_q5_K_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q6_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
            vec_dot_q6_K_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq2_xxs_q8_1_cuda(const void* vx, const void* vy,
                                      scalar_t* dst, const int* topk_ids,
                                      const int top_k, const int tokens,
                                      const int ncols, const int nrows,
                                      const int token_stride,
                                      cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq2_xs_q8_1_cuda(const void* vx, const void* vy,
                                     scalar_t* dst, const int* topk_ids,
                                     const int top_k, const int tokens,
                                     const int ncols, const int nrows,
                                     const int token_stride,
                                     cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq2_s_q8_1_cuda(const void* vx, const void* vy,
                                    scalar_t* dst, const int* topk_ids,
                                    const int top_k, const int tokens,
                                    const int ncols, const int nrows,
                                    const int token_stride,
                                    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq3_xxs_q8_1_cuda(const void* vx, const void* vy,
                                      scalar_t* dst, const int* topk_ids,
                                      const int top_k, const int tokens,
                                      const int ncols, const int nrows,
                                      const int token_stride,
                                      cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq1_s_q8_1_cuda(const void* vx, const void* vy,
                                    scalar_t* dst, const int* topk_ids,
                                    const int top_k, const int tokens,
                                    const int ncols, const int nrows,
                                    const int token_stride,
                                    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq1_m_q8_1_cuda(const void* vx, const void* vy,
                                    scalar_t* dst, const int* topk_ids,
                                    const int top_k, const int tokens,
                                    const int ncols, const int nrows,
                                    const int token_stride,
                                    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq4_nl_q8_1_cuda(const void* vx, const void* vy,
                                     scalar_t* dst, const int* topk_ids,
                                     const int top_k, const int tokens,
                                     const int ncols, const int nrows,
                                     const int token_stride,
                                     cudaStream_t stream) {
  if (try_moe_vec_small_k<scalar_t, QK4_NL, QI4_NL, block_iq4_nl,
                          VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
          vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride,
          stream)) {
    return;
  }
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ,
            vec_dot_iq4_nl_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq4_xs_q8_1_cuda(const void* vx, const void* vy,
                                     scalar_t* dst, const int* topk_ids,
                                     const int top_k, const int tokens,
                                     const int ncols, const int nrows,
                                     const int token_stride,
                                     cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI4_XS, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ,
            vec_dot_iq4_xs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq3_s_q8_1_cuda(const void* vx, const void* vy,
                                    scalar_t* dst, const int* topk_ids,
                                    const int top_k, const int tokens,
                                    const int ncols, const int nrows,
                                    const int token_stride,
                                    cudaStream_t stream) {
#ifdef USE_ROCM
  const char* lut_env = std::getenv("VLLM_GGUF_MOE_IQ3_S_LUT");
  if (lut_env != nullptr && lut_env[0] == '1' &&
      std::string(at::cuda::getCurrentDeviceProperties()->gcnArchName)
              .find("gfx906") == 0) {
    const dim3 grid((nrows + 3) / 4, 1, tokens * top_k);
    mul_mat_vec_iq_lut<scalar_t, 1, 3, true, true>
        <<<grid, 256, 0, stream>>>(
            (const uint8_t*)vx, (const block_q8_1*)vy, dst, ncols,
            nrows, nrows, topk_ids, top_k, token_stride);
    return;
  }
#endif
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}
