#if defined(USE_ROCM)
#define BLOCK_SIZE 64
#else
#define BLOCK_SIZE 128
#endif

static inline int gguf_mmvq_grid_vecs(const int nvecs) {
#if defined(USE_ROCM)
    return nvecs >= 2 && nvecs <= 4 ? 1 : nvecs;
#else
    return nvecs;
#endif
}

// When true, one block computes the dot products of its weight row against all
// activation vectors (2..4), loading the weight row once. Mirrors the
// combine_vecs logic of the generic mul_mat_vec_q kernel above, so multi-token
// batches (e.g. MTP verify with 1 + num_spec tokens) keep weight traffic at the
// single-vector level. The multi-warp epilogues below only handle vec 0, so
// combining is restricted to the single-warp block shapes used on ROCm
// (BLOCK_SIZE == WARP_SIZE there).
static inline __host__ __device__ bool gguf_mmvq_combine_vecs(const int nvecs) {
#if defined(USE_ROCM)
    return nvecs >= 2 && nvecs <= 4;
#else
    return false;
#endif
}

// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs, const int dst_stride) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
#if defined(USE_ROCM)
    const bool combine_vecs = nvecs >= 2 && nvecs <= 4;
#else
    const bool combine_vecs = false;
#endif
    const int first_vec = combine_vecs ? 0 : blockIdx.y;
    const int vec_count = combine_vecs ? nvecs : 1;

    if (row >= nrows || first_vec >= nvecs) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * BLOCK_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    // partial sum for each thread
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            const int vec = first_vec + vec_offset;
            const int iby = vec*(nrows_y/QK8_1) + i * (qk/QK8_1);
            tmp[vec_offset] += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
        }
    }

    constexpr int warp_size = WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / warp_size;
    // sum up partial sums and write back result
#pragma unroll
    for (int mask = warp_size/2; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            tmp[vec_offset] += VLLM_SHFL_XOR_SYNC(tmp[vec_offset], mask);
        }
    }

    if constexpr (num_warps == 1) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                if (vec_offset >= vec_count) {
                    break;
                }
                dst[(first_vec + vec_offset)*dst_stride + row] = tmp[vec_offset];
            }
        }
    } else {
        const int lane = threadIdx.x % warp_size;
        const int warp = threadIdx.x / warp_size;
        __shared__ float shared_sum[GGML_CUDA_MMV_Y][num_warps];

        if (lane == 0) {
            shared_sum[threadIdx.y][warp] = tmp[0];
        }
        __syncthreads();

        if (warp != 0) {
            return;
        }

        tmp[0] = lane < num_warps ? shared_sum[threadIdx.y][lane] : 0.0f;
#pragma unroll
        for (int mask = warp_size/2; mask > 0; mask >>= 1) {
            tmp[0] += VLLM_SHFL_XOR_SYNC(tmp[0], mask);
        }

        if (lane == 0) {
            dst[first_vec*dst_stride + row] = tmp[0];
        }
    }
}

template<typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template <typename scalar_t, int rows_per_wave>
static __global__ void mul_mat_vec_q8_0_q8_1_row_tile(
    const void * __restrict__ vx, const void * __restrict__ vy,
    scalar_t * __restrict__ dst, const int ncols, const int nrows,
    const int nvecs, const int dst_stride) {
    const int first_row = blockIdx.x * rows_per_wave;
    const bool combine_vecs = gguf_mmvq_combine_vecs(nvecs);
    const int first_vec = combine_vecs ? 0 : blockIdx.y;
    const int vec_count = combine_vecs ? nvecs : 1;
    if (first_row >= nrows || first_vec >= nvecs) {
        return;
    }

    const int blocks_per_row = ncols / QK8_0;
    const int blocks_per_wave =
        VDR_Q8_0_Q8_1_MMVQ * WARP_SIZE / QI8_0;
    const int q8_blocks_per_vec =
        ((ncols + 512 - 1) / 512 * 512) / QK8_1;
    const block_q8_0 * x = static_cast<const block_q8_0 *>(vx);
    const block_q8_1 * y = static_cast<const block_q8_1 *>(vy);
    float tmp[4][rows_per_wave] = {};

    for (int i = threadIdx.x / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
         i < blocks_per_row; i += blocks_per_wave) {
        const int iqs = VDR_Q8_0_Q8_1_MMVQ *
            (threadIdx.x % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));
        int activation[4][VDR_Q8_0_Q8_1_MMVQ];
        float activation_scale[4];
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            const block_q8_1 * y_block =
                &y[(first_vec + vec_offset) * q8_blocks_per_vec + i];
#pragma unroll
            for (int word = 0; word < VDR_Q8_0_Q8_1_MMVQ; ++word) {
                activation[vec_offset][word] =
                    get_int_from_int8_aligned(y_block->qs, iqs + word);
            }
            activation_scale[vec_offset] = __low2float(y_block->ds);
        }

#pragma unroll
        for (int row_offset = 0; row_offset < rows_per_wave; ++row_offset) {
            const int row = first_row + row_offset;
            if (row >= nrows) {
                continue;
            }
            const block_q8_0 * weight_block =
                &x[row * blocks_per_row + i];
            const float weight_scale = __half2float(weight_block->d);
#pragma unroll
            for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                if (vec_offset >= vec_count) {
                    break;
                }
                int dot = 0;
#pragma unroll
                for (int word = 0; word < VDR_Q8_0_Q8_1_MMVQ; ++word) {
                    const int weight =
                        get_int_from_int8(weight_block->qs, iqs + word);
                    dot = __dp4a(weight, activation[vec_offset][word], dot);
                }
                tmp[vec_offset][row_offset] +=
                    weight_scale * activation_scale[vec_offset] * dot;
            }
        }
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
#pragma unroll
        for (int row_offset = 0; row_offset < rows_per_wave; ++row_offset) {
#pragma unroll
            for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                if (vec_offset >= vec_count) {
                    break;
                }
                tmp[vec_offset][row_offset] +=
                    VLLM_SHFL_XOR_SYNC(tmp[vec_offset][row_offset], mask);
            }
        }
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int row_offset = 0; row_offset < rows_per_wave; ++row_offset) {
            const int row = first_row + row_offset;
            if (row < nrows) {
#pragma unroll
                for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                    if (vec_offset >= vec_count) {
                        break;
                    }
                    dst[(first_vec + vec_offset) * dst_stride + row] =
                        tmp[vec_offset][row_offset];
                }
            }
        }
    }
}

template <typename scalar_t, int rows_per_wave>
static void mul_mat_vec_q8_0_q8_1_row_tile_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols,
    const int nrows, const int nvecs, cudaStream_t stream,
    const int dst_stride = -1) {
    const dim3 block_nums(
        (nrows + rows_per_wave - 1) / rows_per_wave,
        gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_vec_q8_0_q8_1_row_tile<scalar_t, rows_per_wave>
        <<<block_nums, block_dims, 0, stream>>>(
            vx, vy, dst, ncols, nrows, nvecs,
            dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template <typename scalar_t, int ncols, bool PREPARED = false>
static __global__ void mul_mat_vec_q4_K_q8_1_fixed_cols(
    const void * __restrict__ vx, const void * __restrict__ vy,
    scalar_t * __restrict__ dst, const int nrows, const int nvecs,
    const int dst_stride) {
    constexpr int blocks_per_row = ncols / QK_K;
    constexpr int q8_blocks_per_vec = ncols / QK8_1;
    constexpr int blocks_per_warp = VDR_Q4_K_Q8_1_MMVQ * BLOCK_SIZE / QI4_K;

    const int row = blockIdx.x;
    const bool combine_vecs = gguf_mmvq_combine_vecs(nvecs);
    const int first_vec = combine_vecs ? 0 : blockIdx.y;
    const int vec_count = combine_vecs ? nvecs : 1;
    if (row >= nrows || first_vec >= nvecs) {
        return;
    }

    const block_q4_K * x = (const block_q4_K *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if constexpr (PREPARED) {
      // Multi-vector path: the weight-side unpack runs once per weight block
      // and is shared across the combined vectors.
      MmvqWeightQ4K w4;
      for (int i = threadIdx.x / (QI4_K / VDR_Q4_K_Q8_1_MMVQ);
           i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iqs = VDR_Q4_K_Q8_1_MMVQ *
            (threadIdx.x % (QI4_K / VDR_Q4_K_Q8_1_MMVQ));
        mmvq_prepare_q4_K(&x[ibx], iqs, w4);
        const block_q8_1 * y_row =
            &y[first_vec * q8_blocks_per_vec + i * (QK_K / QK8_1)];
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
          if (vec_offset >= vec_count) {
            break;
          }
          tmp[vec_offset] += mmvq_dot_q4_K(
              w4, y_row + vec_offset * q8_blocks_per_vec, iqs);
        }
      }
    } else {
      // Single-vector instantiation: the original one-call vec_dot, whose
      // codegen measured faster when there is nothing to share across
      // vectors.
      for (int i = threadIdx.x / (QI4_K / VDR_Q4_K_Q8_1_MMVQ);
           i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iqs = VDR_Q4_K_Q8_1_MMVQ *
            (threadIdx.x % (QI4_K / VDR_Q4_K_Q8_1_MMVQ));
        const int iby = first_vec * q8_blocks_per_vec + i * (QK_K / QK8_1);
        tmp[0] += vec_dot_q4_K_q8_1(&x[ibx], &y[iby], iqs);
      }
    }

    constexpr int warp_size = WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / warp_size;
#pragma unroll
    for (int mask = warp_size/2; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            tmp[vec_offset] += VLLM_SHFL_XOR_SYNC(tmp[vec_offset], mask);
        }
    }

    if constexpr (num_warps == 1) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                if (vec_offset >= vec_count) {
                    break;
                }
                dst[(first_vec + vec_offset)*dst_stride + row] = tmp[vec_offset];
            }
        }
    } else {
        const int lane = threadIdx.x % warp_size;
        const int warp = threadIdx.x / warp_size;
        __shared__ float shared_sum[num_warps];
        if (lane == 0) {
            shared_sum[warp] = tmp[0];
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
        tmp[0] = lane < num_warps ? shared_sum[lane] : 0.0f;
#pragma unroll
        for (int mask = warp_size/2; mask > 0; mask >>= 1) {
            tmp[0] += VLLM_SHFL_XOR_SYNC(tmp[0], mask);
        }
        if (lane == 0) {
            dst[first_vec*dst_stride + row] = tmp[0];
        }
    }
}

template<typename scalar_t, int ncols>
static void mul_mat_vec_q4_K_q8_1_fixed_cols_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const dim3 block_nums(nrows, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, 1, 1);
    if (nvecs >= 2) {
        mul_mat_vec_q4_K_q8_1_fixed_cols<scalar_t, ncols, true>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
    } else {
        mul_mat_vec_q4_K_q8_1_fixed_cols<scalar_t, ncols, false>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
    }
}

template<typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_col2560_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    mul_mat_vec_q4_K_q8_1_fixed_cols_cuda<scalar_t, 2560>(
        vx, vy, dst, nrows, nvecs, stream, dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template <typename scalar_t, int ncols, bool PREPARED = false>
static __global__ void mul_mat_vec_q5_K_q8_1_fixed_cols(
    const void * __restrict__ vx, const void * __restrict__ vy,
    scalar_t * __restrict__ dst, const int nrows, const int nvecs,
    const int dst_stride) {
    constexpr int blocks_per_row = ncols / QK_K;
    constexpr int q8_blocks_per_vec = ncols / QK8_1;
    constexpr int blocks_per_warp = VDR_Q5_K_Q8_1_MMVQ * BLOCK_SIZE / QI5_K;

    const int row = blockIdx.x;
    const bool combine_vecs = gguf_mmvq_combine_vecs(nvecs);
    const int first_vec = combine_vecs ? 0 : blockIdx.y;
    const int vec_count = combine_vecs ? nvecs : 1;
    if (row >= nrows || first_vec >= nvecs) {
        return;
    }

    const block_q5_K * x = (const block_q5_K *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if constexpr (PREPARED) {
      // Multi-vector path: the weight-side unpack runs once per weight block
      // and is shared across the combined vectors.
      MmvqWeightQ5K w5;
      for (int i = threadIdx.x / (QI5_K / VDR_Q5_K_Q8_1_MMVQ);
           i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iqs = VDR_Q5_K_Q8_1_MMVQ *
            (threadIdx.x % (QI5_K / VDR_Q5_K_Q8_1_MMVQ));
        mmvq_prepare_q5_K(&x[ibx], iqs, w5);
        const block_q8_1 * y_row =
            &y[first_vec * q8_blocks_per_vec + i * (QK_K / QK8_1)];
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
          if (vec_offset >= vec_count) {
            break;
          }
          tmp[vec_offset] += mmvq_dot_q5_K(
              w5, y_row + vec_offset * q8_blocks_per_vec, iqs);
        }
      }
    } else {
      // Single-vector instantiation: the original one-call vec_dot, whose
      // codegen measured faster when there is nothing to share across
      // vectors.
      for (int i = threadIdx.x / (QI5_K / VDR_Q5_K_Q8_1_MMVQ);
           i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iqs = VDR_Q5_K_Q8_1_MMVQ *
            (threadIdx.x % (QI5_K / VDR_Q5_K_Q8_1_MMVQ));
        const int iby = first_vec * q8_blocks_per_vec + i * (QK_K / QK8_1);
        tmp[0] += vec_dot_q5_K_q8_1(&x[ibx], &y[iby], iqs);
      }
    }

    constexpr int warp_size = WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / warp_size;
#pragma unroll
    for (int mask = warp_size/2; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            tmp[vec_offset] += VLLM_SHFL_XOR_SYNC(tmp[vec_offset], mask);
        }
    }

    if constexpr (num_warps == 1) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                if (vec_offset >= vec_count) {
                    break;
                }
                dst[(first_vec + vec_offset)*dst_stride + row] = tmp[vec_offset];
            }
        }
    } else {
        const int lane = threadIdx.x % warp_size;
        const int warp = threadIdx.x / warp_size;
        __shared__ float shared_sum[num_warps];
        if (lane == 0) {
            shared_sum[warp] = tmp[0];
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
        tmp[0] = lane < num_warps ? shared_sum[lane] : 0.0f;
#pragma unroll
        for (int mask = warp_size/2; mask > 0; mask >>= 1) {
            tmp[0] += VLLM_SHFL_XOR_SYNC(tmp[0], mask);
        }
        if (lane == 0) {
            dst[first_vec*dst_stride + row] = tmp[0];
        }
    }
}

template<typename scalar_t, int ncols>
static void mul_mat_vec_q5_K_q8_1_fixed_cols_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
  const dim3 block_nums(nrows, gguf_mmvq_grid_vecs(nvecs), 1);
  const dim3 block_dims(BLOCK_SIZE, 1, 1);
  if (nvecs >= 2) {
    mul_mat_vec_q5_K_q8_1_fixed_cols<scalar_t, ncols, true>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
  } else {
    mul_mat_vec_q5_K_q8_1_fixed_cols<scalar_t, ncols, false>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
  }
}

template<typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_col2560_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    mul_mat_vec_q5_K_q8_1_fixed_cols_cuda<scalar_t, 2560>(
        vx, vy, dst, nrows, nvecs, stream, dst_stride);
}

#if defined(USE_ROCM)
// Software-pipelined q5_K fixed-cols variant. Same mapping, same math and same
// accumulation order as mul_mat_vec_q5_K_q8_1_fixed_cols above (bit-exact
// output), but each thread issues the next iteration's weight loads before
// running the unpack/dot math of the current one, keeping more HBM requests in
// flight. Measured on gfx906: 454 -> 604 GB/s on the 5120x17408 shape
// (docs gemv-plateau/bench3). All fragment state is held in named scalar
// registers behind reference-inline helpers; struct-by-value returns spill on
// gfx906 and must not be reintroduced.
struct Gfx906Q5KWtFrag {
    int ql0, ql1;
    int qh0w, qh1w;  // raw qh words, bq8_offset shift applied in compute
    int sw0, sw1, sw2;  // scales[0..11]
    half2 dm;
};

struct Gfx906Q5KActFrag {
    int u0, u1, u2, u3;  // u[2*i + {0,1}] for i = 0, 1
    float d80, d81;
};

static __device__ __forceinline__ void gfx906_q5k_ld_wt(
    Gfx906Q5KWtFrag & f, const block_q5_K * __restrict__ bq,
    const int iqs, const int bq8_offset) {
    const int * ql = (const int *)(bq->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq->qh + 4 * ((iqs/2)%4));
    f.ql0 = ql[0];
    f.ql1 = ql[4];
    f.qh0w = qh[0];
    f.qh1w = qh[4];
    f.sw0 = *(const int *)&bq->scales[0];
    f.sw1 = *(const int *)&bq->scales[4];
    f.sw2 = *(const int *)&bq->scales[8];
    f.dm = bq->dm;
}

static __device__ __forceinline__ void gfx906_q5k_ld_act(
    Gfx906Q5KActFrag & f, const block_q8_1 * __restrict__ y,
    const int vec, const int q8_blocks_per_vec, const int i,
    const int bq8_offset, const int iqs) {
    const block_q8_1 * yb = y + vec * q8_blocks_per_vec + i * (QK_K / QK8_1) + bq8_offset;
    f.d80 = __low2float(yb[0].ds);
    f.d81 = __low2float(yb[1].ds);
    const int * q8a = (const int *)yb[0].qs + ((iqs/2)%4);
    const int * q8b = (const int *)yb[1].qs + ((iqs/2)%4);
    f.u0 = q8a[0];
    f.u1 = q8a[4];
    f.u2 = q8b[0];
    f.u3 = q8b[4];
}

// Scalar re-derivation of the 6-bit scale/min bytes; must stay value-identical
// to the uint16 path in vec_dot_q5_K_q8_1 / mmvq_prepare_q5_K.
static __device__ __forceinline__ void gfx906_kq_scales(
    const int sw0, const int sw1, const int sw2, const int bq8_offset,
    int & sc0, int & sc1, int & m0, int & m1) {
    const unsigned s0 = (unsigned)sw0 & 0xffffu;
    const unsigned s1 = ((unsigned)sw0 >> 16) & 0xffffu;
    const unsigned s2 = (unsigned)sw1 & 0xffffu;
    const unsigned s3 = ((unsigned)sw1 >> 16) & 0xffffu;
    const unsigned s4 = (unsigned)sw2 & 0xffffu;
    const unsigned s5 = ((unsigned)sw2 >> 16) & 0xffffu;
    unsigned a0, a1;
    const int j = bq8_offset / 2;
    if (j < 2) {
        const unsigned sa = j == 0 ? s0 : s1;
        const unsigned sb = j == 0 ? s2 : s3;
        a0 = sa & 0x3f3fu;
        a1 = sb & 0x3f3fu;
    } else {
        const unsigned sh = j == 2 ? s4 : s5;
        const unsigned sl = j == 2 ? s0 : s1;
        const unsigned sm = j == 2 ? s2 : s3;
        a0 = (sh & 0x0f0fu) | ((sl & 0xc0c0u) >> 2);
        a1 = ((sh >> 4) & 0x0f0fu) | ((sm & 0xc0c0u) >> 2);
    }
    sc0 = (int)(a0 & 0xffu);
    sc1 = (int)((a0 >> 8) & 0xffu);
    m0 = (int)(a1 & 0xffu);
    m1 = (int)((a1 >> 8) & 0xffu);
}

static __device__ __forceinline__ float gfx906_q5k_comp(
    const Gfx906Q5KWtFrag & f, const int bq8_offset,
    const int u0, const int u1, const int u2, const int u3,
    const float d80, const float d81) {
    int sc0, sc1, m0, m1;
    gfx906_kq_scales(f.sw0, f.sw1, f.sw2, bq8_offset, sc0, sc1, m0, m1);
    const int vh0 = f.qh0w >> bq8_offset;
    const int vh1 = f.qh1w >> bq8_offset;
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    const int v0_0 = ((f.ql0 >> 0) & 0x0F0F0F0F) | (((vh0 >> 0) << 4) & 0x10101010);
    const int v1_0 = ((f.ql1 >> 0) & 0x0F0F0F0F) | (((vh1 >> 0) << 4) & 0x10101010);
    const int v0_1 = ((f.ql0 >> 4) & 0x0F0F0F0F) | (((vh0 >> 1) << 4) & 0x10101010);
    const int v1_1 = ((f.ql1 >> 4) & 0x0F0F0F0F) | (((vh1 >> 1) << 4) & 0x10101010);
    const int d0 = __dp4a(v0_0, u0, __dp4a(v1_0, u1, 0));
    const int e0 = __dp4a(0x01010101, u0, __dp4a(0x01010101, u1, 0));
    const int d1 = __dp4a(v0_1, u2, __dp4a(v1_1, u3, 0));
    const int e1 = __dp4a(0x01010101, u2, __dp4a(0x01010101, u3, 0));
    sumf_d += d80 * (d0 * sc0);
    sumf_m += d80 * (e0 * m0);
    sumf_d += d81 * (d1 * sc1);
    sumf_m += d81 * (e1 * m1);
    const float2 dm5f = __half22float2(f.dm);
    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

template <typename scalar_t, int ncols, bool PREPARED = false>
static __global__ void mul_mat_vec_q5_K_q8_1_fixed_cols_pipe(
    const void * __restrict__ vx, const void * __restrict__ vy,
    scalar_t * __restrict__ dst, const int nrows, const int nvecs,
    const int dst_stride) {
    constexpr int blocks_per_row = ncols / QK_K;
    constexpr int q8_blocks_per_vec = ncols / QK8_1;
    constexpr int blocks_per_warp = VDR_Q5_K_Q8_1_MMVQ * BLOCK_SIZE / QI5_K;

    const int row = blockIdx.x;
    const bool combine_vecs = gguf_mmvq_combine_vecs(nvecs);
    const int first_vec = combine_vecs ? 0 : blockIdx.y;
    const int vec_count = combine_vecs ? nvecs : 1;
    if (row >= nrows || first_vec >= nvecs) {
        return;
    }

    const block_q5_K * x = (const block_q5_K *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int iqs = VDR_Q5_K_Q8_1_MMVQ * (threadIdx.x % (QI5_K / VDR_Q5_K_Q8_1_MMVQ));
    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int i0 = threadIdx.x / (QI5_K / VDR_Q5_K_Q8_1_MMVQ);

    // Only the weight fragment is software-pipelined: weights stream from HBM
    // while activations are L2-hot, and double-buffering them too measurably
    // regresses short-row shapes (bpr <= 24) on gfx906.
    Gfx906Q5KWtFrag wa, wb;
    bool have_a = i0 < blocks_per_row;
    if (have_a) {
        gfx906_q5k_ld_wt(wa, &x[row * blocks_per_row + i0], iqs, bq8_offset);
    }
    int i = i0;
    while (have_a) {
        const int in = i + blocks_per_warp;
        const bool have_b = in < blocks_per_row;
        if (have_b) {
            gfx906_q5k_ld_wt(wb, &x[row * blocks_per_row + in], iqs, bq8_offset);
        }
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            Gfx906Q5KActFrag af;
            gfx906_q5k_ld_act(af, y, first_vec + vec_offset, q8_blocks_per_vec, i, bq8_offset, iqs);
            tmp[vec_offset] += gfx906_q5k_comp(wa, bq8_offset, af.u0, af.u1, af.u2, af.u3, af.d80, af.d81);
        }
        i = in;
        wa = wb;
        have_a = have_b;
    }

    constexpr int warp_size = WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / warp_size;
#pragma unroll
    for (int mask = warp_size/2; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            tmp[vec_offset] += VLLM_SHFL_XOR_SYNC(tmp[vec_offset], mask);
        }
    }

    if constexpr (num_warps == 1) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                if (vec_offset >= vec_count) {
                    break;
                }
                dst[(first_vec + vec_offset)*dst_stride + row] = tmp[vec_offset];
            }
        }
    } else {
        const int lane = threadIdx.x % warp_size;
        const int warp = threadIdx.x / warp_size;
        __shared__ float shared_sum[num_warps];
        if (lane == 0) {
            shared_sum[warp] = tmp[0];
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
        tmp[0] = lane < num_warps ? shared_sum[lane] : 0.0f;
#pragma unroll
        for (int mask = warp_size/2; mask > 0; mask >>= 1) {
            tmp[0] += VLLM_SHFL_XOR_SYNC(tmp[0], mask);
        }
        if (lane == 0) {
            dst[first_vec*dst_stride + row] = tmp[0];
        }
    }
}

template<typename scalar_t, int ncols>
static void mul_mat_vec_q5_K_q8_1_fixed_cols_pipe_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const dim3 block_nums(nrows, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, 1, 1);
    if (nvecs >= 2) {
        mul_mat_vec_q5_K_q8_1_fixed_cols_pipe<scalar_t, ncols, true>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
    } else {
        mul_mat_vec_q5_K_q8_1_fixed_cols_pipe<scalar_t, ncols, false>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
    }
}

// ---- q4_K pipe variant (same scheme as the q5_K pipe above) ----
struct Gfx906Q4KWtFrag {
    int q0, q1;
    int sw0, sw1, sw2;
    half2 dm;
};

static __device__ __forceinline__ void gfx906_q4k_ld_wt(
    Gfx906Q4KWtFrag & f, const block_q4_K * __restrict__ bq,
    const int iqs, const int bq8_offset) {
    const int * q4 = (const int *)(bq->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    f.q0 = q4[0];
    f.q1 = q4[4];
    f.sw0 = *(const int *)&bq->scales[0];
    f.sw1 = *(const int *)&bq->scales[4];
    f.sw2 = *(const int *)&bq->scales[8];
    f.dm = bq->dm;
}

// activation fragment and loader are shared with q5_K (identical layout)

static __device__ __forceinline__ float gfx906_q4k_comp(
    const Gfx906Q4KWtFrag & f, const int bq8_offset,
    const int u0, const int u1, const int u2, const int u3,
    const float d80, const float d81) {
    int sc0, sc1, m0, m1;
    gfx906_kq_scales(f.sw0, f.sw1, f.sw2, bq8_offset, sc0, sc1, m0, m1);
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    const int v0_0 = (f.q0 >> 0) & 0x0F0F0F0F;
    const int v1_0 = (f.q1 >> 0) & 0x0F0F0F0F;
    const int v0_1 = (f.q0 >> 4) & 0x0F0F0F0F;
    const int v1_1 = (f.q1 >> 4) & 0x0F0F0F0F;
    const int d0 = __dp4a(v1_0, u1, __dp4a(v0_0, u0, 0));
    const int e0 = __dp4a(0x01010101, u1, __dp4a(0x01010101, u0, 0));
    const int d1 = __dp4a(v1_1, u3, __dp4a(v0_1, u2, 0));
    const int e1 = __dp4a(0x01010101, u3, __dp4a(0x01010101, u2, 0));
    sumf_d += d80 * (d0 * sc0);
    sumf_m += d80 * (e0 * m0);
    sumf_d += d81 * (d1 * sc1);
    sumf_m += d81 * (e1 * m1);
    const float2 dm4f = __half22float2(f.dm);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

template <typename scalar_t, int ncols, bool PREPARED = false>
static __global__ void mul_mat_vec_q4_K_q8_1_fixed_cols_pipe(
    const void * __restrict__ vx, const void * __restrict__ vy,
    scalar_t * __restrict__ dst, const int nrows, const int nvecs,
    const int dst_stride) {
    constexpr int blocks_per_row = ncols / QK_K;
    constexpr int q8_blocks_per_vec = ncols / QK8_1;
    constexpr int blocks_per_warp = VDR_Q4_K_Q8_1_MMVQ * BLOCK_SIZE / QI4_K;

    const int row = blockIdx.x;
    const bool combine_vecs = gguf_mmvq_combine_vecs(nvecs);
    const int first_vec = combine_vecs ? 0 : blockIdx.y;
    const int vec_count = combine_vecs ? nvecs : 1;
    if (row >= nrows || first_vec >= nvecs) {
        return;
    }

    const block_q4_K * x = (const block_q4_K *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int iqs = VDR_Q4_K_Q8_1_MMVQ * (threadIdx.x % (QI4_K / VDR_Q4_K_Q8_1_MMVQ));
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int i0 = threadIdx.x / (QI4_K / VDR_Q4_K_Q8_1_MMVQ);

    Gfx906Q4KWtFrag wa, wb;
    bool have_a = i0 < blocks_per_row;
    if (have_a) {
        gfx906_q4k_ld_wt(wa, &x[row * blocks_per_row + i0], iqs, bq8_offset);
    }
    int i = i0;
    while (have_a) {
        const int in = i + blocks_per_warp;
        const bool have_b = in < blocks_per_row;
        if (have_b) {
            gfx906_q4k_ld_wt(wb, &x[row * blocks_per_row + in], iqs, bq8_offset);
        }
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            Gfx906Q5KActFrag af;
            gfx906_q5k_ld_act(af, y, first_vec + vec_offset, q8_blocks_per_vec, i, bq8_offset, iqs);
            tmp[vec_offset] += gfx906_q4k_comp(wa, bq8_offset, af.u0, af.u1, af.u2, af.u3, af.d80, af.d81);
        }
        i = in;
        wa = wb;
        have_a = have_b;
    }

    constexpr int warp_size = WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / warp_size;
#pragma unroll
    for (int mask = warp_size/2; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            tmp[vec_offset] += VLLM_SHFL_XOR_SYNC(tmp[vec_offset], mask);
        }
    }

    if constexpr (num_warps == 1) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                if (vec_offset >= vec_count) {
                    break;
                }
                dst[(first_vec + vec_offset)*dst_stride + row] = tmp[vec_offset];
            }
        }
    } else {
        const int lane = threadIdx.x % warp_size;
        const int warp = threadIdx.x / warp_size;
        __shared__ float shared_sum[num_warps];
        if (lane == 0) {
            shared_sum[warp] = tmp[0];
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
        tmp[0] = lane < num_warps ? shared_sum[lane] : 0.0f;
#pragma unroll
        for (int mask = warp_size/2; mask > 0; mask >>= 1) {
            tmp[0] += VLLM_SHFL_XOR_SYNC(tmp[0], mask);
        }
        if (lane == 0) {
            dst[first_vec*dst_stride + row] = tmp[0];
        }
    }
}

template<typename scalar_t, int ncols>
static void mul_mat_vec_q4_K_q8_1_fixed_cols_pipe_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
  const dim3 block_nums(nrows, gguf_mmvq_grid_vecs(nvecs), 1);
  const dim3 block_dims(BLOCK_SIZE, 1, 1);
  if (nvecs >= 2) {
    mul_mat_vec_q4_K_q8_1_fixed_cols_pipe<scalar_t, ncols, true>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
  } else {
    mul_mat_vec_q4_K_q8_1_fixed_cols_pipe<scalar_t, ncols, false>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
  }
}

// ---- q6_K pipe variant ----
struct Gfx906Q6KWtFrag {
    int vl, vhraw;   // raw ql/qh words, vh_shift applied in compute
    int sc0, sc1;    // sign-extended int8 scales
    half d;
};

struct Gfx906Q6KActFrag {
    int u0, u1;      // u[i] for i = 0, 1
    float d80, d81;
};

static __device__ __forceinline__ void gfx906_q6k_ld_wt(
    Gfx906Q6KWtFrag & f, const block_q6_K * __restrict__ bq, const int iqs) {
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    f.vl = get_int_from_uint8(bq->ql, iqs);
    f.vhraw = get_int_from_uint8(bq->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4));
    f.sc0 = bq->scales[scale_offset];
    f.sc1 = bq->scales[scale_offset + 4];
    f.d = bq->d;
}

static __device__ __forceinline__ void gfx906_q6k_ld_act(
    Gfx906Q6KActFrag & f, const block_q8_1 * __restrict__ y,
    const int vec, const int q8_blocks_per_vec, const int i,
    const int bq8_offset, const int iqs) {
    const block_q8_1 * yb = y + vec * q8_blocks_per_vec + i * (QK_K / QK8_1);
    const block_q8_1 * b0 = yb + bq8_offset;
    const block_q8_1 * b1 = yb + bq8_offset + 2;
    f.d80 = __low2float(b0->ds);
    f.d81 = __low2float(b1->ds);
    f.u0 = get_int_from_int8_aligned(b0->qs, iqs % QI8_1);
    f.u1 = get_int_from_int8_aligned(b1->qs, iqs % QI8_1);
}

static __device__ __forceinline__ float gfx906_q6k_comp(
    const Gfx906Q6KWtFrag & f, const int vh_shift,
    const int u0, const int u1, const float d80, const float d81) {
    const int vh = f.vhraw >> vh_shift;
    const int vi0 = __vsubss4(((f.vl >> 0) & 0x0F0F0F0F) | (((vh >> 0) << 4) & 0x30303030), 0x20202020);
    const int vi1 = __vsubss4(((f.vl >> 4) & 0x0F0F0F0F) | (((vh >> 4) << 4) & 0x30303030), 0x20202020);
    float sumf = 0.0f;
    sumf += d80 * (__dp4a(vi0, u0, 0) * f.sc0);
    sumf += d81 * (__dp4a(vi1, u1, 0) * f.sc1);
    return __half2float(f.d) * sumf;
}

template <typename scalar_t, int ncols, bool PREPARED = false>
static __global__ void mul_mat_vec_q6_K_q8_1_fixed_cols_pipe(
    const void * __restrict__ vx, const void * __restrict__ vy,
    scalar_t * __restrict__ dst, const int nrows, const int nvecs,
    const int dst_stride) {
    constexpr int blocks_per_row = ncols / QK_K;
    constexpr int q8_blocks_per_vec = ncols / QK8_1;
    constexpr int blocks_per_warp = VDR_Q6_K_Q8_1_MMVQ * BLOCK_SIZE / QI6_K;

    const int row = blockIdx.x;
    const bool combine_vecs = gguf_mmvq_combine_vecs(nvecs);
    const int first_vec = combine_vecs ? 0 : blockIdx.y;
    const int vec_count = combine_vecs ? nvecs : 1;
    if (row >= nrows || first_vec >= nvecs) {
        return;
    }

    const block_q6_K * x = (const block_q6_K *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int iqs = threadIdx.x % QI6_K;
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
    const int i0 = threadIdx.x / QI6_K;

    Gfx906Q6KWtFrag wa, wb;
    const int iters = (blocks_per_row - 1 - i0) / blocks_per_warp + 1;
    gfx906_q6k_ld_wt(wa, &x[row * blocks_per_row + i0], iqs);
#pragma unroll 2
    for (int k = 0; k < iters; ++k) {
        const int i = i0 + k * blocks_per_warp;
        const int in = i + blocks_per_warp;
        if (in < blocks_per_row) {
            gfx906_q6k_ld_wt(wb, &x[row * blocks_per_row + in], iqs);
        }
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            Gfx906Q6KActFrag af;
            gfx906_q6k_ld_act(af, y, first_vec + vec_offset, q8_blocks_per_vec, i, bq8_offset, iqs);
            tmp[vec_offset] += gfx906_q6k_comp(wa, vh_shift, af.u0, af.u1, af.d80, af.d81);
        }
        wa = wb;
    }

    constexpr int warp_size = WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / warp_size;
#pragma unroll
    for (int mask = warp_size/2; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            tmp[vec_offset] += VLLM_SHFL_XOR_SYNC(tmp[vec_offset], mask);
        }
    }

    if constexpr (num_warps == 1) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                if (vec_offset >= vec_count) {
                    break;
                }
                dst[(first_vec + vec_offset)*dst_stride + row] = tmp[vec_offset];
            }
        }
    } else {
        const int lane = threadIdx.x % warp_size;
        const int warp = threadIdx.x / warp_size;
        __shared__ float shared_sum[num_warps];
        if (lane == 0) {
            shared_sum[warp] = tmp[0];
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
        tmp[0] = lane < num_warps ? shared_sum[lane] : 0.0f;
#pragma unroll
        for (int mask = warp_size/2; mask > 0; mask >>= 1) {
            tmp[0] += VLLM_SHFL_XOR_SYNC(tmp[0], mask);
        }
        if (lane == 0) {
            dst[first_vec*dst_stride + row] = tmp[0];
        }
    }
}

template<typename scalar_t, int ncols>
static void mul_mat_vec_q6_K_q8_1_fixed_cols_pipe_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
  const dim3 block_nums(nrows, gguf_mmvq_grid_vecs(nvecs), 1);
  const dim3 block_dims(BLOCK_SIZE, 1, 1);
  if (nvecs >= 2) {
    mul_mat_vec_q6_K_q8_1_fixed_cols_pipe<scalar_t, ncols, true>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
  } else {
    mul_mat_vec_q6_K_q8_1_fixed_cols_pipe<scalar_t, ncols, false>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
  }
}


// per-byte subtract of 32 without cross-byte borrow; the bytes passed in are
// 0..63 so the result -32..31 never saturates and this is bit-identical to
// __vsubss4(x, 0x20202020) while staying 3 plain logic ops
static __device__ __forceinline__ int gguf_q6_psub32(int x) {
    return (int)(((x | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
}

// ---- planar-repack q6_K MMVQ variants (gfx906) ----
// The tensor buffer is repacked at load time into four byte-conserving
// planes (ql | qh | scales | d) so every field access becomes a wide,
// aligned, coalesced load. Standalone prototypes measured +15% on tall
// shapes with the 2-rows-per-block form and +13% on the wide ones with the
// single-row form vs the in-service generic kernel.
struct gguf_planar_q6_k {
    static constexpr int ql_stride = 128;  // bytes per superblock
    static constexpr int qh_stride = 64;
    static constexpr int sc_stride = 16;
    static constexpr int d_stride = 2;
    static __host__ __device__ size_t ql_off(int rows, int nb) { return 0; }
    static __host__ __device__ size_t qh_off(int rows, int nb) { return (size_t)rows * nb * ql_stride; }
    static __host__ __device__ size_t sc_off(int rows, int nb) { return qh_off(rows, nb) + (size_t)rows * nb * qh_stride; }
    static __host__ __device__ size_t d_off(int rows, int nb) { return sc_off(rows, nb) + (size_t)rows * nb * sc_stride; }
};

template <typename scalar_t, int VECS>
static __global__ void __launch_bounds__(64)
mul_mat_vec_q6_K_q8_1_planar(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                             scalar_t * __restrict__ dst, const int ncols, const int nrows,
                             const int dst_stride) {
    const int row = blockIdx.x;

    const int nb = ncols / QK_K;
    const int iqs = threadIdx.x % QI6_K;  // QI6_K = 32
    const size_t base = (size_t)row * nb;
    const uint8_t * qlr = vx + gguf_planar_q6_k::ql_off(nrows, nb) + base * 128;
    const uint8_t * qhr = vx + gguf_planar_q6_k::qh_off(nrows, nb) + base * 64;
    const uint8_t * scr = vx + gguf_planar_q6_k::sc_off(nrows, nb) + base * 16;
    const __half * dr = (const __half *)(vx + gguf_planar_q6_k::d_off(nrows, nb)) + base;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    float tmp[VECS];
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) tmp[vo] = 0.f;
    for (int i = threadIdx.x / QI6_K; i < nb; i += 64 / QI6_K) {
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vl = *(const int *)(qlr + i * 128 + 4 * iqs);
        const int vh = (*(const int *)(qhr + i * 64 + 32 * (iqs / 16) + 4 * (iqs % 8))) >> vh_shift;
        const int8_t * sc = (const int8_t *)scr + i * 16 + scale_offset;
        const float d = __half2float(dr[i]);
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            const block_q8_1 * yb = y + vo * (nrows_y / QK8_1) + i * (QK_K / QK8_1);
            int u[2]; float d8[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                u[j] = get_int_from_int8_aligned(yb[bq8_offset + 2 * j].qs, iqs % QI8_1);
                d8[j] = __half2float(__low2half(yb[bq8_offset + 2 * j].ds));
            }
            float sumf = 0.f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int scv = sc[4 * j];
                const int vil = (vl >> (4 * j)) & 0x0F0F0F0F;
                const int vih = ((vh >> (4 * j)) << 4) & 0x30303030;
                sumf += d8[j] * (__dp4a(gguf_q6_psub32(vil | vih), u[j], 0) * scv);
            }
            tmp[vo] += d * sumf;
        }
    }
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            tmp[vo] += VLLM_SHFL_XOR_SYNC(tmp[vo], mask);
        }
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            dst[vo * dst_stride + row] = tmp[vo];
        }
    }
}

template <typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_planar_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(64, 1, 1);
    const int stride = dst_stride < 0 ? nrows : dst_stride;
    switch (nvecs) {
      case 2:
        mul_mat_vec_q6_K_q8_1_planar<scalar_t, 2>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
      case 3:
        mul_mat_vec_q6_K_q8_1_planar<scalar_t, 3>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
      default:
        mul_mat_vec_q6_K_q8_1_planar<scalar_t, 4>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
    }
}

// two rows per block: better memory-level parallelism on tall matrices
template <typename scalar_t>
static __global__ void __launch_bounds__(128)
mul_mat_vec_q6_K_q8_1_planar_row_tile2(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                                       scalar_t * __restrict__ dst, const int ncols, const int nrows,
                                       const int dst_stride) {
    const int row = blockIdx.x * 2 + threadIdx.y;
    if (row >= nrows) return;
    const int nb = ncols / QK_K;
    const int iqs = threadIdx.x % QI6_K;
    const size_t base = (size_t)row * nb;
    const uint8_t * qlr = vx + gguf_planar_q6_k::ql_off(nrows, nb) + base * 128;
    const uint8_t * qhr = vx + gguf_planar_q6_k::qh_off(nrows, nb) + base * 64;
    const uint8_t * scr = vx + gguf_planar_q6_k::sc_off(nrows, nb) + base * 16;
    const __half * dr = (const __half *)(vx + gguf_planar_q6_k::d_off(nrows, nb)) + base;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    float tmp = 0.f;
    for (int i = threadIdx.x / QI6_K; i < nb; i += 64 / QI6_K) {
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vl = *(const int *)(qlr + i * 128 + 4 * iqs);
        const int vh = (*(const int *)(qhr + i * 64 + 32 * (iqs / 16) + 4 * (iqs % 8))) >> vh_shift;
        const int8_t * sc = (const int8_t *)scr + i * 16 + scale_offset;
        const block_q8_1 * yb = y + i * (QK_K / QK8_1);
        int u[2]; float d8[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            u[j] = get_int_from_int8_aligned(yb[bq8_offset + 2 * j].qs, iqs % QI8_1);
            d8[j] = __half2float(__low2half(yb[bq8_offset + 2 * j].ds));
        }
        float sumf = 0.f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int scv = sc[4 * j];
            const int vil = (vl >> (4 * j)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4 * j)) << 4) & 0x30303030;
            sumf += d8[j] * (__dp4a(gguf_q6_psub32(vil | vih), u[j], 0) * scv);
        }
        tmp += __half2float(dr[i]) * sumf;
    }
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1)
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    if (threadIdx.x == 0) dst[row] = tmp;
}

template <typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_planar_row_tile2_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_nums((nrows + 1) / 2, 1, 1);
    const dim3 block_dims(64, 2, 1);
    mul_mat_vec_q6_K_q8_1_planar_row_tile2<scalar_t>
        <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, nrows);
}

// Wide-load form: one row per 64-lane block, each lane covers a 4-word group
// (32 weights). For a group start iqs0 = 4*g the ql window (16B), the qh
// window (16B), the scale byte pair, the qh shift and the q8 block pair are
// all group-uniform, so the lane's weights arrive as two fully-consumed int4
// loads per superblock instead of eight 4B loads (the diagnosis that mattered
// on gfx906: in-flight bytes per request, not sector efficiency). The word
// select stays compile-time inside the unrolled t loop -- a runtime index
// into an int4 makes the compiler split it back into scalar loads. Measured
// standalone on gfx906: 610-636 GB/s on the main shapes vs 470-510 for the
// generic kernel and 570-620 for the row-tile planar form; row tiles, a
// software pipeline, 8-word groups and a q8_1 y-plane all measured slower.
template <typename scalar_t, int VECS>
static __global__ void __launch_bounds__(64)
mul_mat_vec_q6_K_q8_1_planar_wide(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                                  scalar_t * __restrict__ dst, const int ncols, const int nrows,
                                  const int dst_stride) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int nb = ncols / QK_K;
    const int lane = threadIdx.x;
    constexpr int NG = 8;              // word groups per superblock (32/4)
    constexpr int STEP = 8;            // superblocks in flight per block (64/8)
    const int g = lane % NG;
    const int iqs0 = 4 * g;
    const int bofs = 4 * (iqs0 / 16) + (iqs0 % 16) / 8;    // q8 block pair base
    const int scbase = 8 * (iqs0 / 16) + (iqs0 % 16) / 4;  // scale byte base
    const int sh = 2 * ((iqs0 % 16) / 8);                  // qh shift
    const int qlb = 4 * iqs0;                             // 16B-aligned ql window
    const int qhb = 4 * (8 * (iqs0 / 16) + iqs0 % 8);     // 16B-aligned qh window
    const int uw0 = iqs0 % 8;                             // first u word (0 or 4)
    const size_t base = (size_t)row * nb;
    const uint8_t * qlr = vx + gguf_planar_q6_k::ql_off(nrows, nb) + base * 128;
    const uint8_t * qhr = vx + gguf_planar_q6_k::qh_off(nrows, nb) + base * 64;
    const uint8_t * scr = vx + gguf_planar_q6_k::sc_off(nrows, nb) + base * 16;
    const __half * dr = (const __half *)(vx + gguf_planar_q6_k::d_off(nrows, nb)) + base;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    float tmp[VECS];
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) tmp[vo] = 0.f;
    for (int i = lane / NG; i < nb; i += STEP) {
        const int4 qlg = *(const int4 *)(qlr + i * 128 + qlb);
        const int4 qhg = *(const int4 *)(qhr + i * 64 + qhb);
        const int8_t * sc = (const int8_t *)scr + i * 16 + scbase;
        const float d = __half2float(dr[i]);
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            const block_q8_1 * ya = y + vo * (nrows_y / QK8_1) + i * (QK_K / QK8_1) + bofs;
            const block_q8_1 * yb = ya + 2;
            const float d8a = __half2float(__low2half(ya->ds));
            const float d8b = __half2float(__low2half(yb->ds));
            float blk = 0.f;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int u0 = get_int_from_int8_aligned(ya->qs, uw0 + t);
                const int u1 = get_int_from_int8_aligned(yb->qs, uw0 + t);
                const int vl = ((const int *)&qlg)[t];
                const int vhk = ((const int *)&qhg)[t] >> sh;
                const float sumf =
                    d8a * (__dp4a(gguf_q6_psub32((vl & 0x0F0F0F0F) | ((vhk << 4) & 0x30303030)), u0, 0) * sc[0]) +
                    d8b * (__dp4a(gguf_q6_psub32(((vl >> 4) & 0x0F0F0F0F) | (((vhk >> 4) << 4) & 0x30303030)), u1, 0) * sc[4]);
                blk += sumf;
            }
            tmp[vo] += d * blk;
        }
    }
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            tmp[vo] += VLLM_SHFL_XOR_SYNC(tmp[vo], mask);
        }
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            dst[vo * dst_stride + row] = tmp[vo];
        }
    }
}

template <typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_planar_wide_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(64, 1, 1);
    const int stride = dst_stride < 0 ? nrows : dst_stride;
    switch (nvecs) {
      case 1:
        mul_mat_vec_q6_K_q8_1_planar_wide<scalar_t, 1>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
      case 2:
        mul_mat_vec_q6_K_q8_1_planar_wide<scalar_t, 2>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
      case 3:
        mul_mat_vec_q6_K_q8_1_planar_wide<scalar_t, 3>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
      default:
        mul_mat_vec_q6_K_q8_1_planar_wide<scalar_t, 4>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
    }
}


template <typename scalar_t>
static __global__ void __launch_bounds__(64)
mul_mat_vec_q6_K_q8_1_planar_v1(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                                scalar_t * __restrict__ dst, const int ncols, const int nrows) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int nb = ncols / QK_K;
    const int iqs = threadIdx.x % QI6_K;
    const size_t base = (size_t)row * nb;
    const uint8_t * qlr = vx + gguf_planar_q6_k::ql_off(nrows, nb) + base * 128;
    const uint8_t * qhr = vx + gguf_planar_q6_k::qh_off(nrows, nb) + base * 64;
    const uint8_t * scr = vx + gguf_planar_q6_k::sc_off(nrows, nb) + base * 16;
    const __half * dr = (const __half *)(vx + gguf_planar_q6_k::d_off(nrows, nb)) + base;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    float tmp = 0.f;
    for (int i = threadIdx.x / QI6_K; i < nb; i += 64 / QI6_K) {
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vl = *(const int *)(qlr + i * 128 + 4 * iqs);
        const int vh = (*(const int *)(qhr + i * 64 + 32 * (iqs / 16) + 4 * (iqs % 8))) >> vh_shift;
        const int8_t * sc = (const int8_t *)scr + i * 16 + scale_offset;
        const block_q8_1 * yb = y + i * (QK_K / QK8_1);
        int u[2]; float d8[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            u[j] = get_int_from_int8_aligned(yb[bq8_offset + 2 * j].qs, iqs % QI8_1);
            d8[j] = __half2float(__low2half(yb[bq8_offset + 2 * j].ds));
        }
        float sumf = 0.f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int scv = sc[4 * j];
            const int vil = (vl >> (4 * j)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4 * j)) << 4) & 0x30303030;
            sumf += d8[j] * (__dp4a(gguf_q6_psub32(vil | vih), u[j], 0) * scv);
        }
        tmp += __half2float(dr[i]) * sumf;
    }
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1)
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    if (threadIdx.x == 0) dst[row] = tmp;
}

template <typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_planar_v1_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, cudaStream_t stream) {
    mul_mat_vec_q6_K_q8_1_planar_v1<scalar_t>
        <<<nrows, 64, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows);
}

// ---- planar-repack q4_K/q5_K MMVQ variants (gfx906) ----
// Same plane idea as q6_K, adapted to the K-quant scale structure
// (6-bit packed scales + min/dmin pair). Plane layout, byte-conserving:
//   q4_K (144B/sb): ql 128 | scales 12 | dm 4
//   q5_K (176B/sb): ql 128 | qh 32 | scales 12 | dm 4
// Per-thread math replicates vec_dot_q4_K_q8_1 / vec_dot_q5_K_q8_1
// (VDR=2: iqs = 2*(tid%16), superblocks tid/16 strided by 4).
template <bool HAS_QH>
struct gguf_planar_k45 {
    static constexpr int ql_stride = 128;
    static constexpr int qh_stride = 32;  // q5_K only
    static constexpr int sc_stride = 12;
    static constexpr int dm_stride = 4;
    static __host__ __device__ size_t ql_off(int, int) { return 0; }
    static __host__ __device__ size_t qh_off(int rows, int nb) { return (size_t)rows * nb * ql_stride; }
    static __host__ __device__ size_t sc_off(int rows, int nb) {
        return qh_off(rows, nb) + (HAS_QH ? (size_t)rows * nb * qh_stride : 0);
    }
    static __host__ __device__ size_t dm_off(int rows, int nb) {
        return sc_off(rows, nb) + (size_t)rows * nb * sc_stride;
    }
};

// unpack the 6-bit scale/min pair used by both sub-block q8 blocks of this
// thread (verbatim port of the uint16 aux dance in vec_dot_q4_K_q8_1)
static __device__ __forceinline__ void gguf_k45_scales(
    const uint8_t * __restrict__ scb, const int bq8_offset, uint8_t out[4]) {
    const uint16_t * scales = (const uint16_t *)scb;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    out[0] = (uint8_t)aux[0]; out[1] = (uint8_t)(aux[0] >> 8);
    out[2] = (uint8_t)aux[1]; out[3] = (uint8_t)(aux[1] >> 8);
}

// software-pipelined form of the k45 planar kernels: the weight fragment is
// double-buffered in registers (weights stream from the planes one superblock
// ahead while the current one is being multiplied); activations stay L2-hot
// and are loaded per iteration, mirroring the in-service fixed-cols pipe
// kernels (double-buffering them too regresses short-row shapes on gfx906)
struct gguf_k45_wt_frag {
    int v0, v1;
    int vh0, vh1;   // only dereferenced when HAS_QH
    int sw0, sw1, sw2;
    half2 dm;
};

template <bool HAS_QH>
static __device__ __forceinline__ void gguf_k45_ld_wt(
    gguf_k45_wt_frag & f,
    const uint8_t * __restrict__ qlr, const uint8_t * __restrict__ qhr,
    const uint8_t * __restrict__ scr, const half2 * __restrict__ dmr,
    const int sb, const int bq8_offset, const int q4) {
    f.v0 = *(const int *)(qlr + sb * 128 + 16 * bq8_offset + 4 * q4);
    f.v1 = *(const int *)(qlr + sb * 128 + 16 * bq8_offset + 4 * q4 + 16);
    if constexpr (HAS_QH) {
        f.vh0 = *(const int *)(qhr + sb * 32 + 4 * q4);
        f.vh1 = *(const int *)(qhr + sb * 32 + 4 * q4 + 16);
    }
    f.sw0 = *(const int *)(scr + sb * 12);
    f.sw1 = *(const int *)(scr + sb * 12 + 4);
    f.sw2 = *(const int *)(scr + sb * 12 + 8);
    f.dm = dmr[sb];
}

template <bool HAS_QH>
static __device__ __forceinline__ float gguf_k45_pipe_comp(
    const gguf_k45_wt_frag & f, const int bq8_offset,
    const block_q8_1 * __restrict__ ysb, const int q4) {
    const block_q8_1 * b0 = ysb + bq8_offset + 0;
    const block_q8_1 * b1 = ysb + bq8_offset + 1;
    const float d80 = __low2float(b0->ds);
    const float d81 = __low2float(b1->ds);
    const int * q8a = (const int *)b0->qs + q4;
    const int * q8b = (const int *)b1->qs + q4;
    const int u0 = q8a[0];
    const int u1 = q8a[4];
    const int u2 = q8b[0];
    const int u3 = q8b[4];
    int sc0, sc1, m0, m1;
    gfx906_kq_scales(f.sw0, f.sw1, f.sw2, bq8_offset, sc0, sc1, m0, m1);
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    if constexpr (HAS_QH) {
        const int vh0 = f.vh0 >> bq8_offset;
        const int vh1 = f.vh1 >> bq8_offset;
        const int v0_0 = ((f.v0 >> 0) & 0x0F0F0F0F) | (((vh0 >> 0) << 4) & 0x10101010);
        const int v1_0 = ((f.v1 >> 0) & 0x0F0F0F0F) | (((vh1 >> 0) << 4) & 0x10101010);
        const int v0_1 = ((f.v0 >> 4) & 0x0F0F0F0F) | (((vh0 >> 1) << 4) & 0x10101010);
        const int v1_1 = ((f.v1 >> 4) & 0x0F0F0F0F) | (((vh1 >> 1) << 4) & 0x10101010);
        const int d0 = __dp4a(v0_0, u0, __dp4a(v1_0, u1, 0));
        const int e0 = __dp4a(0x01010101, u0, __dp4a(0x01010101, u1, 0));
        const int d1 = __dp4a(v0_1, u2, __dp4a(v1_1, u3, 0));
        const int e1 = __dp4a(0x01010101, u2, __dp4a(0x01010101, u3, 0));
        sumf_d += d80 * (d0 * sc0);
        sumf_m += d80 * (e0 * m0);
        sumf_d += d81 * (d1 * sc1);
        sumf_m += d81 * (e1 * m1);
    } else {
        const int v0_0 = (f.v0 >> 0) & 0x0F0F0F0F;
        const int v1_0 = (f.v1 >> 0) & 0x0F0F0F0F;
        const int v0_1 = (f.v0 >> 4) & 0x0F0F0F0F;
        const int v1_1 = (f.v1 >> 4) & 0x0F0F0F0F;
        const int d0 = __dp4a(v1_0, u1, __dp4a(v0_0, u0, 0));
        const int e0 = __dp4a(0x01010101, u1, __dp4a(0x01010101, u0, 0));
        const int d1 = __dp4a(v1_1, u3, __dp4a(v0_1, u2, 0));
        const int e1 = __dp4a(0x01010101, u3, __dp4a(0x01010101, u2, 0));
        sumf_d += d80 * (d0 * sc0);
        sumf_m += d80 * (e0 * m0);
        sumf_d += d81 * (d1 * sc1);
        sumf_m += d81 * (e1 * m1);
    }
    const float2 dmf = __half22float2(f.dm);
    return dmf.x * sumf_d - dmf.y * sumf_m;
}

// core dot for one superblock; qh pointer is only dereferenced when HAS_QH
template <bool HAS_QH>
static __device__ __forceinline__ float gguf_k45_vec_dot(
    const int v0, const int v1, const int vh0, const int vh1,
    const block_q8_1 * __restrict__ yb, const uint8_t * __restrict__ sc,
    const half2 dm, const int bq8_offset, const int q4) {
    int u[2 * QR4_K];
    float d8[QR4_K];
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = yb + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int * q8 = (const int *)bq8i->qs + q4;
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.f;
    float sumf_m = 0.f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        if constexpr (HAS_QH) {
            const int vl0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
            const int vl1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
            const int vh0i = ((vh0 >> i) << 4) & 0x10101010;
            const int vh1i = ((vh1 >> i) << 4) & 0x10101010;
            const int dot1 = __dp4a(vl0i | vh0i, u[2 * i + 0], __dp4a(vl1i | vh1i, u[2 * i + 1], 0));
            const int dot2 = __dp4a(0x01010101, u[2 * i + 0], __dp4a(0x01010101, u[2 * i + 1], 0));
            sumf_d += d8[i] * (dot1 * sc[i]);
            sumf_m += d8[i] * (dot2 * sc[2 + i]);
        } else {
            const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
            const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
            const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i + 0], 0));
            const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i + 0], 0));
            sumf_d += d8[i] * (dot1 * sc[i]);
            sumf_m += d8[i] * (dot2 * sc[2 + i]);
        }
    }
    const float2 dmf = __half22float2(dm);
    return dmf.x * sumf_d - dmf.y * sumf_m;
}

// multi-vector combine form (vecs 2-4)
template <typename scalar_t, int VECS, bool HAS_QH>
static __global__ void __launch_bounds__(64)
mul_mat_vec_k45_q8_1_planar(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                            scalar_t * __restrict__ dst, const int ncols, const int nrows,
                            const int dst_stride) {
    const int row = blockIdx.x;
    const int nb = ncols / QK_K;
    const int iqs = 2 * (threadIdx.x % 16);
    const size_t base = (size_t)row * nb;
    const uint8_t * qlr = vx + gguf_planar_k45<HAS_QH>::ql_off(nrows, nb) + base * 128;
    const uint8_t * qhr = vx + gguf_planar_k45<HAS_QH>::qh_off(nrows, nb) + base * 32;
    const uint8_t * scr = vx + gguf_planar_k45<HAS_QH>::sc_off(nrows, nb) + base * 12;
    const half2 * dmr = (const half2 *)(vx + gguf_planar_k45<HAS_QH>::dm_off(nrows, nb)) + base;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    float tmp[VECS];
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) tmp[vo] = 0.f;
    for (int i = threadIdx.x / 16; i < nb; i += 4) {
        const int bq8_offset = 2 * ((iqs / 2) / 4);
        const int q4 = (iqs / 2) % 4;
        const int v0 = *(const int *)(qlr + i * 128 + 16 * bq8_offset + 4 * q4);
        const int v1 = *(const int *)(qlr + i * 128 + 16 * bq8_offset + 4 * q4 + 16);
        int vh0 = 0, vh1 = 0;
        if constexpr (HAS_QH) {
            vh0 = (*(const int *)(qhr + i * 32 + 4 * q4)) >> bq8_offset;
            vh1 = (*(const int *)(qhr + i * 32 + 4 * q4 + 16)) >> bq8_offset;
        }
        uint8_t sc[4];
        gguf_k45_scales(scr + i * 12, bq8_offset, sc);
        const half2 dm = dmr[i];
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            const block_q8_1 * yb = y + vo * (nrows_y / QK8_1) + i * (QK_K / QK8_1);
            tmp[vo] += gguf_k45_vec_dot<HAS_QH>(v0, v1, vh0, vh1, yb, sc, dm, bq8_offset, q4);
        }
    }
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            tmp[vo] += VLLM_SHFL_XOR_SYNC(tmp[vo], mask);
        }
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            dst[vo * dst_stride + row] = tmp[vo];
        }
    }
}

template <typename scalar_t, bool HAS_QH>
static void mul_mat_vec_k45_q8_1_planar_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(64, 1, 1);
    const int stride = dst_stride < 0 ? nrows : dst_stride;
    switch (nvecs) {
      case 2:
        mul_mat_vec_k45_q8_1_planar<scalar_t, 2, HAS_QH>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
      case 3:
        mul_mat_vec_k45_q8_1_planar<scalar_t, 3, HAS_QH>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
      default:
        mul_mat_vec_k45_q8_1_planar<scalar_t, 4, HAS_QH>
            <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, stride);
        break;
    }
}

// single-vector form, one row per block
template <typename scalar_t, bool HAS_QH>
static __global__ void __launch_bounds__(64)
mul_mat_vec_k45_q8_1_planar_v1(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                               scalar_t * __restrict__ dst, const int ncols, const int nrows) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int nb = ncols / QK_K;
    const int iqs = 2 * (threadIdx.x % 16);
    const size_t base = (size_t)row * nb;
    const uint8_t * qlr = vx + gguf_planar_k45<HAS_QH>::ql_off(nrows, nb) + base * 128;
    const uint8_t * qhr = vx + gguf_planar_k45<HAS_QH>::qh_off(nrows, nb) + base * 32;
    const uint8_t * scr = vx + gguf_planar_k45<HAS_QH>::sc_off(nrows, nb) + base * 12;
    const half2 * dmr = (const half2 *)(vx + gguf_planar_k45<HAS_QH>::dm_off(nrows, nb)) + base;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int q4 = (iqs / 2) % 4;

    float tmp = 0.f;
    gguf_k45_wt_frag wa, wb;
    int i = threadIdx.x / 16;
    bool have_a = i < nb;
    if (have_a) {
        gguf_k45_ld_wt<HAS_QH>(wa, qlr, qhr, scr, dmr, i, bq8_offset, q4);
    }
    while (have_a) {
        const int in = i + 4;
        const bool have_b = in < nb;
        if (have_b) {
            gguf_k45_ld_wt<HAS_QH>(wb, qlr, qhr, scr, dmr, in, bq8_offset, q4);
        }
        tmp += gguf_k45_pipe_comp<HAS_QH>(wa, bq8_offset, y + i * (QK_K / QK8_1), q4);
        i = in;
        wa = wb;
        have_a = have_b;
    }
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1)
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    if (threadIdx.x == 0) dst[row] = tmp;
}

template <typename scalar_t, bool HAS_QH>
static void mul_mat_vec_k45_q8_1_planar_v1_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, cudaStream_t stream) {
    mul_mat_vec_k45_q8_1_planar_v1<scalar_t, HAS_QH>
        <<<nrows, 64, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows);
}

// Wide-load form: each lane covers a group of 4 consecutive p (p = iqs/2,
// i.e. 64 weights per lane per superblock). For a group start p0 = 4*g the
// payload words {p0%4..+3} and {+4..+7} form two 16B-aligned int4 windows in
// the ql plane, the qh words likewise, and the q8 block pair, the packed
// scale index and the qh shift (= bq8_offset = 2*g) are group-uniform. The
// per-p arithmetic mirrors gguf_k45_vec_dot with the q8 d8 scales and u words
// hoisted out of the t loop (the helper form reloaded them per word).
// Integer accumulation applies the shared scales once per q8 block; even
// q5_K's scaled integer sum fits exactly in float32.
template <typename scalar_t, int VECS, bool HAS_QH, bool INT_ACC = false>
static __global__ void __launch_bounds__(64)
mul_mat_vec_k45_q8_1_planar_wide(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                                 scalar_t * __restrict__ dst, const int ncols, const int nrows,
                                 const int dst_stride) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int nb = ncols / QK_K;
    const int lane = threadIdx.x;
    constexpr int NG = 4;              // p-groups per superblock (16 p / 4)
    constexpr int STEP = 16;           // superblocks in flight per block (64/4)
    const int g = lane % NG;
    const int bofs = 2 * g;            // bq8_offset = 2*(p0/4)
    const int qsb = 16 * bofs;         // payload region base byte in the ql plane
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    float tmp[VECS];
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) tmp[vo] = 0.f;
    for (int i = lane / NG; i < nb; i += STEP) {
        // activation: both q8 blocks of the pair, u words 0..7 each
        float d8[2][VECS];
        int ua[8][VECS], ub[8][VECS];
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            const block_q8_1 * ya = y + vo * (nrows_y / QK8_1) + i * (QK_K / QK8_1) + bofs;
            const block_q8_1 * yb = ya + 1;
            d8[0][vo] = __low2float(ya->ds);
            d8[1][vo] = __low2float(yb->ds);
            const int * qa = (const int *)ya->qs;
            const int * qb = (const int *)yb->qs;
#pragma unroll
            for (int w = 0; w < 8; ++w) ua[w][vo] = qa[w];
#pragma unroll
            for (int w = 0; w < 8; ++w) ub[w][vo] = qb[w];
        }
        {
            const size_t base = (size_t)row * nb + i;
            const uint8_t * qlr = vx + gguf_planar_k45<HAS_QH>::ql_off(nrows, nb) + base * 128;
            const int4 qs0 = *(const int4 *)(qlr + qsb);
            const int4 qs1 = *(const int4 *)(qlr + qsb + 16);
            int4 qh0, qh1;
            if constexpr (HAS_QH) {
                const uint8_t * qhr = vx + gguf_planar_k45<HAS_QH>::qh_off(nrows, nb) + base * 32;
                qh0 = *(const int4 *)(qhr);
                qh1 = *(const int4 *)(qhr + 16);
            }
            uint8_t sc4[4];
            gguf_k45_scales(vx + gguf_planar_k45<HAS_QH>::sc_off(nrows, nb) + base * 12, bofs, sc4);
            const float2 dmf = __half22float2(
                *(const half2 *)(vx + gguf_planar_k45<HAS_QH>::dm_off(nrows, nb) + base * 4));
#pragma unroll
            for (int vo = 0; vo < VECS; ++vo) {
                float sumf_d = 0.f;
                float sumf_m = 0.f;
                int sumi_d[2] = {0, 0};
                int sumi_m[2] = {0, 0};
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int v0 = ((const int *)&qs0)[t];
                    const int v1 = ((const int *)&qs1)[t];
                    int vh0 = 0, vh1 = 0;
                    if constexpr (HAS_QH) {
                        vh0 = ((const int *)&qh0)[t] >> bofs;
                        vh1 = ((const int *)&qh1)[t] >> bofs;
                    }
#pragma unroll
                    for (int ii = 0; ii < 2; ++ii) {
                        const int u0 = ii ? ub[t][vo] : ua[t][vo];
                        const int u1 = ii ? ub[t + 4][vo] : ua[t + 4][vo];
                        int v0i, v1i;
                        if constexpr (HAS_QH) {
                            v0i = ((v0 >> (4 * ii)) & 0x0F0F0F0F) | (((vh0 >> ii) << 4) & 0x10101010);
                            v1i = ((v1 >> (4 * ii)) & 0x0F0F0F0F) | (((vh1 >> ii) << 4) & 0x10101010);
                        } else {
                            v0i = (v0 >> (4 * ii)) & 0x0F0F0F0F;
                            v1i = (v1 >> (4 * ii)) & 0x0F0F0F0F;
                        }
                        if constexpr (INT_ACC) {
                            sumi_d[ii] = __dp4a(
                                v0i, u0, __dp4a(v1i, u1, sumi_d[ii]));
                            sumi_m[ii] = __dp4a(
                                0x01010101, u0,
                                __dp4a(0x01010101, u1, sumi_m[ii]));
                        } else {
                            const int dot1 = __dp4a(v0i, u0, __dp4a(v1i, u1, 0));
                            const int dot2 = __dp4a(0x01010101, u0, __dp4a(0x01010101, u1, 0));
                            sumf_d += d8[ii][vo] * (dot1 * (ii ? sc4[1] : sc4[0]));
                            sumf_m += d8[ii][vo] * (dot2 * (ii ? sc4[3] : sc4[2]));
                        }
                    }
                }
                if constexpr (INT_ACC) {
                    sumf_d = d8[0][vo] * (sumi_d[0] * sc4[0]) +
                             d8[1][vo] * (sumi_d[1] * sc4[1]);
                    sumf_m = d8[0][vo] * (sumi_m[0] * sc4[2]) +
                             d8[1][vo] * (sumi_m[1] * sc4[3]);
                }
                tmp[vo] += dmf.x * sumf_d - dmf.y * sumf_m;
            }
        }
    }
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) {
        float t = tmp[vo];
#pragma unroll
        for (int mask = 32; mask > 0; mask >>= 1)
            t += VLLM_SHFL_XOR_SYNC(t, mask);
        if (threadIdx.x == 0) dst[vo * dst_stride + row] = t;
    }
}

template <typename scalar_t, bool HAS_QH>
static void mul_mat_vec_k45_q8_1_planar_wide_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, const int nvecs, cudaStream_t stream,
        const int dst_stride = -1) {
    const int stride = dst_stride < 0 ? nrows : dst_stride;
    const uint8_t * wx = (const uint8_t *)vx;
    const char * int_acc_env = std::getenv("VLLM_GGUF_K45_INT_ACC");
    const bool int_acc = int_acc_env == nullptr || int_acc_env[0] != '0';
#define K45W_LAUNCH(VECS_)                                                 \
    do {                                                                   \
        if (int_acc) {                                                     \
            mul_mat_vec_k45_q8_1_planar_wide<scalar_t, VECS_, HAS_QH, true>   \
                <<<nrows, 64, 0, stream>>>(wx, vy, dst, ncols, nrows, stride); \
        } else {                                                           \
            mul_mat_vec_k45_q8_1_planar_wide<scalar_t, VECS_, HAS_QH>          \
                <<<nrows, 64, 0, stream>>>(wx, vy, dst, ncols, nrows, stride); \
        }                                                                  \
    } while (0)
    switch (nvecs) {
      case 1:  K45W_LAUNCH(1); break;
      case 2:  K45W_LAUNCH(2); break;
      case 3:  K45W_LAUNCH(3); break;
      default: K45W_LAUNCH(4); break;
    }
#undef K45W_LAUNCH
}

// two rows per block for tall matrices; deliberately NOT software-pipelined:
// the two-row form already has enough memory-level parallelism and the
// double-buffered fragment measurably regressed it (register pressure)
template <typename scalar_t, bool HAS_QH>
static __global__ void __launch_bounds__(128)
mul_mat_vec_k45_q8_1_planar_row_tile2(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                                      scalar_t * __restrict__ dst, const int ncols, const int nrows,
                                      const int dst_stride) {
    const int row = blockIdx.x * 2 + threadIdx.y;
    if (row >= nrows) return;
    const int nb = ncols / QK_K;
    const int iqs = 2 * (threadIdx.x % 16);
    const size_t base = (size_t)row * nb;
    const uint8_t * qlr = vx + gguf_planar_k45<HAS_QH>::ql_off(nrows, nb) + base * 128;
    const uint8_t * qhr = vx + gguf_planar_k45<HAS_QH>::qh_off(nrows, nb) + base * 32;
    const uint8_t * scr = vx + gguf_planar_k45<HAS_QH>::sc_off(nrows, nb) + base * 12;
    const half2 * dmr = (const half2 *)(vx + gguf_planar_k45<HAS_QH>::dm_off(nrows, nb)) + base;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int q4 = (iqs / 2) % 4;

    float tmp = 0.f;
    for (int i = threadIdx.x / 16; i < nb; i += 4) {
        const int v0 = *(const int *)(qlr + i * 128 + 16 * bq8_offset + 4 * q4);
        const int v1 = *(const int *)(qlr + i * 128 + 16 * bq8_offset + 4 * q4 + 16);
        int vh0 = 0, vh1 = 0;
        if constexpr (HAS_QH) {
            vh0 = (*(const int *)(qhr + i * 32 + 4 * q4)) >> bq8_offset;
            vh1 = (*(const int *)(qhr + i * 32 + 4 * q4 + 16)) >> bq8_offset;
        }
        uint8_t sc[4];
        gguf_k45_scales(scr + i * 12, bq8_offset, sc);
        const block_q8_1 * yb = y + i * (QK_K / QK8_1);
        tmp += gguf_k45_vec_dot<HAS_QH>(v0, v1, vh0, vh1, yb, sc, dmr[i], bq8_offset, q4);
    }
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1)
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    if (threadIdx.x == 0) dst[row] = tmp;
}

template <typename scalar_t, bool HAS_QH>
static void mul_mat_vec_k45_q8_1_planar_row_tile2_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_nums((nrows + 1) / 2, 1, 1);
    const dim3 block_dims(64, 2, 1);
    mul_mat_vec_k45_q8_1_planar_row_tile2<scalar_t, HAS_QH>
        <<<block_nums, block_dims, 0, stream>>>((const uint8_t *)vx, vy, dst, ncols, nrows, nrows);
}

#endif  // USE_ROCM

template<typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template <typename scalar_t, int ncols, bool PREPARED = false>
static __global__ void mul_mat_vec_q6_K_q8_1_fixed_cols(
    const void * __restrict__ vx, const void * __restrict__ vy,
    scalar_t * __restrict__ dst, const int nrows, const int nvecs,
    const int dst_stride) {
    constexpr int blocks_per_row = ncols / QK_K;
    constexpr int q8_blocks_per_vec = ncols / QK8_1;
    constexpr int blocks_per_warp = VDR_Q6_K_Q8_1_MMVQ * BLOCK_SIZE / QI6_K;

    const int row = blockIdx.x;
    const bool combine_vecs = gguf_mmvq_combine_vecs(nvecs);
    const int first_vec = combine_vecs ? 0 : blockIdx.y;
    const int vec_count = combine_vecs ? nvecs : 1;
    if (row >= nrows || first_vec >= nvecs) {
        return;
    }

    const block_q6_K * x = (const block_q6_K *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if constexpr (PREPARED) {
      // Multi-vector path: the weight-side unpack runs once per weight block
      // and is shared across the combined vectors.
      MmvqWeightQ6K w6;
      for (int i = threadIdx.x / QI6_K; i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iqs = threadIdx.x % QI6_K;
        mmvq_prepare_q6_K(&x[ibx], iqs, w6);
        const block_q8_1 * y_row =
            &y[first_vec * q8_blocks_per_vec + i * (QK_K / QK8_1)];
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
          if (vec_offset >= vec_count) {
            break;
          }
          tmp[vec_offset] += mmvq_dot_q6_K(
              w6, y_row + vec_offset * q8_blocks_per_vec, iqs);
        }
      }
    } else {
      // Single-vector instantiation: the original one-call vec_dot, whose
      // codegen measured faster when there is nothing to share across
      // vectors.
      for (int i = threadIdx.x / QI6_K; i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iqs = threadIdx.x % QI6_K;
        const int iby = first_vec * q8_blocks_per_vec + i * (QK_K / QK8_1);
        tmp[0] += vec_dot_q6_K_q8_1(&x[ibx], &y[iby], iqs);
      }
    }

    constexpr int warp_size = WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / warp_size;
#pragma unroll
    for (int mask = warp_size/2; mask > 0; mask >>= 1) {
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            tmp[vec_offset] += VLLM_SHFL_XOR_SYNC(tmp[vec_offset], mask);
        }
    }

    if constexpr (num_warps == 1) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
                if (vec_offset >= vec_count) {
                    break;
                }
                dst[(first_vec + vec_offset) * dst_stride + row] = tmp[vec_offset];
            }
        }
    } else {
        const int lane = threadIdx.x % warp_size;
        const int warp = threadIdx.x / warp_size;
        __shared__ float shared_sum[num_warps];
        if (lane == 0) {
            shared_sum[warp] = tmp[0];
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
        tmp[0] = lane < num_warps ? shared_sum[lane] : 0.0f;
#pragma unroll
        for (int mask = warp_size/2; mask > 0; mask >>= 1) {
            tmp[0] += VLLM_SHFL_XOR_SYNC(tmp[0], mask);
        }
        if (lane == 0) {
            dst[first_vec * dst_stride + row] = tmp[0];
        }
    }
}

template<typename scalar_t, int ncols>
static void mul_mat_vec_q6_K_q8_1_fixed_cols_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
  const dim3 block_nums(nrows, gguf_mmvq_grid_vecs(nvecs), 1);
  const dim3 block_dims(BLOCK_SIZE, 1, 1);
  if (nvecs >= 2) {
    mul_mat_vec_q6_K_q8_1_fixed_cols<scalar_t, ncols, true>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
  } else {
    mul_mat_vec_q6_K_q8_1_fixed_cols<scalar_t, ncols, false>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
  }
}

template<typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_col2560_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    mul_mat_vec_q6_K_q8_1_fixed_cols_cuda<scalar_t, 2560>(
        vx, vy, dst, nrows, nvecs, stream, dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_y2_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    constexpr int mmv_y = 2;
    const int block_num_y = (nrows + mmv_y - 1) / mmv_y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, mmv_y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI4_XS, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

// Wide-load form on the ORIGINAL block layout (no planar repack needed): the
// 136-byte block stride keeps every payload window 8-byte aligned, so each
// lane covers one whole 32-weight group (16B payload = 2x u64, one q8_1
// block, one 6-bit scale). Lanes j = lane%8 tile the superblock's 128B
// payload contiguously; the gfx906 v_perm lattice lookup keeps the per-weight
// cost at the dp4a level. Same dp4a chain order as vec_dot_iq4_xs_q8_1, so
// only the final float reduction order differs.
template <typename scalar_t, int VECS>
static __global__ void __launch_bounds__(64)
mul_mat_vec_iq4xs_q8_1_wide(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                            scalar_t * __restrict__ dst, const int ncols, const int nrows,
                            const int dst_stride) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int nb = ncols / QK_K;
    const int lane = threadIdx.x;
    const int j = lane & 7;              // 32-weight group inside the superblock
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    float tmp[VECS];
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) tmp[vo] = 0.f;
    for (int i = lane >> 3; i < nb; i += 8) {
        float d8[VECS];
        int q8w[8][VECS];
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            const block_q8_1 * y8 = y + (size_t)vo * (nrows_y / QK8_1) + i * (QK_K / QK8_1) + j;
            d8[vo] = __low2float(y8->ds);
            const int * qa = (const int *)y8->qs;
#pragma unroll
            for (int w = 0; w < 8; ++w) q8w[w][vo] = qa[w];
        }
        {
            const uint8_t * wrow = vx + (size_t)row * nb * sizeof(block_iq4_xs);
            // 8B header: d | scales_h | scales_l[4], 8-byte aligned (136 % 8 == 0)
            const unsigned long long hd =
                *(const unsigned long long *)(wrow + (size_t)i * sizeof(block_iq4_xs));
            const uint32_t lo = (uint32_t)hd;
            const uint32_t hi = (uint32_t)(hd >> 32);
            const int ls = (int)(((hi >> (8 * (j >> 1) + 4 * (j & 1))) & 0x0F)
                                 | ((((lo >> 16) >> (2 * j)) & 0x03) << 4));
            const float d = __half2float(__ushort_as_half((unsigned short)lo)) * (ls - 32);
            // payload window: 16B at block+8+16*j, 8-byte aligned
            const uint8_t * pw = wrow + (size_t)i * sizeof(block_iq4_xs) + 8 + 16 * j;
            const unsigned long long q0 = *(const unsigned long long *)pw;
            const unsigned long long q1 = *(const unsigned long long *)(pw + 8);
#pragma unroll
            for (int vo = 0; vo < VECS; ++vo) {
                int sumi = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint32_t aux = (uint32_t)(t == 0 ? q0 : t == 1 ? (q0 >> 32)
                                                        : t == 2 ? q1 : (q1 >> 32));
                    const int v1 = lookup_iq4nl_values_gfx906(aux & 0x0F0F0F0F);
                    const int v2 = lookup_iq4nl_values_gfx906((aux >> 4) & 0x0F0F0F0F);
                    sumi = __dp4a(v1, q8w[t][vo], sumi);
                    sumi = __dp4a(v2, q8w[t + 4][vo], sumi);
                }
                tmp[vo] += d * (d8[vo] * (float)sumi);
            }
        }
    }
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) {
        float t = tmp[vo];
#pragma unroll
        for (int mask = 32; mask > 0; mask >>= 1)
            t += VLLM_SHFL_XOR_SYNC(t, mask);
        if (threadIdx.x == 0) dst[vo * dst_stride + row] = t;
    }
}

template <typename scalar_t>
static void mul_mat_vec_iq4xs_q8_1_wide_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, const int nvecs, cudaStream_t stream,
        const int dst_stride = -1) {
    const int stride = dst_stride < 0 ? nrows : dst_stride;
    const uint8_t * wx = (const uint8_t *)vx;
#define IQ4W_LAUNCH(VECS_)                                                  \
    do {                                                                    \
        mul_mat_vec_iq4xs_q8_1_wide<scalar_t, VECS_>                        \
            <<<nrows, 64, 0, stream>>>(wx, vy, dst, ncols, nrows, stride);  \
    } while (0)
    switch (nvecs) {
      case 1:  IQ4W_LAUNCH(1); break;
      case 2:  IQ4W_LAUNCH(2); break;
      case 3:  IQ4W_LAUNCH(3); break;
      default: IQ4W_LAUNCH(4); break;
    }
#undef IQ4W_LAUNCH
}


template<typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

// ---------------------------------------------------------------------------
// gfx906 wide-load MMVQ for the i-quant family (iq2_xxs/xs/s, iq3_xxs/s,
// iq4_nl, iq1_s). Same skeleton as mul_mat_vec_iq4xs_q8_1_wide: 64 lanes,
// lane j = 32-weight group inside the superblock, superblocks warp-strided,
// activation staged into registers, butterfly shuffle reduce. The per-group
// weight decode is plugged in through a decoder type; the decode math is the
// one validated by the MMQ loaders (grid lookups + SIMD sign negation).
//
// MODE 0 (SINGLE): one scale per 32 weights -> one integer accumulator
// MODE 1 (DUAL):   16-weight scale granularity (iq2_xs / iq2_s)
// MODE 2 (IQ1S):   d1q scale pair; the delta term consumes the q8_1 block
//                  sum (ds.y) like the MMVQ reference

template <typename scalar_t, int VECS, typename DEC>
static __global__ void __launch_bounds__(64)
mul_mat_vec_iq_wide(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                    scalar_t * __restrict__ dst, const int ncols, const int nrows,
                    const int dst_stride) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int nb = ncols / QK_K;
    const int lane = threadIdx.x;
    const int j = lane & 7;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    const uint8_t * wrow = vx + (size_t)row * nb * DEC::BLOCK_BYTES;
    float tmp[VECS];
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) tmp[vo] = 0.f;
    for (int i = lane >> 3; i < nb; i += 8) {
        float d8[VECS], d8y[VECS];
        int q8w[8][VECS];
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            const block_q8_1 * y8 = y + (size_t)vo * (nrows_y / QK8_1) + i * (QK_K / QK8_1) + j;
            const half2 ds = y8->ds;
            d8[vo]  = __low2float(ds);
            d8y[vo] = __high2float(ds);
            const int * qa = (const int *)y8->qs;
#pragma unroll
            for (int w = 0; w < 8; ++w) q8w[w][vo] = qa[w];
        }
        int vw[8];
        float s0, s1;
        DEC::group(wrow, i, j, vw, s0, s1);
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            if (DEC::MODE == 0) {
                int sumi = 0;
#pragma unroll
                for (int w = 0; w < 8; ++w) sumi = __dp4a(vw[w], q8w[w][vo], sumi);
                tmp[vo] += d8[vo] * (s0 * (float)sumi);
            } else if (DEC::MODE == 1) {
                int sumi0 = 0, sumi1 = 0;
#pragma unroll
                for (int w = 0; w < 4; ++w)
                    sumi0 = __dp4a(vw[w], q8w[w][vo], sumi0);
#pragma unroll
                for (int w = 4; w < 8; ++w)
                    sumi1 = __dp4a(vw[w], q8w[w][vo], sumi1);
                tmp[vo] += d8[vo] * (s0 * (float)sumi0 + s1 * (float)sumi1);
            } else {
                int sumi = 0;
#pragma unroll
                for (int w = 0; w < 8; ++w) sumi = __dp4a(vw[w], q8w[w][vo], sumi);
                tmp[vo] += d8[vo] * (s0 * (float)sumi) + d8y[vo] * s1;
            }
        }
    }
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) {
        float t = tmp[vo];
#pragma unroll
        for (int mask = 32; mask > 0; mask >>= 1)
            t += VLLM_SHFL_XOR_SYNC(t, mask);
        if (threadIdx.x == 0) dst[vo * dst_stride + row] = t;
    }
}

// iq1_m wide: the packed iq1s grid word stays packed (nibble planes unpacked
// here) and the scale/delta pairs are per 8-weight word, so the dot mixes
// ints and floats word by word, exactly like the MMQ kernel
template <typename scalar_t, int VECS>
static __global__ void __launch_bounds__(64)
mul_mat_vec_iq1m_wide(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                      scalar_t * __restrict__ dst, const int ncols, const int nrows,
                      const int dst_stride) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int nb = ncols / QK_K;
    const int lane = threadIdx.x;
    const int j = lane & 7;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    const uint8_t * wrow = vx + (size_t)row * nb * sizeof(block_iq1_m);
    float tmp[VECS];
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) tmp[vo] = 0.f;
    for (int i = lane >> 3; i < nb; i += 8) {
        float d8[VECS];
        int q8w[8][VECS];
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            const block_q8_1 * y8 = y + (size_t)vo * (nrows_y / QK8_1) + i * (QK_K / QK8_1) + j;
            d8[vo] = __low2float(y8->ds);
            const int * qa = (const int *)y8->qs;
#pragma unroll
            for (int w = 0; w < 8; ++w) q8w[w][vo] = qa[w];
        }
        const block_iq1_m * blk = (const block_iq1_m *)(wrow + (size_t)i * sizeof(block_iq1_m));
        const uint16_t * sc = (const uint16_t *) blk->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d = __half2float(scale.f16);
        const int tmps = sc[j >> 1] >> (6*(j & 1));
        const float sc0 = d * (2*((tmps >> 0) & 0x07) + 1);
        const float sc1 = d * (2*((tmps >> 3) & 0x07) + 1);
#pragma unroll
        for (int vo = 0; vo < VECS; ++vo) {
            float acc = 0.0f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int qhl = blk->qh[2*j + (l >> 1)] >> (4 * (l & 1));
                const int grid = (int)iq1s_grid_gpu[blk->qs[4*j + l] | ((qhl & 0x07) << 8)];
                const float scl = (l >> 1) ? sc1 : sc0;
                const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
                const int s  = __dp4a((grid >> 4) & 0x0F0F0F0F, q8w[2*l+1][vo],
                                      __dp4a(grid & 0x0F0F0F0F, q8w[2*l][vo], 0));
                const int sy = __dp4a(0x01010101, q8w[2*l+1][vo],
                                      __dp4a(0x01010101, q8w[2*l][vo], 0));
                acc += scl * ((float)s + delta * (float)sy);
            }
            tmp[vo] += d8[vo] * acc;
        }
    }
#pragma unroll
    for (int vo = 0; vo < VECS; ++vo) {
        float t = tmp[vo];
#pragma unroll
        for (int mask = 32; mask > 0; mask >>= 1)
            t += VLLM_SHFL_XOR_SYNC(t, mask);
        if (threadIdx.x == 0) dst[vo * dst_stride + row] = t;
    }
}

// ---- per-type group decoders (addressing + decode validated in mmq.cuh) ----

struct dec_iq2_xxs {
    static constexpr int MODE = 0;
    static constexpr int BLOCK_BYTES = (int)sizeof(block_iq2_xxs);
    static __device__ __forceinline__ void group(const uint8_t * wrow, int i, int j,
                                                 int (&vw)[8], float & s0, float & s1) {
        const block_iq2_xxs * blk = (const block_iq2_xxs *)(wrow + (size_t)i * BLOCK_BYTES);
        const uint16_t * q2 = blk->qs + 4*j;
        // the four grid indices are the BYTES of q2[0..1], one per lookup
        const uint32_t idxw = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux32 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const float d = __half2float(blk->d) * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g_lo = ((const uint32_t *)(iq2xxs_grid + ((idxw >> (8*l)) & 0xFF)))[0];
            const uint32_t g_hi = ((const uint32_t *)(iq2xxs_grid + ((idxw >> (8*l)) & 0xFF)))[1];
            iq_sign_words(g_lo, g_hi, ksigns_iq2xs[(aux32 >> (7*l)) & 127], vw[2*l], vw[2*l+1]);
        }
        s0 = s1 = d;
    }
};

struct dec_iq2_xs {
    static constexpr int MODE = 1;
    static constexpr int BLOCK_BYTES = (int)sizeof(block_iq2_xs);
    static __device__ __forceinline__ void group(const uint8_t * wrow, int i, int j,
                                                 int (&vw)[8], float & s0, float & s1) {
        const block_iq2_xs * blk = (const block_iq2_xs *)(wrow + (size_t)i * BLOCK_BYTES);
        const uint16_t * q2 = blk->qs + 4*j;
        const float d = __half2float(blk->d) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g_lo = ((const uint32_t *)(iq2xs_grid + (q2[l] & 511)))[0];
            const uint32_t g_hi = ((const uint32_t *)(iq2xs_grid + (q2[l] & 511)))[1];
            iq_sign_words(g_lo, g_hi, ksigns_iq2xs[q2[l] >> 9], vw[2*l], vw[2*l+1]);
        }
        s0 = d * (0.5f + (blk->scales[j] & 0xf));
        s1 = d * (0.5f + (blk->scales[j] >>  4));
    }
};

struct dec_iq2_s {
    static constexpr int MODE = 1;
    static constexpr int BLOCK_BYTES = (int)sizeof(block_iq2_s);
    static __device__ __forceinline__ void group(const uint8_t * wrow, int i, int j,
                                                 int (&vw)[8], float & s0, float & s1) {
        const block_iq2_s * blk = (const block_iq2_s *)(wrow + (size_t)i * BLOCK_BYTES);
        const float d = __half2float(blk->d) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = blk->qs[4*j + l] | ((blk->qh[j] << (8 - 2*l)) & 0x300);
            const uint32_t g_lo = ((const uint32_t *)(iq2s_grid + idx))[0];
            const uint32_t g_hi = ((const uint32_t *)(iq2s_grid + idx))[1];
            iq_sign_words(g_lo, g_hi, blk->qs[QK_K/8 + 4*j + l], vw[2*l], vw[2*l+1]);
        }
        s0 = d * (0.5f + (blk->scales[j] & 0xf));
        s1 = d * (0.5f + (blk->scales[j] >>  4));
    }
};

struct dec_iq3_xxs {
    static constexpr int MODE = 0;
    static constexpr int BLOCK_BYTES = (int)sizeof(block_iq3_xxs);
    static __device__ __forceinline__ void group(const uint8_t * wrow, int i, int j,
                                                 int (&vw)[8], float & s0, float & s1) {
        const block_iq3_xxs * blk = (const block_iq3_xxs *)(wrow + (size_t)i * BLOCK_BYTES);
        const uint8_t * q3 = blk->qs + 8*j;
        const uint16_t * gas = (const uint16_t *)(blk->qs + QK_K/4) + 2*j;
        const uint32_t aux32 = (uint32_t)gas[0] | ((uint32_t)gas[1] << 16);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t grid1 = iq3xxs_grid[q3[2*l + 0]];
            const uint32_t grid2 = iq3xxs_grid[q3[2*l + 1]];
            const uint32_t * sgn = (const uint32_t *)(ksigns64 + ((aux32 >> (7*l)) & 127));
            vw[2*l + 0] = (int)__vsub4(grid1 ^ sgn[0], sgn[0]);
            vw[2*l + 1] = (int)__vsub4(grid2 ^ sgn[1], sgn[1]);
        }
        s0 = s1 = __half2float(blk->d) * (0.5f + (aux32 >> 28)) * 0.5f;
    }
};

struct dec_iq3_s {
    static constexpr int MODE = 0;
    static constexpr int BLOCK_BYTES = (int)sizeof(block_iq3_s);
    static __device__ __forceinline__ void group(const uint8_t * wrow, int i, int j,
                                                 int (&vw)[8], float & s0, float & s1) {
        const block_iq3_s * blk = (const block_iq3_s *)(wrow + (size_t)i * BLOCK_BYTES);
        const uint8_t * qs = blk->qs + 8*j;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t grid1 = iq3xs_grid[qs[2*l + 0] | ((blk->qh[j] << (8 - 2*l)) & 256)];
            const uint32_t grid2 = iq3xs_grid[qs[2*l + 1] | ((blk->qh[j] << (7 - 2*l)) & 256)];
            iq_sign_words(grid1, grid2, blk->signs[4*j + l], vw[2*l], vw[2*l+1]);
        }
        const float s = 0.5f + ((blk->scales[j >> 1] >> 4*(j & 1)) & 0xf);
        s0 = s1 = __half2float(blk->d) * s * 0.5f;
    }
};

struct dec_iq4_nl {
    static constexpr int MODE = 0;
    static constexpr int BLOCK_BYTES = 8 * (int)sizeof(block_iq4_nl);   // 8 sub-blocks = 256 weights
    static __device__ __forceinline__ void group(const uint8_t * wrow, int i, int j,
                                                 int (&vw)[8], float & s0, float & s1) {
        // 16B payload of sub-block j: word m -> lo quads m, hi quads m+4
        const uint8_t * sb = wrow + (size_t)i * BLOCK_BYTES + (size_t)j * sizeof(block_iq4_nl);
        const half d = *(const half *)sb;
        const uint32_t * pw = (const uint32_t *)(sb + 2);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            vw[m]     = iq4xs_lookup_word(pw[m] & 0x0F0F0F0F);
            vw[m + 4] = iq4xs_lookup_word((pw[m] >> 4) & 0x0F0F0F0F);
        }
        s0 = s1 = __half2float(d);
    }
};

struct dec_iq1_s {
    static constexpr int MODE = 2;
    static constexpr int BLOCK_BYTES = (int)sizeof(block_iq1_s);
    static __device__ __forceinline__ void group(const uint8_t * wrow, int i, int j,
                                                 int (&vw)[8], float & s0, float & s1) {
        const block_iq1_s * blk = (const block_iq1_s *)(wrow + (size_t)i * BLOCK_BYTES);
        const int qh = blk->qh[j];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int grid = (int)iq1s_grid_gpu[blk->qs[4*j + l] | (((qh >> 3*l) & 0x07) << 8)];
            vw[2*l + 0] = grid & 0x0F0F0F0F;
            vw[2*l + 1] = (grid >> 4) & 0x0F0F0F0F;
        }
        const float d1 = __half2float(blk->d) * (((qh >> 11) & 0x0E) + 1);
        const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
        s0 = d1;
        s1 = d1 * delta;
    }
};

#if defined(USE_ROCM)
// Four independent rows share the small codebooks. The global-table variant
// isolates the benefit of LDS from the signed-byte decode and block mapping.
template <typename scalar_t, int VECS, int IQ_KIND, bool LDS>
static __global__ void __launch_bounds__(256)
mul_mat_vec_iq_lut(const uint8_t * __restrict__ vx,
                   const block_q8_1 * __restrict__ y,
                   scalar_t * __restrict__ dst, const int ncols,
                   const int nrows, const int dst_stride) {
    constexpr bool IQ1 = IQ_KIND == 1;
    constexpr bool IQ2XXS = IQ_KIND == 0;
    constexpr bool IQ3XXS = IQ_KIND == 2;
    constexpr bool IQ3S = IQ_KIND == 3;
    constexpr bool IQ2XS = IQ_KIND == 4;
    constexpr bool IQ2S = IQ_KIND == 5;
    constexpr bool IQ1M = IQ_KIND == 6;
    constexpr bool IQ4NL = IQ_KIND == 7;
    constexpr bool IQ4XS = IQ_KIND == 8;
    constexpr int table_size = (IQ1 || IQ1M) ? 2048
        : (IQ2XXS ? 768 : (IQ2XS ? 1280 : (IQ2S ? 2560
        : (IQ3S ? 1024 : ((IQ4NL || IQ4XS) ? 1 : 512)))));
    __shared__ uint32_t table[LDS ? table_size : 1];
    if constexpr (LDS) {
        for (int k = threadIdx.x; k < table_size; k += blockDim.x) {
            if constexpr (IQ1 || IQ1M) {
                table[k] = (uint32_t)iq1s_grid_gpu[k];
            } else if constexpr (IQ2XXS) {
                table[k] = k < 512
                    ? ((const uint32_t *)iq2xxs_grid)[k]
                    : ((const uint32_t *)ksigns64)[k - 512];
            } else if constexpr (IQ2XS) {
                table[k] = k < 1024 ? ((const uint32_t *)iq2xs_grid)[k]
                    : ((const uint32_t *)ksigns64)[k - 1024];
            } else if constexpr (IQ2S) {
                if (k < 2048) table[k] = ((const uint32_t *)iq2s_grid)[k];
                else {
                    const int sign = (k - 2048) / 2;
                    const int nibble = (k & 1) ? sign >> 4 : sign & 15;
                    table[k] = __vcmpeq4((nibble * 0x01010101) & 0x08040201, 0x08040201);
                }
            } else if constexpr (IQ4NL || IQ4XS) {
                table[k] = 0;
            } else {
                if constexpr (IQ3XXS) {
                    table[k] = k < 256 ? iq3xxs_grid[k]
                        : ((const uint32_t *)ksigns64)[k - 256];
                } else {
                    if (k < 512) table[k] = iq3xs_grid[k];
                    else {
                        const int sign = (k - 512) / 2;
                        const int nibble = (k & 1) ? sign >> 4 : sign & 15;
                        table[k] = __vcmpeq4((nibble * 0x01010101) & 0x08040201, 0x08040201);
                    }
                }
            }
        }
        __syncthreads();
    }
    extern __shared__ int staged_activation[];
    if constexpr ((IQ4NL || IQ4XS) && LDS && VECS == 1) {
        const int words = ((ncols + 511) / 512 * 512) / 32 * 9;
        for (int k = threadIdx.x; k < words; k += blockDim.x)
            staged_activation[k] = ((const int *)y)[k];
        __syncthreads();
        y = (const block_q8_1 *)staged_activation;
    }
    const int row = 4 * blockIdx.x + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const int lane = threadIdx.x & 63;
    const int j = lane & 7;
    const int nb = ncols / QK_K;
    const int ny = ((ncols + 511) / 512 * 512) / QK8_1;
    constexpr int block_bytes = IQ1 ? sizeof(block_iq1_s)
        : (IQ2XXS ? sizeof(block_iq2_xxs)
        : (IQ3XXS ? sizeof(block_iq3_xxs) : (IQ3S ? sizeof(block_iq3_s)
        : (IQ2XS ? sizeof(block_iq2_xs) : (IQ2S ? sizeof(block_iq2_s)
        : (IQ1M ? sizeof(block_iq1_m) : (IQ4NL ? 8*sizeof(block_iq4_nl)
        : sizeof(block_iq4_xs))))))));
    const uint8_t * wrow = vx + (size_t)row * nb * block_bytes;
    float tmp[VECS] = {};
    for (int i = lane >> 3; i < nb; i += 8) {
        uint32_t indices, aux;
        float scale, scale_other = 0.f;
        float sub_scale[4], sub_delta[4];
        if constexpr (IQ1) {
            const block_iq1_s * b = (const block_iq1_s *)(wrow + (size_t)i * block_bytes);
            indices = get_int_b2(b->qs, j);
            aux = b->qh[j];
            scale = __half2float(b->d) * (((aux >> 11) & 14) + 1);
        } else if constexpr (IQ2XXS) {
            const block_iq2_xxs * b = (const block_iq2_xxs *)(wrow + (size_t)i * block_bytes);
            const uint16_t * q = b->qs + 4 * j;
            indices = uint32_t(q[0]) | (uint32_t(q[1]) << 16);
            aux = uint32_t(q[2]) | (uint32_t(q[3]) << 16);
            scale = __half2float(b->d) * (0.5f + (aux >> 28));
        } else if constexpr (IQ3XXS) {
            const block_iq3_xxs * b = (const block_iq3_xxs *)(wrow + (size_t)i * block_bytes);
            const uint16_t * gas = (const uint16_t *)(b->qs + QK_K / 4) + 2 * j;
            aux = uint32_t(gas[0]) | (uint32_t(gas[1]) << 16);
            indices = 0;
            scale = __half2float(b->d) * (0.5f + (aux >> 28)) * 0.5f;
        } else if constexpr (IQ3S) {
            const block_iq3_s * b = (const block_iq3_s *)(wrow + (size_t)i * block_bytes);
            indices = aux = 0;
            const float s = 0.5f + ((b->scales[j >> 1] >> (4 * (j & 1))) & 0xf);
            scale = __half2float(b->d) * s * 0.5f;
        } else if constexpr (IQ2XS || IQ2S) {
            const float d = IQ2XS
                ? __half2float(((const block_iq2_xs *)(wrow + (size_t)i * block_bytes))->d)
                : __half2float(((const block_iq2_s *)(wrow + (size_t)i * block_bytes))->d);
            const int sc = IQ2XS
                ? ((const block_iq2_xs *)(wrow + (size_t)i * block_bytes))->scales[j]
                : ((const block_iq2_s *)(wrow + (size_t)i * block_bytes))->scales[j];
            scale = d * 0.25f * (0.5f + (sc & 15));
            scale_other = d * 0.25f * (0.5f + (sc >> 4));
            indices = aux = 0;
        } else if constexpr (IQ1M) {
            const block_iq1_m * b = (const block_iq1_m *)(wrow + (size_t)i * block_bytes);
            const uint16_t * sc = (const uint16_t *)b->scales;
            iq1m_scale_t d;
            d.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                    ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            const int tmps = sc[j >> 1] >> (6 * (j & 1));
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int qh = b->qh[2*j + (l >> 1)] >> (4 * (l & 1));
                sub_scale[l] = 2*((tmps >> (3*(l >> 1))) & 7) + 1;
                sub_delta[l] = -1.0f + IQ1M_DELTA - (qh & 8)*(2.0f*IQ1M_DELTA/8);
            }
            scale = __half2float(d.f16);
            indices = aux = 0;
        } else if constexpr (IQ4NL) {
            const block_iq4_nl * b = (const block_iq4_nl *)(wrow + (size_t)i * block_bytes) + j;
            scale = __half2float(b->d);
            indices = aux = 0;
        } else {
            const block_iq4_xs * b = (const block_iq4_xs *)(wrow + (size_t)i * block_bytes);
            const int ls = ((b->scales_l[j >> 1] >> (4*(j & 1))) & 15) |
                           (((b->scales_h >> (2*j)) & 3) << 4);
            scale = __half2float(b->d) * (ls - 32);
            indices = aux = 0;
        }
        int weight[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            if constexpr (IQ1) {
                const int k = ((indices >> (8*l)) & 255) |
                              (((aux >> (3*l)) & 7) << 8);
                const uint32_t grid = LDS ? table[k] : (uint32_t)iq1s_grid_gpu[k];
                weight[2*l] = grid & 0x0F0F0F0F;
                weight[2*l+1] = (grid >> 4) & 0x0F0F0F0F;
            } else if constexpr (IQ2XXS) {
                const int k = (indices >> (8*l)) & 255;
                const int s = (aux >> (7*l)) & 127;
                const uint32_t lo = LDS ? table[2*k] : ((const uint32_t *)iq2xxs_grid)[2*k];
                const uint32_t hi = LDS ? table[2*k+1] : ((const uint32_t *)iq2xxs_grid)[2*k+1];
                const uint32_t sl = LDS ? table[512+2*s] : ((const uint32_t *)ksigns64)[2*s];
                const uint32_t sh = LDS ? table[512+2*s+1] : ((const uint32_t *)ksigns64)[2*s+1];
                // IQ2_XXS magnitudes are strictly positive, so adding one
                // to a negated byte cannot carry into the next byte.
                weight[2*l] = (lo ^ sl) + (sl & 0x01010101);
                weight[2*l+1] = (hi ^ sh) + (sh & 0x01010101);
            } else if constexpr (IQ3XXS) {
                const block_iq3_xxs * b = (const block_iq3_xxs *)(wrow + (size_t)i * block_bytes);
                const int grid0 = b->qs[8*j + 2*l];
                const int grid1 = b->qs[8*j + 2*l+1];
                const uint32_t lo = LDS ? table[grid0] : iq3xxs_grid[grid0];
                const uint32_t hi = LDS ? table[grid1] : iq3xxs_grid[grid1];
                const int sign = (aux >> (7*l)) & 127;
                const uint32_t sl = LDS ? table[256+2*sign]
                    : ((const uint32_t *)ksigns64)[2*sign];
                const uint32_t sh = LDS ? table[256+2*sign+1]
                    : ((const uint32_t *)ksigns64)[2*sign+1];
                weight[2*l] = (lo ^ sl) + (sl & 0x01010101);
                weight[2*l+1] = (hi ^ sh) + (sh & 0x01010101);
            } else if constexpr (IQ3S) {
                const block_iq3_s * b = (const block_iq3_s *)(wrow + (size_t)i * block_bytes);
                const int grid0 = b->qs[8*j + 2*l] | ((b->qh[j] << (8 - 2*l)) & 256);
                const int grid1 = b->qs[8*j + 2*l+1] | ((b->qh[j] << (7 - 2*l)) & 256);
                const int s0 = b->signs[4*j+l];
                const uint32_t g0 = LDS ? table[grid0] : iq3xs_grid[grid0];
                const uint32_t g1 = LDS ? table[grid1] : iq3xs_grid[grid1];
                if constexpr (LDS) {
                    const uint32_t sl = table[512+2*s0];
                    const uint32_t sh = table[512+2*s0+1];
                    weight[2*l] = (g0 ^ sl) + (sl & 0x01010101);
                    weight[2*l+1] = (g1 ^ sh) + (sh & 0x01010101);
                } else {
                    iq_sign_words(g0, g1, s0, weight[2*l], weight[2*l+1]);
                }
            } else if constexpr (IQ2XS || IQ2S) {
                const uint8_t * p = wrow + (size_t)i * block_bytes;
                int index, sign;
                if constexpr (IQ2XS) {
                    const uint16_t q = ((const block_iq2_xs *)p)->qs[4*j+l];
                    index = q & 511;
                    sign = q >> 9;
                } else {
                    const block_iq2_s * b = (const block_iq2_s *)p;
                    index = b->qs[4*j+l] | ((b->qh[j] << (8 - 2*l)) & 0x300);
                    sign = b->qs[32+4*j+l];
                }
                const uint32_t * grid = IQ2XS ? (const uint32_t *)iq2xs_grid
                                             : (const uint32_t *)iq2s_grid;
                const uint32_t lo = LDS ? table[2*index] : grid[2*index];
                const uint32_t hi = LDS ? table[2*index+1] : grid[2*index+1];
                if constexpr (IQ2XS) {
                    const uint32_t sl = LDS ? table[1024+2*sign]
                        : ((const uint32_t *)ksigns64)[2*sign];
                    const uint32_t sh = LDS ? table[1024+2*sign+1]
                        : ((const uint32_t *)ksigns64)[2*sign+1];
                    weight[2*l] = (lo ^ sl) + (sl & 0x01010101);
                    weight[2*l+1] = (hi ^ sh) + (sh & 0x01010101);
                } else {
                    if constexpr (LDS) {
                        const uint32_t sl = table[2048+2*sign];
                        const uint32_t sh = table[2048+2*sign+1];
                        weight[2*l] = (lo ^ sl) + (sl & 0x01010101);
                        weight[2*l+1] = (hi ^ sh) + (sh & 0x01010101);
                    } else {
                        iq_sign_words(lo, hi, sign, weight[2*l], weight[2*l+1]);
                    }
                }
            } else if constexpr (IQ1M) {
                const block_iq1_m * b = (const block_iq1_m *)(wrow + (size_t)i * block_bytes);
                const int qh = b->qh[2*j + (l >> 1)] >> (4 * (l & 1));
                const int index = b->qs[4*j+l] | ((qh & 7) << 8);
                const uint32_t grid = LDS ? table[index] : (uint32_t)iq1s_grid_gpu[index];
                weight[2*l] = grid & 0x0f0f0f0f;
                weight[2*l+1] = (grid >> 4) & 0x0f0f0f0f;
            } else {
                const uint8_t * q;
                if constexpr (IQ4NL) {
                    q = ((const block_iq4_nl *)(wrow + (size_t)i * block_bytes))[j].qs;
                } else {
                    q = ((const block_iq4_xs *)(wrow + (size_t)i * block_bytes))->qs + 16*j;
                }
                const uint32_t packed = get_int_b2(q, l);
                weight[l] = lookup_iq4nl_values_gfx906(packed & 0x0f0f0f0f);
                weight[l+4] = lookup_iq4nl_values_gfx906((packed >> 4) & 0x0f0f0f0f);
            }
        }
#pragma unroll
        for (int v = 0; v < VECS; ++v) {
            const block_q8_1 * y8 = y + (size_t)v * ny + 8*i + j;
            const int * qa = (const int *)y8->qs;
            const float2 ds = __half22float2(y8->ds);
            if constexpr (IQ1M) {
                int sumi[2] = {};
                float sumf[2] = {};
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int k = l >> 1;
                    sumi[k] = __dp4a(weight[2*l], qa[2*l],
                                     __dp4a(weight[2*l+1], qa[2*l+1], sumi[k]));
                    const int sy = __dp4a(qa[2*l], 0x01010101,
                                         __dp4a(qa[2*l+1], 0x01010101, 0));
                    sumf[k] += sub_delta[l] * sy;
                }
                tmp[v] += (scale * ds.x) *
                          ((sumi[0] + sumf[0]) * sub_scale[0] +
                           (sumi[1] + sumf[1]) * sub_scale[2]);
                continue;
            }
            int sum0 = 0, sum1 = 0;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                if constexpr (IQ2XS || IQ2S) {
                    sum0 = __dp4a(weight[l], qa[l], sum0);
                    sum1 = __dp4a(weight[l+4], qa[l+4], sum1);
                } else {
                    sum0 = __dp4a(weight[2*l], qa[2*l], sum0);
                    sum1 = __dp4a(weight[2*l+1], qa[2*l+1], sum1);
                }
            }
            if constexpr (IQ1) {
                const float delta = -1.0f + IQ1S_DELTA -
                    (aux & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
                tmp[v] += scale * (ds.x * (sum0 + sum1) + ds.y * delta);
            } else if constexpr (IQ2XXS) {
                tmp[v] += (scale * ds.x * 0.25f) * (sum0 + sum1);
            } else if constexpr (IQ2XS || IQ2S) {
                tmp[v] += ds.x * (scale * sum0 + scale_other * sum1);
            } else if constexpr (IQ3S || IQ3XXS) {
                // Keep the deployed wide kernel's floating point order.
                tmp[v] += ds.x * (scale * (sum0 + sum1));
            } else if constexpr (IQ4XS) {
                tmp[v] += scale * (ds.x * (sum0 + sum1));
            } else if constexpr (IQ4NL) {
                tmp[v] += (scale * ds.x) * (sum0 + sum1);
            } else {
                tmp[v] += ds.x * (scale * sum0) + ds.x * (scale * sum1);
            }
        }
    }
#pragma unroll
    for (int v = 0; v < VECS; ++v) {
        float t = tmp[v];
#pragma unroll
        for (int mask = 32; mask > 0; mask >>= 1)
            t += VLLM_SHFL_XOR_SYNC(t, mask);
        if (lane == 0) dst[v * dst_stride + row] = t;
    }
}

template <typename scalar_t, int IQ_KIND, bool LDS>
static void mul_mat_vec_iq_lut_cuda(
        const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, const int nvecs,
        cudaStream_t stream, const int dst_stride) {
    const int shared_bytes = (IQ_KIND >= 7 && LDS && nvecs == 1)
        ? ((ncols + 511) / 512 * 512) / 32 * 36 : 0;
#define IQL_LAUNCH(V_) \
    mul_mat_vec_iq_lut<scalar_t, V_, IQ_KIND, LDS> \
        <<<(nrows + 3) / 4, 256, shared_bytes, stream>>>( \
            (const uint8_t *)vx, (const block_q8_1 *)vy, dst, \
            ncols, nrows, dst_stride)
    switch (nvecs) {
      case 1: IQL_LAUNCH(1); break;
      case 2: IQL_LAUNCH(2); break;
      case 3: IQL_LAUNCH(3); break;
      case 4: IQL_LAUNCH(4); break;
    }
#undef IQL_LAUNCH
}
#endif

template <typename scalar_t, typename DEC>
static void mul_mat_vec_iq_wide_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, const int nvecs, cudaStream_t stream,
        const int dst_stride = -1) {
    const int stride = dst_stride < 0 ? nrows : dst_stride;
    const uint8_t * wx = (const uint8_t *)vx;
#define IQW_LAUNCH(VECS_)                                                        \
    do {                                                                         \
        mul_mat_vec_iq_wide<scalar_t, VECS_, DEC>                                \
            <<<nrows, 64, 0, stream>>>(wx, vy, dst, ncols, nrows, stride);       \
    } while (0)
    switch (nvecs) {
      case 1:  IQW_LAUNCH(1); break;
      case 2:  IQW_LAUNCH(2); break;
      case 3:  IQW_LAUNCH(3); break;
      default: IQW_LAUNCH(4); break;
    }
#undef IQW_LAUNCH
}

template <typename scalar_t>
static void mul_mat_vec_iq1m_wide_cuda(const void * vx, const void * vy, scalar_t * dst,
        const int ncols, const int nrows, const int nvecs, cudaStream_t stream,
        const int dst_stride = -1) {
    const int stride = dst_stride < 0 ? nrows : dst_stride;
    const uint8_t * wx = (const uint8_t *)vx;
#define IQ1MW_LAUNCH(VECS_)                                                      \
    do {                                                                         \
        mul_mat_vec_iq1m_wide<scalar_t, VECS_>                                   \
            <<<nrows, 64, 0, stream>>>(wx, vy, dst, ncols, nrows, stride);       \
    } while (0)
    switch (nvecs) {
      case 1:  IQ1MW_LAUNCH(1); break;
      case 2:  IQ1MW_LAUNCH(2); break;
      case 3:  IQ1MW_LAUNCH(3); break;
      default: IQ1MW_LAUNCH(4); break;
    }
#undef IQ1MW_LAUNCH
}
