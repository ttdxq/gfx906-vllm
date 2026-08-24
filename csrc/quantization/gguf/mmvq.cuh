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
static __device__ __forceinline__ void gfx906_q5k_scales(
    const Gfx906Q5KWtFrag & f, const int bq8_offset,
    int & sc0, int & sc1, int & m0, int & m1) {
    const unsigned s0 = (unsigned)f.sw0 & 0xffffu;
    const unsigned s1 = ((unsigned)f.sw0 >> 16) & 0xffffu;
    const unsigned s2 = (unsigned)f.sw1 & 0xffffu;
    const unsigned s3 = ((unsigned)f.sw1 >> 16) & 0xffffu;
    const unsigned s4 = (unsigned)f.sw2 & 0xffffu;
    const unsigned s5 = ((unsigned)f.sw2 >> 16) & 0xffffu;
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
    gfx906_q5k_scales(f, bq8_offset, sc0, sc1, m0, m1);
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

template<typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}
