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

template <typename scalar_t, int ncols>
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

    for (int i = threadIdx.x / (QI4_K / VDR_Q4_K_Q8_1_MMVQ);
         i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iqs = VDR_Q4_K_Q8_1_MMVQ *
            (threadIdx.x % (QI4_K / VDR_Q4_K_Q8_1_MMVQ));
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            const int iby = (first_vec + vec_offset) * q8_blocks_per_vec +
                i * (QK_K / QK8_1);
            tmp[vec_offset] += vec_dot_q4_K_q8_1(&x[ibx], &y[iby], iqs);
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
    mul_mat_vec_q4_K_q8_1_fixed_cols<scalar_t, ncols>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
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

template <typename scalar_t, int ncols>
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

    for (int i = threadIdx.x / (QI5_K / VDR_Q5_K_Q8_1_MMVQ);
         i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iqs = VDR_Q5_K_Q8_1_MMVQ *
            (threadIdx.x % (QI5_K / VDR_Q5_K_Q8_1_MMVQ));
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            const int iby = (first_vec + vec_offset) * q8_blocks_per_vec +
                i * (QK_K / QK8_1);
            tmp[vec_offset] += vec_dot_q5_K_q8_1(&x[ibx], &y[iby], iqs);
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
    mul_mat_vec_q5_K_q8_1_fixed_cols<scalar_t, ncols>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_col2560_cuda(const void * vx, const void * vy, scalar_t * dst, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    mul_mat_vec_q5_K_q8_1_fixed_cols_cuda<scalar_t, 2560>(
        vx, vy, dst, nrows, nvecs, stream, dst_stride);
}

template<typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream, const int dst_stride = -1) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
    const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
}

template <typename scalar_t, int ncols>
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

    for (int i = threadIdx.x / QI6_K; i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iqs = threadIdx.x % QI6_K;
#pragma unroll
        for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
            if (vec_offset >= vec_count) {
                break;
            }
            const int iby = (first_vec + vec_offset) * q8_blocks_per_vec +
                i * (QK_K / QK8_1);
            tmp[vec_offset] += vec_dot_q6_K_q8_1(&x[ibx], &y[iby], iqs);
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
    mul_mat_vec_q6_K_q8_1_fixed_cols<scalar_t, ncols>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, nvecs, dst_stride < 0 ? nrows : dst_stride);
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
