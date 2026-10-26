// copied from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmq.cu

// defined in gguf_kernel.cu; reports whether the buffer was produced by
// ggml_planar_pack (plane layout for q4_K/q5_K/q6_K)
bool gguf_planar_is_planar(const void* w);
// forward declaration: the q4_K/q5_K planar kernels below are defined before
// mul_mat_q_planar's definition
template <typename scalar_t, int qk, int qr, int qi, bool need_sum, int mmq_x, int mmq_y, int nwarps,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_planar_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot, bool need_check>
static __device__ __forceinline__ void mul_mat_q_planar(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst);

template <typename scalar_t, int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x, int mmq_y, int nwarps,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void mul_mat_q(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE_GGUF / qi;

    const int & ncols_dst = ncols_y;

    const auto row_dst_0 = blockIdx.x*mmq_y;
    const int & row_x_0 = row_dst_0;

    const auto col_dst_0 = blockIdx.y*mmq_x;
    const int & col_y_0 = col_dst_0;

    int   * tile_x_ql = nullptr;
    half2 * tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    __shared__ int    tile_y_qs[mmq_x * WARP_SIZE_GGUF];
    __shared__ half2  tile_y_ds[mmq_x * WARP_SIZE_GGUF/QI8_1];

    float sum[mmq_y/WARP_SIZE_GGUF][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0*blocks_per_row_x + ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   threadIdx.y, nrows_x-row_x_0-1, threadIdx.x, blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr && ib0 + ir * blocks_per_warp/qr < blocks_per_row_x; ++ir) {
            const auto kqs = ir*WARP_SIZE_GGUF + threadIdx.x;
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); // to prevent out-of-bounds memory accesses
                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];
                const int index_y = (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE_GGUF/QI8_1)) % mmq_x;
                const auto kby = threadIdx.x % (WARP_SIZE_GGUF/QI8_1);
                const int col_y_eff = min(col_y_0 + ids, ncols_y-1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const half2 * dsi_src = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(WARP_SIZE_GGUF/QI8_1) + kby].ds;
                half2       * dsi_dst = &tile_y_ds[ids * (WARP_SIZE_GGUF/QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = __low2float(*dsi_src);
                }
            }

            __syncthreads();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE_GGUF/qr; k < (ir+1)*WARP_SIZE_GGUF/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
                        sum[i/WARP_SIZE_GGUF][j/nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                            threadIdx.x + i, threadIdx.y + j, k);
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const auto col_dst = col_dst_0 + j + threadIdx.y;
        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            const auto row_dst = row_dst_0 + threadIdx.x + i;
            if (row_dst >= nrows_dst) {
                continue;
            }
            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE_GGUF][j/nwarps];
        }
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q4_0  64
#define  MMQ_Y_Q4_0  128
#define NWARPS_Q4_0  8
#else
#define  MMQ_X_Q4_0 4
#define  MMQ_Y_Q4_0 32
#define NWARPS_Q4_0 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_0, 2)
#endif
mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_0;
    const int mmq_y  =  MMQ_Y_Q4_0;
    const int nwarps = NWARPS_Q4_0;

    mul_mat_q<scalar_t, QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int mmq_x  =  MMQ_X_Q4_0;
    int mmq_y  =  MMQ_Y_Q4_0;
    int nwarps = NWARPS_Q4_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q4_1 64
#define  MMQ_Y_Q4_1 128
#define NWARPS_Q4_1 8
#else
#define  MMQ_X_Q4_1 4
#define  MMQ_Y_Q4_1 32
#define NWARPS_Q4_1 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_1, 2)
#endif
mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_1;
    const int mmq_y  =  MMQ_Y_Q4_1;
    const int nwarps = NWARPS_Q4_1;

    mul_mat_q<scalar_t, QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_1_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int mmq_x  =  MMQ_X_Q4_1;
    int mmq_y  =  MMQ_Y_Q4_1;
    int nwarps = NWARPS_Q4_1;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q5_0 64
#define  MMQ_Y_Q5_0 128
#define NWARPS_Q5_0 8
#else
#define  MMQ_X_Q5_0 4
#define  MMQ_Y_Q5_0 32
#define NWARPS_Q5_0 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_0, 2)
#endif
mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_0;
    const int mmq_y  =  MMQ_Y_Q5_0;
    const int nwarps = NWARPS_Q5_0;

    mul_mat_q<scalar_t, QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int mmq_x  =  MMQ_X_Q5_0;
    const int mmq_y  =  MMQ_Y_Q5_0;
    const int nwarps = NWARPS_Q5_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q5_1 64
#define  MMQ_Y_Q5_1 128
#define NWARPS_Q5_1 8
#else
#define  MMQ_X_Q5_1 4
#define  MMQ_Y_Q5_1 32
#define NWARPS_Q5_1 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_1, 2)
#endif
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_1;
    const int mmq_y  =  MMQ_Y_Q5_1;
    const int nwarps = NWARPS_Q5_1;

    mul_mat_q<scalar_t, QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_1_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q5_1;
    const int mmq_y  =  MMQ_Y_Q5_1;
    const int nwarps = NWARPS_Q5_1;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q8_0 64
#define  MMQ_Y_Q8_0 128
#define NWARPS_Q8_0 8
#else
#define  MMQ_X_Q8_0 4
#define  MMQ_Y_Q8_0 32
#define NWARPS_Q8_0 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q8_0, 2)
#endif
mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q8_0;
    const int mmq_y  =  MMQ_Y_Q8_0;
    const int nwarps = NWARPS_Q8_0;

    mul_mat_q<scalar_t, QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q8_0_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q8_0;
    const int mmq_y  =  MMQ_Y_Q8_0;
    const int nwarps = NWARPS_Q8_0;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q2_K 64
#define  MMQ_Y_Q2_K 128
#define NWARPS_Q2_K 8
#else
#define  MMQ_X_Q2_K 4
#define  MMQ_Y_Q2_K 32
#define NWARPS_Q2_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q2_K, 2)
#endif
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q2_K;
    const int mmq_y  =  MMQ_Y_Q2_K;
    const int nwarps = NWARPS_Q2_K;

    mul_mat_q<scalar_t, QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q2_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q2_K;
    const int mmq_y  =  MMQ_Y_Q2_K;
    const int nwarps = NWARPS_Q2_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q3_K 64
#define  MMQ_Y_Q3_K 128
#define NWARPS_Q3_K 8
#else
#define  MMQ_X_Q3_K 4
#define  MMQ_Y_Q3_K 32
#define NWARPS_Q3_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q3_K, 2)
#endif
mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const int mmq_x  =  MMQ_X_Q3_K;
    const int mmq_y  =  MMQ_Y_Q3_K;
    const int nwarps = NWARPS_Q3_K;

    mul_mat_q<scalar_t, QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q3_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int mmq_x  =  MMQ_X_Q3_K;
    const int mmq_y  =  MMQ_Y_Q3_K;
    const int nwarps = NWARPS_Q3_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q4_K 64
#define  MMQ_Y_Q4_K 128
#define NWARPS_Q4_K 8
#else
#define  MMQ_X_Q4_K 4
#define  MMQ_Y_Q4_K 32
#define NWARPS_Q4_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q4_K, 2)
#endif
mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_K;
    const int mmq_y  =  MMQ_Y_Q4_K;
    const int nwarps = NWARPS_Q4_K;

    mul_mat_q<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

// q4_K plane-split kernel: pairs allocate/load_tiles_q4_K_split with the
// shift-free vec_dot (see vecdotq.cuh). 64x64 tile, 8 warps.
template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*8, 2)
#endif
mul_mat_q4_K_split(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  = 64;
    const int mmq_y  = 64;
    const int nwarps = 8;

    mul_mat_q<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K_split<mmq_y>,
        load_tiles_q4_K_split<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat_split>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

// planar-buffer twin of the split kernel above (same tiles, same vec_dot)
template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*8, 2)
#endif
mul_mat_q4_K_split_planar(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  = 64;
    const int mmq_y  = 64;
    const int nwarps = 8;

    mul_mat_q_planar<scalar_t, QK_K, QR4_K, QI4_K, true, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K_split<mmq_y>,
        load_tiles_q4_K_split_planar<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat_split, need_check>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q4_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q4_K;
    const int mmq_y  =  MMQ_Y_Q4_K;
    const int nwarps = NWARPS_Q4_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    static int use_split = -1;
    if (use_split < 0) {
        // Plane-split q4_K path, bitwise-identical to the packed kernel and
        // ~2.6x faster on gfx906 (the packed path re-applies a shift+mask
        // inside every __dp4a for every output column). Set 0 to restore.
        const char * env = getenv("VLLM_GFX906_MMQ_Q4K_SPLIT");
        use_split = env ? atoi(env) : 1;
    }

    if (use_split) {
        // plane-split variant on a 64x64 tile: the doubled x_ql tile stays
        // within the 2-blocks-per-CU LDS budget on gfx906
        constexpr int sx = 64, sy = 64;
        const dim3 split_nums((nrows_x + sy - 1) / sy, (ncols_y + sx - 1) / sx, 1);
        const bool planar = gguf_planar_is_planar(vx);
        if (nrows_x % sy == 0) {
            if (planar) {
                mul_mat_q4_K_split_planar<scalar_t, false><<<split_nums, block_dims, 0, stream>>>
                    (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
            } else {
                mul_mat_q4_K_split<scalar_t, false><<<split_nums, block_dims, 0, stream>>>
                    (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
            }
        } else {
            if (planar) {
                mul_mat_q4_K_split_planar<scalar_t, true><<<split_nums, block_dims, 0, stream>>>
                    (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
            } else {
                mul_mat_q4_K_split<scalar_t, true><<<split_nums, block_dims, 0, stream>>>
                    (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
            }
        }
        return;
    }

    if (gguf_planar_is_planar(vx)) {
        // planar buffers cannot feed the original-layout kernels; the split
        // planar kernel is bitwise-equivalent regardless of the env override
        constexpr int sx = 64, sy = 64;
        const dim3 split_nums((nrows_x + sy - 1) / sy, (ncols_y + sx - 1) / sx, 1);
        if (nrows_x % sy == 0) {
            mul_mat_q4_K_split_planar<scalar_t, false><<<split_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        } else {
            mul_mat_q4_K_split_planar<scalar_t, true><<<split_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        }
        return;
    }

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

#if defined(USE_ROCM)
#define  MMQ_X_Q5_K 64
#define  MMQ_Y_Q5_K 128
#define NWARPS_Q5_K 8
#else
#define  MMQ_X_Q5_K 4
#define  MMQ_Y_Q5_K 32
#define NWARPS_Q5_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_K, 2)
#endif
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_K;
    const int mmq_y  =  MMQ_Y_Q5_K;
    const int nwarps = NWARPS_Q5_K;

    mul_mat_q<scalar_t, QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q5_K, 2)
#endif
mul_mat_q5_K_planar(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_K;
    const int mmq_y  =  MMQ_Y_Q5_K;
    const int nwarps = NWARPS_Q5_K;

    mul_mat_q_planar<scalar_t, QK_K, QR5_K, QI5_K, true, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K_planar<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat, need_check>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q5_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int mmq_x  =  MMQ_X_Q5_K;
    const int mmq_y  =  MMQ_Y_Q5_K;
    const int nwarps = NWARPS_Q5_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    const bool planar = gguf_planar_is_planar(vx);
    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        if (planar) {
            mul_mat_q5_K_planar<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        } else {
            mul_mat_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        }
    } else {
        const bool need_check = true;
        if (planar) {
            mul_mat_q5_K_planar<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        } else {
            mul_mat_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        }
    }
}

// Planar-repack variant of mul_mat_q: same body, but weight tiles
// are loaded from the plane layouts via an extended-signature loader (the
// original load_tiles pointer convention cannot carry the plane bases).
template <typename scalar_t, int qk, int qr, int qi, bool need_sum, int mmq_x, int mmq_y, int nwarps,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_planar_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot, bool need_check>
static __device__ __forceinline__ void mul_mat_q_planar(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE_GGUF / qi;

    const int & ncols_dst = ncols_y;

    const auto row_dst_0 = blockIdx.x*mmq_y;
    const int & row_x_0 = row_dst_0;

    const auto col_dst_0 = blockIdx.y*mmq_x;
    const int & col_y_0 = col_dst_0;

    int   * tile_x_ql = nullptr;
    half2 * tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    __shared__ int    tile_y_qs[mmq_x * WARP_SIZE_GGUF];
    __shared__ half2  tile_y_ds[mmq_x * WARP_SIZE_GGUF/QI8_1];

    float sum[mmq_y/WARP_SIZE_GGUF][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(
                   vx, nrows_x, row_x_0, ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   threadIdx.y, nrows_x-row_x_0-1, threadIdx.x, blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr && ib0 + ir * blocks_per_warp/qr < blocks_per_row_x; ++ir) {
            const auto kqs = ir*WARP_SIZE_GGUF + threadIdx.x;
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); // to prevent out-of-bounds memory accesses
                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];
                const int index_y = (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE_GGUF/QI8_1)) % mmq_x;
                const auto kby = threadIdx.x % (WARP_SIZE_GGUF/QI8_1);
                const int col_y_eff = min(col_y_0 + ids, ncols_y-1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const half2 * dsi_src = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(WARP_SIZE_GGUF/QI8_1) + kby].ds;
                half2       * dsi_dst = &tile_y_ds[ids * (WARP_SIZE_GGUF/QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = __low2float(*dsi_src);
                }
            }

            __syncthreads();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE_GGUF/qr; k < (ir+1)*WARP_SIZE_GGUF/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
                        sum[i/WARP_SIZE_GGUF][j/nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                            threadIdx.x + i, threadIdx.y + j, k);
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const auto col_dst = col_dst_0 + j + threadIdx.y;
        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            const auto row_dst = row_dst_0 + threadIdx.x + i;
            if (row_dst >= nrows_dst) {
                continue;
            }
            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE_GGUF][j/nwarps];
        }
    }
}


#if defined(USE_ROCM)
#define  MMQ_X_Q6_K 64
#define  MMQ_Y_Q6_K 128
#define NWARPS_Q6_K 8
#else
#define  MMQ_X_Q6_K 4
#define  MMQ_Y_Q6_K 32
#define NWARPS_Q6_K 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q6_K, 2)
#endif
mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q6_K;
    const int mmq_y  =  MMQ_Y_Q6_K;
    const int nwarps = NWARPS_Q6_K;

    mul_mat_q<scalar_t, QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_Q6_K, 2)
#endif
mul_mat_q6_K_planar(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q6_K;
    const int mmq_y  =  MMQ_Y_Q6_K;
    const int nwarps = NWARPS_Q6_K;

    mul_mat_q_planar<scalar_t, QK_K, QR6_K, QI6_K, false, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K_planar<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat, need_check>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_q6_K_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_Q6_K;
    const int mmq_y  =  MMQ_Y_Q6_K;
    const int nwarps = NWARPS_Q6_K;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    const bool planar = gguf_planar_is_planar(vx);
    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        if (planar) {
            mul_mat_q6_K_planar<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        } else {
            mul_mat_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        }
    } else {
        const bool need_check = true;
        if (planar) {
            mul_mat_q6_K_planar<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        } else {
            mul_mat_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>
                (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
        }
    }
}


#if defined(USE_ROCM)
// mmq_y=128 like the K-quants; the 64 value-word rows make the weight tile
// ~33KB so one block is resident per CU either way, but the 2-block launch
// bound is kept: its 128-register cap is what keeps the vec_dot from
// spilling to scratch (measured 26x slower without it on gfx906).
#define  MMQ_X_IQ4_XS 64
#define  MMQ_Y_IQ4_XS 128
#define NWARPS_IQ4_XS 8
#else
#define  MMQ_X_IQ4_XS 4
#define  MMQ_Y_IQ4_XS 32
#define NWARPS_IQ4_XS 4
#endif

template<typename scalar_t, bool need_check> static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_IQ4_XS, 2)
#endif
mul_mat_iq4_xs(
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_IQ4_XS;
    const int mmq_y  =  MMQ_Y_IQ4_XS;
    const int nwarps = NWARPS_IQ4_XS;

    mul_mat_q<scalar_t, QK_K, QR4_XS, QI4_XS, false, block_iq4_xs, mmq_x, mmq_y, nwarps, allocate_tiles_iq4_xs<mmq_y>,
        load_tiles_iq4_xs<mmq_y, nwarps, need_check>, VDR_IQ4_XS_Q8_1_MMQ, vec_dot_iq4_xs_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template<typename scalar_t>
static void ggml_mul_mat_iq4_xs_q8_1_cuda(
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {
    const int mmq_x  =  MMQ_X_IQ4_XS;
    const int mmq_y  =  MMQ_Y_IQ4_XS;
    const int nwarps = NWARPS_IQ4_XS;

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        mul_mat_iq4_xs<scalar_t, false><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_iq4_xs<scalar_t, true><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}


// gfx906 i-quant family MMQ: same tile geometry as iq4_xs (mmq_x=64,
// mmq_y=128, 8 warps, 2-block launch bound for the 128-register cap). The
// x_ql/x_sc strides differ per type (value words vs packed iq1_m words,
// 4/8/16/64 scale slots) and are baked into each allocator instantiation.

#define  MMQ_X_IQ 64
#define  MMQ_Y_IQ 128
#define NWARPS_IQ 8

#define DEFINE_MUL_MAT_IQ(NAME, BLOCK_T, LOADER, VECDOT, QLSTR, SCSTR, NEED_SUM)        \
template<typename scalar_t, bool need_check> static __global__ void                     \
__launch_bounds__(WARP_SIZE_GGUF*NWARPS_IQ, 2)                                          \
NAME(                                                                                    \
    const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) { \
    mul_mat_q<scalar_t, QK_K, QR4_XS, QI4_XS, NEED_SUM, BLOCK_T, MMQ_X_IQ, MMQ_Y_IQ, NWARPS_IQ, \
        allocate_tiles_iq<MMQ_Y_IQ, QLSTR, SCSTR>,                                       \
        LOADER<MMQ_Y_IQ, NWARPS_IQ, need_check>, VDR_IQ_Q8_1_MMQ, VECDOT>                \
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);                    \
}

DEFINE_MUL_MAT_IQ(mul_mat_iq4_nl,  block_iq4_nl_x8, load_tiles_iq4_nl,  vec_dot_iq_v4_mul_mat,  2*WARP_SIZE_GGUF + 1, 8, false)
DEFINE_MUL_MAT_IQ(mul_mat_iq2_xxs, block_iq2_xxs,   load_tiles_iq2_xxs, vec_dot_iq_v4_mul_mat,  2*WARP_SIZE_GGUF + 1, 8, false)
DEFINE_MUL_MAT_IQ(mul_mat_iq2_xs,  block_iq2_xs,    load_tiles_iq2_xs,  vec_dot_iq_v8_mul_mat,  2*WARP_SIZE_GGUF + 1, 16, false)
DEFINE_MUL_MAT_IQ(mul_mat_iq2_s,   block_iq2_s,     load_tiles_iq2_s,   vec_dot_iq_v8_mul_mat,  2*WARP_SIZE_GGUF + 1, 16, false)
DEFINE_MUL_MAT_IQ(mul_mat_iq3_xxs, block_iq3_xxs,   load_tiles_iq3_xxs, vec_dot_iq_v4_mul_mat,  2*WARP_SIZE_GGUF + 1, 8, false)
DEFINE_MUL_MAT_IQ(mul_mat_iq3_s,   block_iq3_s,     load_tiles_iq3_s,   vec_dot_iq_v4_mul_mat,  2*WARP_SIZE_GGUF + 1, 8, false)
DEFINE_MUL_MAT_IQ(mul_mat_iq1_s,   block_iq1_s,     load_tiles_iq1_s,   vec_dot_iq1_s_mul_mat,  2*WARP_SIZE_GGUF + 1, 16, true)
DEFINE_MUL_MAT_IQ(mul_mat_iq1_m,   block_iq1_m,     load_tiles_iq1_m,   vec_dot_iq1_m_mul_mat,  WARP_SIZE_GGUF + 1,   64, false)

#define DEFINE_GGML_MUL_MAT_IQ_CUDA(NAME, KERNEL)                                       \
template<typename scalar_t>                                                             \
static void NAME(                                                                       \
    const void * vx, const void * vy, scalar_t * dst, const int ncols_x, const int nrows_x, \
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {   \
    const int block_num_x = (nrows_x + MMQ_Y_IQ - 1) / MMQ_Y_IQ;                        \
    const int block_num_y = (ncols_y + MMQ_X_IQ - 1) / MMQ_X_IQ;                        \
    const dim3 block_nums(block_num_x, block_num_y, 1);                                 \
    const dim3 block_dims(WARP_SIZE_GGUF, NWARPS_IQ, 1);                                \
    if (nrows_x % MMQ_Y_IQ == 0) {                                                      \
        KERNEL<scalar_t, false><<<block_nums, block_dims, 0, stream>>>                  \
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);               \
    } else {                                                                            \
        KERNEL<scalar_t, true><<<block_nums, block_dims, 0, stream>>>                   \
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);               \
    }                                                                                   \
}

DEFINE_GGML_MUL_MAT_IQ_CUDA(ggml_mul_mat_iq4_nl_q8_1_cuda,  mul_mat_iq4_nl)
DEFINE_GGML_MUL_MAT_IQ_CUDA(ggml_mul_mat_iq2_xxs_q8_1_cuda, mul_mat_iq2_xxs)
DEFINE_GGML_MUL_MAT_IQ_CUDA(ggml_mul_mat_iq2_xs_q8_1_cuda,  mul_mat_iq2_xs)
DEFINE_GGML_MUL_MAT_IQ_CUDA(ggml_mul_mat_iq2_s_q8_1_cuda,   mul_mat_iq2_s)
DEFINE_GGML_MUL_MAT_IQ_CUDA(ggml_mul_mat_iq3_xxs_q8_1_cuda, mul_mat_iq3_xxs)
DEFINE_GGML_MUL_MAT_IQ_CUDA(ggml_mul_mat_iq3_s_q8_1_cuda,   mul_mat_iq3_s)
DEFINE_GGML_MUL_MAT_IQ_CUDA(ggml_mul_mat_iq1_s_q8_1_cuda,   mul_mat_iq1_s)
DEFINE_GGML_MUL_MAT_IQ_CUDA(ggml_mul_mat_iq1_m_q8_1_cuda,   mul_mat_iq1_m)
