#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <string>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../cuda_compat.h"
#include "dispatch_utils.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"

// Q8 gemv
template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static __global__ void silu_and_mul_quantize_q8_1(
    const scalar_t* __restrict__ x, void* __restrict__ vy, const int kx,
    const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  float xi = 0.0f;
  if (ix < kx) {
    const scalar_t gate_value = x[iy * 2 * kx + ix];
    const scalar_t up_value = x[iy * 2 * kx + kx + ix];
    const float gate = static_cast<float>(gate_value);
    const scalar_t silu_gate =
        static_cast<scalar_t>(gate / (1.0f + expf(-gate)));
    const scalar_t activated = silu_gate * up_value;
    xi = static_cast<float>(activated);
  }

  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static __global__ void sigmoid_and_mul_quantize_q8_1(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ gate,
    void* __restrict__ vy, const int kx, const int kx_padded,
    const int64_t x_stride_b, const int64_t x_stride_d,
    const int64_t gate_stride_b, const int64_t gate_stride_d) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  float xi = 0.0f;
  if (ix < kx) {
    const float x_value =
        static_cast<float>(x[iy * x_stride_b + ix * x_stride_d]);
    const float gate_value =
        static_cast<float>(gate[iy * gate_stride_b + ix * gate_stride_d]);
    const scalar_t sigmoid_gate =
        static_cast<scalar_t>(1.0f / (1.0f + expf(-gate_value)));
    const scalar_t activated =
        static_cast<scalar_t>(x_value) * sigmoid_gate;
    xi = static_cast<float>(activated);
  }

  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static __global__ void rms_norm_gated_quantize_q8_1(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ gate, void* __restrict__ vy,
    const float epsilon, const int heads, const int head_dim,
    const int hidden_size_padded, const int64_t x_stride_t,
    const int64_t x_stride_h, const int64_t x_stride_d,
    const int64_t gate_stride_t, const int64_t gate_stride_h,
    const int64_t gate_stride_d, const bool norm_before_gate) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;

  float variance = 0.0f;
  for (int idx = threadIdx.x; idx < head_dim; idx += blockDim.x) {
    float value = static_cast<float>(
        x[token * x_stride_t + head * x_stride_h + idx * x_stride_d]);
    if (!norm_before_gate) {
      const float gate_value = static_cast<float>(
          gate[token * gate_stride_t + head * gate_stride_h +
               idx * gate_stride_d]);
      value *= gate_value / (1.0f + expf(-gate_value));
    }
    variance += value * value;
  }

  __shared__ float warp_sums[32];
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    variance += VLLM_SHFL_XOR_SYNC(variance, mask);
  }
  if (lane == 0) {
    warp_sums[warp] = variance;
  }
  __syncthreads();

  float total_variance = 0.0f;
  if (warp == 0) {
    total_variance = lane < num_warps ? warp_sums[lane] : 0.0f;
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      total_variance += VLLM_SHFL_XOR_SYNC(total_variance, mask);
    }
    if (lane == 0) {
      warp_sums[0] = rsqrtf(total_variance / head_dim + epsilon);
    }
  }
  __syncthreads();
  const float inv_rms = warp_sums[0];

  block_q8_1* y = (block_q8_1*)vy;
  const int qblocks_per_head = head_dim / QK8_1;
  const int qblocks_per_row = hidden_size_padded / QK8_1;
  const int row_qblock_base = token * qblocks_per_row + head * qblocks_per_head;
  const int quant_lane = threadIdx.x % QK8_1;
  const int quant_group = threadIdx.x / QK8_1;
  const int quant_groups = blockDim.x / QK8_1;

  for (int qblock = quant_group; qblock < qblocks_per_head;
       qblock += quant_groups) {
    const int idx = qblock * QK8_1 + quant_lane;
    float xi = 0.0f;
    if (idx < head_dim) {
      const float value = static_cast<float>(
          x[token * x_stride_t + head * x_stride_h + idx * x_stride_d]);
      const float gate_value = static_cast<float>(
          gate[token * gate_stride_t + head * gate_stride_h +
               idx * gate_stride_d]);
      const float silu_gate = gate_value / (1.0f + expf(-gate_value));
      const float normed =
          norm_before_gate ? value * inv_rms * silu_gate
                           : value * silu_gate * inv_rms;
      const scalar_t rounded =
          static_cast<scalar_t>(normed * static_cast<float>(weight[idx]));
      xi = static_cast<float>(rounded);
    }

    float amax = fabsf(xi);
    float sum = xi;
#pragma unroll
    for (int mask = QK8_1 / 2; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, QK8_1));
      sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, QK8_1);
    }

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    block_q8_1& out_block = y[row_qblock_base + qblock];
    out_block.qs[quant_lane] = q;
    if (quant_lane == 0) {
      out_block.ds.x = __float2half(d);
      out_block.ds.y = __float2half(sum);
    }
  }
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

template <typename scalar_t>
static void silu_and_mul_quantize_row_q8_1_cuda(const scalar_t* x, void* vy,
                                                const int kx, const int ky,
                                                cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    silu_and_mul_quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * 2 * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx,
        kx_padded);
  }
}

template <typename scalar_t>
static void sigmoid_and_mul_quantize_row_q8_1_cuda(
    const scalar_t* x, const scalar_t* gate, void* vy, const int kx,
    const int ky, const int64_t x_stride_b, const int64_t x_stride_d,
    const int64_t gate_stride_b, const int64_t gate_stride_d,
    cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    sigmoid_and_mul_quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        x + off * x_stride_b, gate + off * gate_stride_b,
        (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded,
        x_stride_b, x_stride_d, gate_stride_b, gate_stride_d);
  }
}

torch::Tensor ggml_quantize_row_q8_1(torch::Tensor X);
void ggml_quantize_row_q8_1_out(torch::Tensor X, torch::Tensor quant_X);
torch::Tensor ggml_silu_and_mul_quantize_row_q8_1(torch::Tensor X);
void ggml_silu_and_mul_quantize_row_q8_1_out(torch::Tensor X,
                                             torch::Tensor quant_X);
torch::Tensor ggml_sigmoid_and_mul_quantize_row_q8_1(torch::Tensor X,
                                                     torch::Tensor gate);
void ggml_sigmoid_and_mul_quantize_row_q8_1_out(torch::Tensor X,
                                                torch::Tensor gate,
                                                torch::Tensor quant_X);
torch::Tensor ggml_rms_norm_gated_quantize_row_q8_1(
    torch::Tensor X, torch::Tensor weight, torch::Tensor gate, double epsilon,
    bool norm_before_gate);
void ggml_rms_norm_gated_quantize_row_q8_1_out(
    torch::Tensor X, torch::Tensor weight, torch::Tensor gate,
    torch::Tensor quant_X, double epsilon, bool norm_before_gate);
torch::Tensor ggml_mul_mat_vec_q8(torch::Tensor W, torch::Tensor quant_X,
                                  int64_t type, int64_t row, int64_t col,
                                  at::ScalarType dtype);

static __device__ __forceinline__ int8_t gguf_i8_from_i32(const int packed,
                                                          const int index) {
  return static_cast<int8_t>((packed >> (index * 8)) & 0xff);
}

template <typename scalar_t, int THREADS>
static __global__ void q8_0_q8_1_decode_mmvq_kernel(
    const block_q8_0* __restrict__ w, const block_q8_1* __restrict__ x,
    scalar_t* __restrict__ y, const int rows, const int cols,
    const int vecs, const int dst_stride, const int x_blocks_per_vec) {
  const int row = blockIdx.x;
  const int vec = blockIdx.y;
  const int tid = threadIdx.x;
  const int w_blocks_per_row = cols / QK8_0;
  float partial = 0.0f;

  for (int block = tid; block < w_blocks_per_row; block += THREADS) {
    const block_q8_0& wb = w[row * w_blocks_per_row + block];
    const block_q8_1& xb = x[vec * x_blocks_per_vec + block];
    const int* wq = reinterpret_cast<const int*>(wb.qs);
    const int* xq = reinterpret_cast<const int*>(xb.qs);
    int isum = 0;
#pragma unroll
    for (int word = 0; word < QK8_0 / 4; ++word) {
      const int wi = wq[word];
      const int xi = xq[word];
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        isum += static_cast<int>(gguf_i8_from_i32(wi, byte)) *
                static_cast<int>(gguf_i8_from_i32(xi, byte));
      }
    }
    partial += static_cast<float>(isum) * __half2float(wb.d) *
               __low2float(xb.ds);
  }

  __shared__ float reductions[THREADS];
  reductions[tid] = partial;
  __syncthreads();

  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] += reductions[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    y[vec * dst_stride + row] = static_cast<scalar_t>(reductions[0]);
  }
}

template <typename scalar_t>
static void q8_0_q8_1_decode_mmvq_cuda(const void* W, const void* quant_X,
                                       scalar_t* dst, const int cols,
                                       const int rows, const int vecs,
                                       cudaStream_t stream,
                                       const int dst_stride) {
  constexpr int threads = 64;
  const int x_cols_padded = (cols + 512 - 1) / 512 * 512;
  const int x_blocks_per_vec = x_cols_padded / QK8_1;
  const dim3 grid(rows, vecs, 1);
  const dim3 block(threads, 1, 1);
  q8_0_q8_1_decode_mmvq_kernel<scalar_t, threads><<<grid, block, 0, stream>>>(
      static_cast<const block_q8_0*>(W), static_cast<const block_q8_1*>(quant_X),
      dst, rows, cols, vecs, dst_stride, x_blocks_per_vec);
}

torch::Tensor ggml_dequantize(torch::Tensor W,  // quant weight
                              int64_t type, int64_t m, int64_t n,
                              std::optional<at::ScalarType> const& dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  DW.zero_();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

static __global__ void repack_iq4_xs_to_q8_0_kernel(
    const block_iq4_xs* __restrict__ src, block_q8_0* __restrict__ dst,
    const int rows, const int blocks_per_row) {
  const int row = blockIdx.x;
  const int q8_block = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || q8_block >= blocks_per_row * 8 || lane >= QK8_0) {
    return;
  }

  const int super_block = q8_block / 8;
  const int group = q8_block % 8;
  const block_iq4_xs& in = src[row * blocks_per_row + super_block];
  block_q8_0& out = dst[row * blocks_per_row * 8 + q8_block];

  const int q_index = lane < 16 ? lane : lane - 16;
  const uint8_t packed = in.qs[group * 16 + q_index];
  const uint8_t nibble = lane < 16 ? (packed & 0x0f) : (packed >> 4);
  out.qs[lane] = kvalues_iq4nl[nibble];

  if (lane == 0) {
    const int scale =
        ((in.scales_l[group / 2] >> (4 * (group % 2))) & 0x0f) |
        (((in.scales_h >> (2 * group)) & 0x03) << 4);
    const float d = __half2float(in.d) * static_cast<float>(scale - 32);
    out.d = __float2half(d);
  }
}

torch::Tensor ggml_repack_iq4_xs_to_q8_0(torch::Tensor W, int64_t row,
                                         int64_t col) {
  TORCH_CHECK(W.is_cuda(), "IQ4_XS repack input must be a CUDA tensor");
  TORCH_CHECK(W.is_contiguous(), "IQ4_XS repack input must be contiguous");
  TORCH_CHECK(col % QK_K == 0, "IQ4_XS repack col must be divisible by QK_K");
  TORCH_CHECK(W.size(0) == row, "IQ4_XS repack row mismatch");

  const int blocks_per_row = col / QK_K;
  const int q8_blocks_per_row = col / QK8_0;
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(W.device());
  at::Tensor out = torch::empty(
      {row, q8_blocks_per_row * static_cast<int>(sizeof(block_q8_0))},
      options);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(row, q8_blocks_per_row, 1);
  const dim3 block(QK8_0, 1, 1);
  repack_iq4_xs_to_q8_0_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_iq4_xs*>(W.data_ptr()),
      static_cast<block_q8_0*>(out.data_ptr()), row, blocks_per_row);
  return out;
}

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W,  // quant weight
                                  torch::Tensor X,  // input
                                  int64_t type, int64_t row) {
  int64_t col = X.sizes()[1];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  at::Tensor quant_X = ggml_quantize_row_q8_1(X);
  return ggml_mul_mat_vec_q8(W, quant_X, type, row, col, X.scalar_type());
}

torch::Tensor ggml_quantize_row_q8_1(torch::Tensor X) {
  int64_t col = X.sizes()[1];
  int64_t vecs = X.sizes()[0];
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(X.device());
  at::Tensor quant_X = torch::empty({vecs, padded / 32 * 9}, options);
  ggml_quantize_row_q8_1_out(X, quant_X);
  return quant_X;
}

void ggml_quantize_row_q8_1_out(torch::Tensor X, torch::Tensor quant_X) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(quant_X.is_cuda(), "quant_X must be a CUDA tensor");
  TORCH_CHECK(X.dim() == 2, "X must have shape [rows, cols]");
  TORCH_CHECK(quant_X.scalar_type() == torch::kInt32,
              "quant_X must have dtype int32");
  TORCH_CHECK(quant_X.device() == X.device(),
              "quant_X must be on the same device as X");
  int col = X.sizes()[1];
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  TORCH_CHECK(quant_X.dim() == 2 && quant_X.size(0) == vecs &&
                  quant_X.size(1) == padded / 32 * 9,
              "quant_X has incorrect shape");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_quantize_row_q8_1", [&] {
    quantize_row_q8_1_cuda<scalar_t>(
        (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, vecs, stream);
  });
}

torch::Tensor ggml_silu_and_mul_quantize_row_q8_1(torch::Tensor X) {
  int col = X.sizes()[1] / 2;
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(X.device());
  at::Tensor quant_X = torch::empty({vecs, padded / 32 * 9}, options);
  ggml_silu_and_mul_quantize_row_q8_1_out(X, quant_X);
  return quant_X;
}

void ggml_silu_and_mul_quantize_row_q8_1_out(torch::Tensor X,
                                             torch::Tensor quant_X) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(quant_X.is_cuda(), "quant_X must be a CUDA tensor");
  TORCH_CHECK(X.dim() == 2, "X must have shape [rows, 2 * cols]");
  TORCH_CHECK(X.sizes()[1] % 2 == 0, "X last dimension must be even");
  TORCH_CHECK(quant_X.scalar_type() == torch::kInt32,
              "quant_X must have dtype int32");
  TORCH_CHECK(quant_X.device() == X.device(),
              "quant_X must be on the same device as X");
  int col = X.sizes()[1] / 2;
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  TORCH_CHECK(quant_X.dim() == 2 && quant_X.size(0) == vecs &&
                  quant_X.size(1) == padded / 32 * 9,
              "quant_X has incorrect shape");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  VLLM_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_silu_and_mul_quantize_row_q8_1", [&] {
        silu_and_mul_quantize_row_q8_1_cuda<scalar_t>(
            (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, vecs,
            stream);
      });
}

torch::Tensor ggml_sigmoid_and_mul_quantize_row_q8_1(torch::Tensor X,
                                                     torch::Tensor gate) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
  TORCH_CHECK(X.dim() == 2, "X must have shape [rows, cols]");
  TORCH_CHECK(gate.sizes() == X.sizes(), "gate shape must match X");
  const int col = X.sizes()[1];
  const int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(X.device());
  at::Tensor quant_X = torch::empty({vecs, padded / 32 * 9}, options);
  ggml_sigmoid_and_mul_quantize_row_q8_1_out(X, gate, quant_X);
  return quant_X;
}

void ggml_sigmoid_and_mul_quantize_row_q8_1_out(torch::Tensor X,
                                                torch::Tensor gate,
                                                torch::Tensor quant_X) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
  TORCH_CHECK(quant_X.is_cuda(), "quant_X must be a CUDA tensor");
  TORCH_CHECK(X.dim() == 2, "X must have shape [rows, cols]");
  TORCH_CHECK(gate.sizes() == X.sizes(), "gate shape must match X");
  TORCH_CHECK(X.scalar_type() == gate.scalar_type(),
              "X and gate must have the same dtype");
  TORCH_CHECK(quant_X.scalar_type() == torch::kInt32,
              "quant_X must have dtype int32");
  TORCH_CHECK(quant_X.device() == X.device(),
              "quant_X must be on the same device as X");
  TORCH_CHECK(gate.device() == X.device(),
              "gate must be on the same device as X");
  TORCH_CHECK(X.stride(-1) == 1, "X last dimension must be contiguous");
  TORCH_CHECK(gate.stride(-1) == 1, "gate last dimension must be contiguous");
  const int col = X.sizes()[1];
  const int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  TORCH_CHECK(quant_X.dim() == 2 && quant_X.size(0) == vecs &&
                  quant_X.size(1) == padded / 32 * 9,
              "quant_X has incorrect shape");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  VLLM_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_sigmoid_and_mul_quantize_row_q8_1", [&] {
        sigmoid_and_mul_quantize_row_q8_1_cuda<scalar_t>(
            (scalar_t*)X.data_ptr(), (scalar_t*)gate.data_ptr(),
            (void*)quant_X.data_ptr(), col, vecs, X.stride(0), X.stride(1),
            gate.stride(0), gate.stride(1), stream);
      });
}

torch::Tensor ggml_rms_norm_gated_quantize_row_q8_1(
    torch::Tensor X, torch::Tensor weight, torch::Tensor gate, double epsilon,
    bool norm_before_gate) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
  TORCH_CHECK(X.dim() == 3, "X must have shape [tokens, heads, head_dim]");
  TORCH_CHECK(gate.sizes() == X.sizes(), "gate shape must match X");
  TORCH_CHECK(X.scalar_type() == gate.scalar_type() &&
                  X.scalar_type() == weight.scalar_type(),
              "X, gate, and weight must have the same dtype");
  TORCH_CHECK(X.stride(-1) == 1, "X last dimension must be contiguous");
  TORCH_CHECK(gate.stride(-1) == 1, "gate last dimension must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int tokens = X.sizes()[0];
  const int heads = X.sizes()[1];
  const int head_dim = X.sizes()[2];
  TORCH_CHECK(weight.numel() == head_dim,
              "weight size must match X head_dim");
  TORCH_CHECK(head_dim % QK8_1 == 0,
              "head_dim must be divisible by Q8_1 block size");
  const int hidden_size = heads * head_dim;
  const int padded = (hidden_size + 512 - 1) / 512 * 512;
  TORCH_CHECK(padded == hidden_size,
              "flattened hidden size must already be 512-aligned");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(X.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  ggml_rms_norm_gated_quantize_row_q8_1_out(
      X, weight, gate, quant_X, epsilon, norm_before_gate);
  return quant_X;
}

void ggml_rms_norm_gated_quantize_row_q8_1_out(
    torch::Tensor X, torch::Tensor weight, torch::Tensor gate,
    torch::Tensor quant_X, double epsilon, bool norm_before_gate) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
  TORCH_CHECK(quant_X.is_cuda(), "quant_X must be a CUDA tensor");
  TORCH_CHECK(X.dim() == 3, "X must have shape [tokens, heads, head_dim]");
  TORCH_CHECK(gate.sizes() == X.sizes(), "gate shape must match X");
  TORCH_CHECK(X.scalar_type() == gate.scalar_type() &&
                  X.scalar_type() == weight.scalar_type(),
              "X, gate, and weight must have the same dtype");
  TORCH_CHECK(quant_X.scalar_type() == torch::kInt32,
              "quant_X must have dtype int32");
  TORCH_CHECK(quant_X.device() == X.device(),
              "quant_X must be on the same device as X");
  TORCH_CHECK(X.stride(-1) == 1, "X last dimension must be contiguous");
  TORCH_CHECK(gate.stride(-1) == 1, "gate last dimension must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int tokens = X.sizes()[0];
  const int heads = X.sizes()[1];
  const int head_dim = X.sizes()[2];
  TORCH_CHECK(weight.numel() == head_dim,
              "weight size must match X head_dim");
  TORCH_CHECK(head_dim % QK8_1 == 0,
              "head_dim must be divisible by Q8_1 block size");
  const int hidden_size = heads * head_dim;
  const int padded = (hidden_size + 512 - 1) / 512 * 512;
  TORCH_CHECK(padded == hidden_size,
              "flattened hidden size must already be 512-aligned");
  TORCH_CHECK(quant_X.dim() == 2 && quant_X.size(0) == tokens &&
                  quant_X.size(1) == padded / 32 * 9,
              "quant_X has incorrect shape");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int threads = 256;
  const dim3 grid(tokens, heads, 1);
  const dim3 block(threads, 1, 1);
  VLLM_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_rms_norm_gated_quantize_row_q8_1", [&] {
        rms_norm_gated_quantize_q8_1<scalar_t>
            <<<grid, block, 0, stream>>>(
                (scalar_t*)X.data_ptr(), (scalar_t*)weight.data_ptr(),
                (scalar_t*)gate.data_ptr(), (void*)quant_X.data_ptr(),
                static_cast<float>(epsilon), heads, head_dim, padded,
                X.stride(0), X.stride(1), X.stride(2), gate.stride(0),
                gate.stride(1), gate.stride(2), norm_before_gate);
      });
}

static bool q4_k_fixed_cols_fast_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("VLLM_GGUF_Q4_K_FIXED_COLS_FAST");
    if (env != nullptr) {
      return env[0] != '0';
    }
#if defined(USE_ROCM)
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    return std::string(properties->gcnArchName).find("gfx906") == 0;
#else
    return false;
#endif
  }();
  return enabled;
}

static bool q5_k_fixed_cols_fast_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("VLLM_GGUF_Q5_K_FIXED_COLS_FAST");
    if (env != nullptr) {
      return env[0] != '0';
    }
#if defined(USE_ROCM)
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    return std::string(properties->gcnArchName).find("gfx906") == 0;
#else
    return false;
#endif
  }();
  return enabled;
}

static bool q6_k_fixed_cols_fast_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("VLLM_GGUF_Q6_K_FIXED_COLS_FAST");
    if (env != nullptr) {
      return env[0] != '0';
    }
#if defined(USE_ROCM)
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    return std::string(properties->gcnArchName).find("gfx906") == 0;
#else
    return false;
#endif
  }();
  return enabled;
}

static int q6_k_fixed_cols_min_rows() {
  static const int min_rows = [] {
    const char* env = std::getenv("VLLM_GGUF_Q6_K_FIXED_COLS_MIN_ROWS");
    return env == nullptr ? 65536 : std::max(1, std::atoi(env));
  }();
  return min_rows;
}

static int q8_0_row_tile() {
  static const int rows = [] {
    const char* env = std::getenv("VLLM_GGUF_Q8_0_ROW_TILE");
    if (env != nullptr) {
      return std::atoi(env);
    }
#if defined(USE_ROCM)
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    return std::string(properties->gcnArchName).find("gfx906") == 0 ? 4 : 0;
#else
    return 0;
#endif
  }();
  return rows;
}

static int q8_0_row_tile_min_rows() {
  static const int min_rows = [] {
    const char* env = std::getenv("VLLM_GGUF_Q8_0_ROW_TILE_MIN_ROWS");
    return env == nullptr ? 2048 : std::max(1, std::atoi(env));
  }();
  return min_rows;
}

template <typename scalar_t>
static void ggml_mul_mat_vec_q8_dispatch(
    const void* W, const void* quant_X, scalar_t* dst, int col, int row,
    int vecs, int64_t type, cudaStream_t stream, int dst_stride) {
  switch (type) {
    case 2:
      mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 3:
      mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 6:
      mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 7:
      mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 8:
      if (vecs >= 1 && vecs <= 4 && row >= q8_0_row_tile_min_rows() &&
          q8_0_row_tile() == 2) {
        mul_mat_vec_q8_0_q8_1_row_tile_cuda<scalar_t, 2>(
            W, quant_X, dst, col, row, vecs, stream, dst_stride);
      } else if (vecs >= 1 && vecs <= 4 && row >= q8_0_row_tile_min_rows() &&
                 q8_0_row_tile() == 4) {
        mul_mat_vec_q8_0_q8_1_row_tile_cuda<scalar_t, 4>(
            W, quant_X, dst, col, row, vecs, stream, dst_stride);
      } else {
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            W, quant_X, dst, col, row, vecs, stream, dst_stride);
      }
      break;
    case 10:
      mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 11:
      mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 12: {
      const bool q4_k_fixed_cols_fast = q4_k_fixed_cols_fast_enabled();
      static const bool q4_k_col2560_fast = [] {
        const char* env = std::getenv("VLLM_GGUF_Q4_K_COL2560_FAST");
        return env == nullptr || env[0] != '0';
      }();
      if (q4_k_fixed_cols_fast && vecs >= 1 && vecs <= 4 && col != 2560) {
        switch (col) {
          case 2048:
            mul_mat_vec_q4_K_q8_1_fixed_cols_cuda<scalar_t, 2048>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          case 4096:
            mul_mat_vec_q4_K_q8_1_fixed_cols_cuda<scalar_t, 4096>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          case 5120:
            mul_mat_vec_q4_K_q8_1_fixed_cols_cuda<scalar_t, 5120>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          case 6144:
            mul_mat_vec_q4_K_q8_1_fixed_cols_cuda<scalar_t, 6144>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          case 9216:
            mul_mat_vec_q4_K_q8_1_fixed_cols_cuda<scalar_t, 9216>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          case 17408:
            mul_mat_vec_q4_K_q8_1_fixed_cols_cuda<scalar_t, 17408>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          default:
            mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
                W, quant_X, dst, col, row, vecs, stream, dst_stride);
            break;
        }
      } else if (q4_k_col2560_fast && col == 2560) {
        mul_mat_vec_q4_K_q8_1_col2560_cuda<scalar_t>(
            W, quant_X, dst, row, vecs, stream, dst_stride);
      } else {
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            W, quant_X, dst, col, row, vecs, stream, dst_stride);
      }
      break;
    }
    case 13: {
      const bool q5_k_fixed_cols_fast = q5_k_fixed_cols_fast_enabled();
      static const bool q5_k_5120_6144_fast = [] {
        const char* env = std::getenv("VLLM_GGUF_Q5_K_5120_6144_FAST");
        return env == nullptr || env[0] != '0';
      }();
      static const bool q5_k_col2560_fast = [] {
        const char* env = std::getenv("VLLM_GGUF_Q5_K_COL2560_FAST");
        return env == nullptr || env[0] != '0';
      }();
      if (q5_k_fixed_cols_fast && vecs >= 1 && vecs <= 4 && col != 2560) {
        switch (col) {
          case 5120:
            if (q5_k_5120_6144_fast) {
              mul_mat_vec_q5_K_q8_1_fixed_cols_cuda<scalar_t, 5120>(
                  W, quant_X, dst, row, vecs, stream, dst_stride);
            } else {
              mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
                  W, quant_X, dst, col, row, vecs, stream, dst_stride);
            }
            break;
          case 6144:
            if (q5_k_5120_6144_fast) {
              mul_mat_vec_q5_K_q8_1_fixed_cols_cuda<scalar_t, 6144>(
                  W, quant_X, dst, row, vecs, stream, dst_stride);
            } else {
              mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
                  W, quant_X, dst, col, row, vecs, stream, dst_stride);
            }
            break;
          case 9216:
            mul_mat_vec_q5_K_q8_1_fixed_cols_cuda<scalar_t, 9216>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          case 17408:
            mul_mat_vec_q5_K_q8_1_fixed_cols_cuda<scalar_t, 17408>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          default:
            mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
                W, quant_X, dst, col, row, vecs, stream, dst_stride);
            break;
        }
      } else if (q5_k_col2560_fast && col == 2560) {
        mul_mat_vec_q5_K_q8_1_col2560_cuda<scalar_t>(
            W, quant_X, dst, row, vecs, stream, dst_stride);
      } else {
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            W, quant_X, dst, col, row, vecs, stream, dst_stride);
      }
      break;
    }
    case 14: {
      static const bool q6_k_col2560_fast = [] {
        const char* env = std::getenv("VLLM_GGUF_Q6_K_COL2560_FAST");
        return env == nullptr || env[0] != '0';
      }();
      if (q6_k_fixed_cols_fast_enabled() && vecs >= 1 && vecs <= 4 &&
          col == 5120 && row >= q6_k_fixed_cols_min_rows()) {
        mul_mat_vec_q6_K_q8_1_fixed_cols_cuda<scalar_t, 5120>(
            W, quant_X, dst, row, vecs, stream, dst_stride);
      } else if (q6_k_fixed_cols_fast_enabled() && vecs >= 2 && vecs <= 4 &&
                 col != 5120) {
        // The prepared-weight fixed-cols kernel re-uses the weight unpack
        // across the combined vectors; the win is multi-vector only (the
        // single-vector code path stays on the generic kernel). col 5120
        // keeps the row gate above: measured on gfx906, small-row 5120
        // matrices run faster on the 2-rows-per-block generic kernel.
        switch (col) {
          case 6144:
            mul_mat_vec_q6_K_q8_1_fixed_cols_cuda<scalar_t, 6144>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          case 9216:
            mul_mat_vec_q6_K_q8_1_fixed_cols_cuda<scalar_t, 9216>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          case 17408:
            mul_mat_vec_q6_K_q8_1_fixed_cols_cuda<scalar_t, 17408>(
                W, quant_X, dst, row, vecs, stream, dst_stride);
            break;
          default:
            mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
                W, quant_X, dst, col, row, vecs, stream, dst_stride);
            break;
        }
      } else if (q6_k_col2560_fast && col == 2560 && row > 65536) {
        mul_mat_vec_q6_K_q8_1_col2560_cuda<scalar_t>(
            W, quant_X, dst, row, vecs, stream, dst_stride);
      } else {
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            W, quant_X, dst, col, row, vecs, stream, dst_stride);
      }
      break;
    }
    case 16:
      mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 17:
      mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 18:
      mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 19:
      mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 20:
      mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 21:
      mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 22:
      mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 23:
      mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
    case 29:
      mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
          W, quant_X, dst, col, row, vecs, stream, dst_stride);
      break;
  }
}

static bool gguf_sharded_group_same_type_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("VLLM_GGUF_SHARDED_GROUP_SAME_TYPE");
    return env == nullptr || env[0] != '0';
  }();
  return enabled;
}

// Weight-prepare ops for grouped K-quant MMVQ. When one thread block combines
// 2-4 activation vectors, the weight-side unpack runs once per weight block
// instead of once per vector (see the MmvqWeight* helpers in vecdotq.cuh).
// Only instantiated for the K-quants; other types keep the per-vector
// vec_dot_q_cuda path.
template <typename block_q_t> struct MmvqKQuantOps;

template <> struct MmvqKQuantOps<block_q4_K> {
  using Weight = MmvqWeightQ4K;
  static __device__ __forceinline__ void prepare(
      const void* vbq, const int& iqs, Weight& w) {
    mmvq_prepare_q4_K(static_cast<const block_q4_K*>(vbq), iqs, w);
  }
  static __device__ __forceinline__ float dot(
      const Weight& w, const block_q8_1* y_block, const int& iqs) {
    return mmvq_dot_q4_K(w, y_block, iqs);
  }
};

template <> struct MmvqKQuantOps<block_q5_K> {
  using Weight = MmvqWeightQ5K;
  static __device__ __forceinline__ void prepare(
      const void* vbq, const int& iqs, Weight& w) {
    mmvq_prepare_q5_K(static_cast<const block_q5_K*>(vbq), iqs, w);
  }
  static __device__ __forceinline__ float dot(
      const Weight& w, const block_q8_1* y_block, const int& iqs) {
    return mmvq_dot_q5_K(w, y_block, iqs);
  }
};

template <> struct MmvqKQuantOps<block_q6_K> {
  using Weight = MmvqWeightQ6K;
  static __device__ __forceinline__ void prepare(
      const void* vbq, const int& iqs, Weight& w) {
    mmvq_prepare_q6_K(static_cast<const block_q6_K*>(vbq), iqs, w);
  }
  static __device__ __forceinline__ float dot(
      const Weight& w, const block_q8_1* y_block, const int& iqs) {
    return mmvq_dot_q6_K(w, y_block, iqs);
  }
};

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, bool KQ_PREPARED = false>
static __global__ void mul_mat_vec_q_grouped_same_type(
    const void* __restrict__ vx0, const void* __restrict__ vx1,
    const void* __restrict__ vx2, const void* __restrict__ vx3,
    const void* __restrict__ vx4, const void* __restrict__ vx5,
    const void* __restrict__ vx6, const void* __restrict__ vx7,
    const void* __restrict__ vy, scalar_t* __restrict__ dst,
    const int ncols, const int total_rows, const int nvecs,
    const int dst_stride, const int n0, const int n1, const int n2,
    const int n3, const int n4, const int n5, const int n6, const int n7) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int vec = blockIdx.y;

  if (row >= total_rows || vec >= nvecs) {
    return;
  }

  int local_row = row;
  const void* vx = vx0;
  int acc = n0;
  if (row >= acc) {
    local_row = row - acc;
    vx = vx1;
    acc += n1;
    if (row >= acc) {
      local_row = row - acc;
      vx = vx2;
      acc += n2;
      if (row >= acc) {
        local_row = row - acc;
        vx = vx3;
        acc += n3;
        if (row >= acc) {
          local_row = row - acc;
          vx = vx4;
          acc += n4;
          if (row >= acc) {
            local_row = row - acc;
            vx = vx5;
            acc += n5;
            if (row >= acc) {
              local_row = row - acc;
              vx = vx6;
              acc += n6;
              if (row >= acc) {
                local_row = row - acc;
                vx = vx7;
              }
            }
          }
        }
      }
    }
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * BLOCK_SIZE / qi;
  const int nrows_y = (ncols + 512 - 1) / 512 * 512;
  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;
  const bool combine_vecs = gguf_mmvq_combine_vecs(nvecs);
  const int vec_count = combine_vecs ? nvecs : 1;
  float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if constexpr (KQ_PREPARED) {
    // Multi-vector instantiation: the weight-side unpack runs once per
    // weight block and is shared across the combined vectors.
    for (int i = threadIdx.x / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
      const int ibx = local_row * blocks_per_row + i;
      const int iqs = vdr * (threadIdx.x % (qi / vdr));
      typename MmvqKQuantOps<block_q_t>::Weight w;
      MmvqKQuantOps<block_q_t>::prepare(&x[ibx], iqs, w);
#pragma unroll
      for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
        if (vec_offset >= vec_count) {
          break;
        }
        const int iby =
            (vec + vec_offset) * (nrows_y / QK8_1) + i * (qk / QK8_1);
        tmp[vec_offset] +=
            MmvqKQuantOps<block_q_t>::dot(w, &y[iby], iqs);
      }
    }
  } else {
    for (int i = threadIdx.x / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
      const int ibx = local_row * blocks_per_row + i;
      const int iqs = vdr * (threadIdx.x % (qi / vdr));
#pragma unroll
      for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
        if (vec_offset >= vec_count) {
          break;
        }
        const int iby =
            (vec + vec_offset) * (nrows_y / QK8_1) + i * (qk / QK8_1);
        tmp[vec_offset] += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
      }
    }
  }

  constexpr int warp_size = WARP_SIZE;
  constexpr int num_warps = BLOCK_SIZE / warp_size;
#pragma unroll
  for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
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
        dst[(vec + vec_offset) * dst_stride + row] = tmp[vec_offset];
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
    for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
      tmp[0] += VLLM_SHFL_XOR_SYNC(tmp[0], mask);
    }
    if (lane == 0) {
      dst[vec * dst_stride + row] = tmp[0];
    }
  }
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, bool KQ_PREPARED = false>
static void mul_mat_vec_q_grouped_same_type_cuda(
    const void* vx0, const void* vx1, const void* vx2, const void* vx3,
    const void* vx4, const void* vx5, const void* vx6, const void* vx7,
    const void* vy, scalar_t* dst, const int ncols, const int total_rows,
    const int nvecs, cudaStream_t stream, const int n0, const int n1,
    const int n2, const int n3, const int n4, const int n5, const int n6,
    const int n7, const int dst_stride = -1) {
  const int block_num_y = (total_rows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, gguf_mmvq_grid_vecs(nvecs), 1);
  const dim3 block_dims(BLOCK_SIZE, GGML_CUDA_MMV_Y, 1);
  if constexpr (KQ_PREPARED) {
    // The prepared variant only pays off for multi-vector batches; select
    // the instantiation at launch time so single-vector batches keep the
    // original codegen.
    if (nvecs >= 2) {
      mul_mat_vec_q_grouped_same_type<scalar_t, qk, qi, block_q_t, vdr,
                                      vec_dot_q_cuda, true>
          <<<block_nums, block_dims, 0, stream>>>(
              vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, vy, dst, ncols,
              total_rows, nvecs, dst_stride > 0 ? dst_stride : total_rows, n0,
              n1, n2, n3, n4, n5, n6, n7);
    } else {
      mul_mat_vec_q_grouped_same_type<scalar_t, qk, qi, block_q_t, vdr,
                                      vec_dot_q_cuda, false>
          <<<block_nums, block_dims, 0, stream>>>(
              vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, vy, dst, ncols,
              total_rows, nvecs, dst_stride > 0 ? dst_stride : total_rows, n0,
              n1, n2, n3, n4, n5, n6, n7);
    }
  } else {
    mul_mat_vec_q_grouped_same_type<scalar_t, qk, qi, block_q_t, vdr,
                                    vec_dot_q_cuda, false>
        <<<block_nums, block_dims, 0, stream>>>(
            vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, vy, dst, ncols, total_rows,
            nvecs, dst_stride > 0 ? dst_stride : total_rows, n0, n1, n2, n3,
            n4, n5, n6, n7);
  }
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, bool KQ_PREPARED = false>
static __global__ void mul_mat_vec_q_grouped_same_type_col2560(
    const void* __restrict__ vx0, const void* __restrict__ vx1,
    const void* __restrict__ vx2, const void* __restrict__ vx3,
    const void* __restrict__ vx4, const void* __restrict__ vx5,
    const void* __restrict__ vx6, const void* __restrict__ vx7,
    const void* __restrict__ vy, scalar_t* __restrict__ dst,
    const int total_rows, const int nvecs, const int dst_stride,
    const int n0, const int n1, const int n2, const int n3, const int n4,
    const int n5, const int n6, const int n7) {
  constexpr int ncols = 2560;
  constexpr int blocks_per_row = ncols / qk;
  constexpr int q8_blocks_per_vec = ncols / QK8_1;
  constexpr int blocks_per_warp = vdr * BLOCK_SIZE / qi;

  const int row = blockIdx.x;
  const int vec = blockIdx.y;
  if (row >= total_rows || vec >= nvecs) {
    return;
  }

  int local_row = row;
  const void* vx = vx0;
  int acc = n0;
  if (row >= acc) {
    local_row = row - acc;
    vx = vx1;
    acc += n1;
    if (row >= acc) {
      local_row = row - acc;
      vx = vx2;
      acc += n2;
      if (row >= acc) {
        local_row = row - acc;
        vx = vx3;
        acc += n3;
        if (row >= acc) {
          local_row = row - acc;
          vx = vx4;
          acc += n4;
          if (row >= acc) {
            local_row = row - acc;
            vx = vx5;
            acc += n5;
            if (row >= acc) {
              local_row = row - acc;
              vx = vx6;
              acc += n6;
              if (row >= acc) {
                local_row = row - acc;
                vx = vx7;
              }
            }
          }
        }
      }
    }
  }

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;
  const bool combine_vecs = gguf_mmvq_combine_vecs(nvecs);
  const int vec_count = combine_vecs ? nvecs : 1;
  float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if constexpr (KQ_PREPARED) {
    // Multi-vector instantiation: the weight-side unpack runs once per
    // weight block and is shared across the combined vectors.
    for (int i = threadIdx.x / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
      const int ibx = local_row * blocks_per_row + i;
      const int iqs = vdr * (threadIdx.x % (qi / vdr));
      typename MmvqKQuantOps<block_q_t>::Weight w;
      MmvqKQuantOps<block_q_t>::prepare(&x[ibx], iqs, w);
#pragma unroll
      for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
        if (vec_offset >= vec_count) {
          break;
        }
        const int iby = (vec + vec_offset) * q8_blocks_per_vec +
            i * (qk / QK8_1);
        tmp[vec_offset] +=
            MmvqKQuantOps<block_q_t>::dot(w, &y[iby], iqs);
      }
    }
  } else {
    for (int i = threadIdx.x / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
      const int ibx = local_row * blocks_per_row + i;
      const int iqs = vdr * (threadIdx.x % (qi / vdr));
#pragma unroll
      for (int vec_offset = 0; vec_offset < 4; ++vec_offset) {
        if (vec_offset >= vec_count) {
          break;
        }
        const int iby = (vec + vec_offset) * q8_blocks_per_vec +
            i * (qk / QK8_1);
        tmp[vec_offset] += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
      }
    }
  }

  constexpr int warp_size = WARP_SIZE;
  constexpr int num_warps = BLOCK_SIZE / warp_size;
#pragma unroll
  for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
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
        dst[(vec + vec_offset) * dst_stride + row] = tmp[vec_offset];
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
    for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
      tmp[0] += VLLM_SHFL_XOR_SYNC(tmp[0], mask);
    }
    if (lane == 0) {
      dst[vec * dst_stride + row] = tmp[0];
    }
  }
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, bool KQ_PREPARED = false>
static void mul_mat_vec_q_grouped_same_type_col2560_cuda(
    const void* vx0, const void* vx1, const void* vx2, const void* vx3,
    const void* vx4, const void* vx5, const void* vx6, const void* vx7,
    const void* vy, scalar_t* dst, const int total_rows, const int nvecs,
    cudaStream_t stream, const int n0, const int n1, const int n2,
    const int n3, const int n4, const int n5, const int n6, const int n7,
    const int dst_stride = -1) {
  const dim3 block_nums(total_rows, gguf_mmvq_grid_vecs(nvecs), 1);
  const dim3 block_dims(BLOCK_SIZE, 1, 1);
  if constexpr (KQ_PREPARED) {
    if (nvecs >= 2) {
      mul_mat_vec_q_grouped_same_type_col2560<scalar_t, qk, qi, block_q_t,
                                              vdr, vec_dot_q_cuda, true>
          <<<block_nums, block_dims, 0, stream>>>(
              vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, vy, dst, total_rows,
              nvecs, dst_stride > 0 ? dst_stride : total_rows, n0, n1, n2, n3,
              n4, n5, n6, n7);
    } else {
      mul_mat_vec_q_grouped_same_type_col2560<scalar_t, qk, qi, block_q_t,
                                              vdr, vec_dot_q_cuda, false>
          <<<block_nums, block_dims, 0, stream>>>(
              vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, vy, dst, total_rows,
              nvecs, dst_stride > 0 ? dst_stride : total_rows, n0, n1, n2, n3,
              n4, n5, n6, n7);
    }
  } else {
    mul_mat_vec_q_grouped_same_type_col2560<scalar_t, qk, qi, block_q_t, vdr,
                                            vec_dot_q_cuda, false>
        <<<block_nums, block_dims, 0, stream>>>(
            vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, vy, dst, total_rows, nvecs,
            dst_stride > 0 ? dst_stride : total_rows, n0, n1, n2, n3, n4, n5,
            n6, n7);
  }
}

template <typename scalar_t>
static void ggml_mul_mat_vec_q8_grouped_dispatch(
    const void* vx0, const void* vx1, const void* vx2, const void* vx3,
    const void* vx4, const void* vx5, const void* vx6, const void* vx7,
    const void* quant_X, scalar_t* dst, int col, int total_rows, int vecs,
    int64_t type, cudaStream_t stream, const int* rows,
    const int dst_stride = -1) {
  switch (type) {
    case 2:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
          vec_dot_q4_0_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                             dst, col, total_rows, vecs, stream, rows[0],
                             rows[1], rows[2], rows[3], rows[4], rows[5],
                             rows[6], rows[7], dst_stride);
      break;
    case 3:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
          vec_dot_q4_1_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                             dst, col, total_rows, vecs, stream, rows[0],
                             rows[1], rows[2], rows[3], rows[4], rows[5],
                             rows[6], rows[7], dst_stride);
      break;
    case 6:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
          vec_dot_q5_0_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                             dst, col, total_rows, vecs, stream, rows[0],
                             rows[1], rows[2], rows[3], rows[4], rows[5],
                             rows[6], rows[7], dst_stride);
      break;
    case 7:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
          vec_dot_q5_1_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                             dst, col, total_rows, vecs, stream, rows[0],
                             rows[1], rows[2], rows[3], rows[4], rows[5],
                             rows[6], rows[7], dst_stride);
      break;
    case 8:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
          vec_dot_q8_0_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                             dst, col, total_rows, vecs, stream, rows[0],
                             rows[1], rows[2], rows[3], rows[4], rows[5],
                             rows[6], rows[7], dst_stride);
      break;
    case 10:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
          vec_dot_q2_K_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                             dst, col, total_rows, vecs, stream, rows[0],
                             rows[1], rows[2], rows[3], rows[4], rows[5],
                             rows[6], rows[7], dst_stride);
      break;
    case 11:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
          vec_dot_q3_K_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                             dst, col, total_rows, vecs, stream, rows[0],
                             rows[1], rows[2], rows[3], rows[4], rows[5],
                             rows[6], rows[7], dst_stride);
      break;
    case 12:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
          vec_dot_q4_K_q8_1, true>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7,
                                   quant_X, dst, col, total_rows, vecs,
                                   stream, rows[0], rows[1], rows[2], rows[3],
                                   rows[4], rows[5], rows[6], rows[7],
                                   dst_stride);
      break;
    case 13:
      static const bool q5_k_grouped_col2560_fast = [] {
        const char* env = std::getenv("VLLM_GGUF_Q5_K_GROUPED_COL2560_FAST");
        return env == nullptr || env[0] != '0';
      }();
      if (q5_k_grouped_col2560_fast && col == 2560) {
        mul_mat_vec_q_grouped_same_type_col2560_cuda<
            scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
            vec_dot_q5_K_q8_1, true>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7,
                                     quant_X, dst, total_rows, vecs, stream,
                                     rows[0], rows[1], rows[2], rows[3],
                                     rows[4], rows[5], rows[6], rows[7],
                                     dst_stride);
      } else {
        mul_mat_vec_q_grouped_same_type_cuda<
            scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
            vec_dot_q5_K_q8_1, true>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7,
                                     quant_X, dst, col, total_rows, vecs,
                                     stream, rows[0], rows[1], rows[2],
                                     rows[3], rows[4], rows[5], rows[6],
                                     rows[7], dst_stride);
      }
      break;
    case 14:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
          vec_dot_q6_K_q8_1, true>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7,
                                   quant_X, dst, col, total_rows, vecs,
                                   stream, rows[0], rows[1], rows[2], rows[3],
                                   rows[4], rows[5], rows[6], rows[7],
                                   dst_stride);
      break;
    case 16:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1,
          vec_dot_iq2_xxs_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7,
                                quant_X, dst, col, total_rows, vecs, stream,
                                rows[0], rows[1], rows[2], rows[3], rows[4],
                                rows[5], rows[6], rows[7], dst_stride);
      break;
    case 17:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI2_XS, block_iq2_xs, 1,
          vec_dot_iq2_xs_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7,
                               quant_X, dst, col, total_rows, vecs, stream,
                               rows[0], rows[1], rows[2], rows[3], rows[4],
                               rows[5], rows[6], rows[7], dst_stride);
      break;
    case 18:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1,
          vec_dot_iq3_xxs_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7,
                                quant_X, dst, col, total_rows, vecs, stream,
                                rows[0], rows[1], rows[2], rows[3], rows[4],
                                rows[5], rows[6], rows[7], dst_stride);
      break;
    case 19:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI1_S, block_iq1_s, 1,
          vec_dot_iq1_s_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                              dst, col, total_rows, vecs, stream, rows[0],
                              rows[1], rows[2], rows[3], rows[4], rows[5],
                              rows[6], rows[7], dst_stride);
      break;
    case 20:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ,
          vec_dot_iq4_nl_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7,
                               quant_X, dst, col, total_rows, vecs, stream,
                               rows[0], rows[1], rows[2], rows[3], rows[4],
                               rows[5], rows[6], rows[7], dst_stride);
      break;
    case 21:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI3_XS, block_iq3_s, 1,
          vec_dot_iq3_s_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                              dst, col, total_rows, vecs, stream, rows[0],
                              rows[1], rows[2], rows[3], rows[4], rows[5],
                              rows[6], rows[7], dst_stride);
      break;
    case 22:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI2_S, block_iq2_s, 1,
          vec_dot_iq2_s_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                              dst, col, total_rows, vecs, stream, rows[0],
                              rows[1], rows[2], rows[3], rows[4], rows[5],
                              rows[6], rows[7], dst_stride);
      break;
    case 23:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI4_XS, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ,
          vec_dot_iq4_xs_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7,
                               quant_X, dst, col, total_rows, vecs, stream,
                               rows[0], rows[1], rows[2], rows[3], rows[4],
                               rows[5], rows[6], rows[7], dst_stride);
      break;
    case 29:
      mul_mat_vec_q_grouped_same_type_cuda<
          scalar_t, QK_K, QI1_M, block_iq1_m, 1,
          vec_dot_iq1_m_q8_1>(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7, quant_X,
                              dst, col, total_rows, vecs, stream, rows[0],
                              rows[1], rows[2], rows[3], rows[4], rows[5],
                              rows[6], rows[7], dst_stride);
      break;
    default:
      TORCH_CHECK(false, "Unsupported grouped GGUF MMVQ quantization type");
  }
}

template <typename scalar_t, typename block_k_t, int qi_k, int vdr_k,
          vec_dot_q_cuda_t vec_dot_k>
static __global__ void mul_mat_vec_qkv3_q8_kernel(
    const void* __restrict__ vw0, const void* __restrict__ vw1,
    const void* __restrict__ vw2, const void* __restrict__ vx,
    scalar_t* __restrict__ dst, const int col, const int rows0,
    const int rows1, const int rows2, const int vecs,
    const int dst_stride) {
  const int row = blockIdx.x;
  const int vec = blockIdx.y;
  const int total_rows = rows0 + rows1 + rows2;
  if (row >= total_rows || vec >= vecs) {
    return;
  }

  constexpr int qk = QK_K;
  const int blocks_per_row = col / qk;
  const int q8_blocks_per_vec = ((col + 512 - 1) / 512 * 512) / QK8_1;
  const block_q8_1* x = (const block_q8_1*)vx;
  float tmp = 0.0f;

  if (row < rows0 + rows1) {
    const block_k_t* w = (const block_k_t*)(row < rows0 ? vw0 : vw1);
    const int local_row = row < rows0 ? row : row - rows0;
    constexpr int blocks_per_warp = vdr_k * BLOCK_SIZE / qi_k;
    for (int i = threadIdx.x / (qi_k / vdr_k); i < blocks_per_row;
         i += blocks_per_warp) {
      const int ibw = local_row * blocks_per_row + i;
      const int ibx = vec * q8_blocks_per_vec + i * (qk / QK8_1);
      const int iqs = vdr_k * (threadIdx.x % (qi_k / vdr_k));
      tmp += vec_dot_k(&w[ibw], &x[ibx], iqs);
    }
  } else {
    const block_q6_K* w = (const block_q6_K*)vw2;
    const int local_row = row - rows0 - rows1;
    constexpr int blocks_per_warp = VDR_Q6_K_Q8_1_MMVQ * BLOCK_SIZE / QI6_K;
    for (int i = threadIdx.x / QI6_K; i < blocks_per_row;
         i += blocks_per_warp) {
      const int ibw = local_row * blocks_per_row + i;
      const int ibx = vec * q8_blocks_per_vec + i * (qk / QK8_1);
      const int iqs = threadIdx.x % QI6_K;
      tmp += vec_dot_q6_K_q8_1(&w[ibw], &x[ibx], iqs);
    }
  }

  constexpr int warp_size = WARP_SIZE;
  constexpr int num_warps = BLOCK_SIZE / warp_size;
#pragma unroll
  for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
    tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
  }

  if constexpr (num_warps == 1) {
    if (threadIdx.x == 0) {
      dst[vec * dst_stride + row] = tmp;
    }
  } else {
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    __shared__ float shared_sum[num_warps];
    if (lane == 0) {
      shared_sum[warp] = tmp;
    }
    __syncthreads();
    if (warp != 0) {
      return;
    }
    tmp = lane < num_warps ? shared_sum[lane] : 0.0f;
#pragma unroll
    for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
      tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }
    if (lane == 0) {
      dst[vec * dst_stride + row] = tmp;
    }
  }
}

template <typename scalar_t>
static void ggml_mul_mat_vec_q8_qkv3_dispatch(
    const void* W0, const void* W1, const void* W2, const void* quant_X,
    scalar_t* dst, const int col, const int rows0, const int rows1,
    const int rows2, const int vecs, const int64_t type01,
    cudaStream_t stream) {
  const int total_rows = rows0 + rows1 + rows2;
  const dim3 block_nums(total_rows, vecs, 1);
  const dim3 block_dims(BLOCK_SIZE, 1, 1);
  switch (type01) {
    case 12:
      mul_mat_vec_qkv3_q8_kernel<scalar_t, block_q4_K, QI4_K,
                                 VDR_Q4_K_Q8_1_MMVQ,
                                 vec_dot_q4_K_q8_1>
          <<<block_nums, block_dims, 0, stream>>>(
              W0, W1, W2, quant_X, dst, col, rows0, rows1, rows2, vecs,
              total_rows);
      break;
    case 13:
      mul_mat_vec_qkv3_q8_kernel<scalar_t, block_q5_K, QI5_K,
                                 VDR_Q5_K_Q8_1_MMVQ,
                                 vec_dot_q5_K_q8_1>
          <<<block_nums, block_dims, 0, stream>>>(
              W0, W1, W2, quant_X, dst, col, rows0, rows1, rows2, vecs,
              total_rows);
      break;
    default:
      TORCH_CHECK(false, "QKV3 GGUF MMVQ supports only Q4_K/Q5_K + Q6_K");
  }
}

torch::Tensor ggml_mul_mat_vec_q8(torch::Tensor W, torch::Tensor quant_X,
                                  int64_t type, int64_t row, int64_t col,
                                  at::ScalarType dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  int64_t vecs = quant_X.sizes()[0];
  auto options = torch::TensorOptions().dtype(dtype).device(W.device());
  at::Tensor Y = torch::empty({vecs, row}, options);
  Y.zero_();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(dtype, "ggml_mul_mat_vec_q8", [&] {
    ggml_mul_mat_vec_q8_dispatch<scalar_t>(
        (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(),
        col, row, vecs, type, stream, row);
  });
  return Y;
}

void ggml_mul_mat_vec_q8_out(torch::Tensor W, torch::Tensor quant_X,
                             torch::Tensor Y, int64_t type, int64_t row,
                             int64_t col) {
  TORCH_CHECK(W.is_cuda(), "GGUF MMVQ weight must be a CUDA tensor");
  TORCH_CHECK(quant_X.is_cuda(), "GGUF MMVQ input must be a CUDA tensor");
  TORCH_CHECK(Y.is_cuda(), "GGUF MMVQ output must be a CUDA tensor");
  TORCH_CHECK(W.device() == quant_X.device() && W.device() == Y.device(),
              "GGUF MMVQ tensors must be on the same device");
  TORCH_CHECK(quant_X.dim() == 2, "GGUF MMVQ input must have shape [vecs, col]");
  TORCH_CHECK(Y.dim() == 2, "GGUF MMVQ output must have shape [vecs, row]");
  const int vecs = quant_X.sizes()[0];
  TORCH_CHECK(Y.size(0) == vecs && Y.size(1) == row,
              "GGUF MMVQ output has incorrect shape");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(Y.scalar_type(), "ggml_mul_mat_vec_q8_out", [&] {
    ggml_mul_mat_vec_q8_dispatch<scalar_t>(
        (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(),
        col, row, vecs, type, stream, row);
  });
}

torch::Tensor ggml_mul_mat_vec_q8_0_fast(torch::Tensor W, torch::Tensor quant_X,
                                         int64_t row, int64_t col,
                                         at::ScalarType dtype) {
  TORCH_CHECK(W.is_cuda(), "Q8_0 fast GGUF MMVQ weight must be a CUDA tensor");
  TORCH_CHECK(quant_X.is_cuda(),
              "Q8_0 fast GGUF MMVQ input must be a CUDA tensor");
  TORCH_CHECK(W.is_contiguous(),
              "Q8_0 fast GGUF MMVQ requires contiguous weight");
  TORCH_CHECK(quant_X.is_contiguous(),
              "Q8_0 fast GGUF MMVQ requires contiguous quantized input");
  TORCH_CHECK(col % QK8_0 == 0,
              "Q8_0 fast GGUF MMVQ requires col to be divisible by 32");
  TORCH_CHECK(W.sizes()[0] == row,
              "Q8_0 fast GGUF MMVQ row does not match weight rows");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  const int vecs = quant_X.sizes()[0];
  auto options = torch::TensorOptions().dtype(dtype).device(W.device());
  at::Tensor Y = torch::empty({vecs, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(dtype, "ggml_mul_mat_vec_q8_0_fast", [&] {
    q8_0_q8_1_decode_mmvq_cuda<scalar_t>(
        (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
        (scalar_t*)Y.data_ptr(), col, row, vecs, stream, row);
  });
  return Y;
}

torch::Tensor ggml_mul_mat_vec_q8_grouped_same_type(
    std::vector<torch::Tensor> W, torch::Tensor quant_X, int64_t type,
    int64_t col, at::ScalarType dtype) {
  TORCH_CHECK(!W.empty(), "W must have at least one shard");
  TORCH_CHECK(W.size() <= 8, "Grouped GGUF MMVQ supports at most 8 shards");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W[0]));
  int vecs = quant_X.sizes()[0];
  int rows[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const void* ptrs[8] = {nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr};
  int64_t total_rows = 0;
  for (size_t i = 0; i < W.size(); ++i) {
    TORCH_CHECK(W[i].is_cuda(), "All grouped GGUF shards must be CUDA tensors");
    TORCH_CHECK(W[i].device() == W[0].device(),
                "All grouped GGUF shards must be on the same device");
    rows[i] = W[i].sizes()[0];
    ptrs[i] = W[i].data_ptr();
    total_rows += rows[i];
  }

  auto options = torch::TensorOptions().dtype(dtype).device(W[0].device());
  at::Tensor Y = torch::empty({vecs, total_rows}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(
      dtype, "ggml_mul_mat_vec_q8_grouped_same_type", [&] {
        ggml_mul_mat_vec_q8_grouped_dispatch<scalar_t>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6],
            ptrs[7], (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col,
            total_rows, vecs, type, stream, rows);
      });
  return Y;
}

void ggml_mul_mat_vec_q8_grouped_same_type_out(
    std::vector<torch::Tensor> W, torch::Tensor quant_X, torch::Tensor Y,
    int64_t type, int64_t col) {
  TORCH_CHECK(!W.empty(), "W must have at least one shard");
  TORCH_CHECK(W.size() <= 8, "Grouped GGUF MMVQ supports at most 8 shards");
  TORCH_CHECK(quant_X.is_cuda(), "Grouped GGUF MMVQ input must be CUDA");
  TORCH_CHECK(Y.is_cuda(), "Grouped GGUF MMVQ output must be CUDA");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W[0]));
  const int vecs = quant_X.sizes()[0];
  int rows[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const void* ptrs[8] = {nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr};
  int64_t total_rows = 0;
  for (size_t i = 0; i < W.size(); ++i) {
    TORCH_CHECK(W[i].is_cuda(), "All grouped GGUF shards must be CUDA tensors");
    TORCH_CHECK(W[i].device() == W[0].device(),
                "All grouped GGUF shards must be on the same device");
    TORCH_CHECK(W[i].device() == quant_X.device() && W[i].device() == Y.device(),
                "Grouped GGUF MMVQ tensors must be on the same device");
    rows[i] = W[i].sizes()[0];
    ptrs[i] = W[i].data_ptr();
    total_rows += rows[i];
  }
  TORCH_CHECK(Y.dim() == 2 && Y.size(0) == vecs && Y.size(1) == total_rows,
              "Grouped GGUF MMVQ output has incorrect shape");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  VLLM_DISPATCH_FLOATING_TYPES(
      Y.scalar_type(), "ggml_mul_mat_vec_q8_grouped_same_type_out", [&] {
        ggml_mul_mat_vec_q8_grouped_dispatch<scalar_t>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6],
            ptrs[7], (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col,
            total_rows, vecs, type, stream, rows);
      });
}

torch::Tensor ggml_mul_mat_vec_q8_qkv3(torch::Tensor W0, torch::Tensor W1,
                                       torch::Tensor W2,
                                       torch::Tensor quant_X, int64_t type01,
                                       int64_t col, at::ScalarType dtype) {
  TORCH_CHECK(W0.is_cuda() && W1.is_cuda() && W2.is_cuda(),
              "QKV3 GGUF MMVQ weights must be CUDA tensors");
  TORCH_CHECK(quant_X.is_cuda(), "QKV3 GGUF MMVQ input must be CUDA");
  TORCH_CHECK(W0.device() == W1.device() && W0.device() == W2.device() &&
                  W0.device() == quant_X.device(),
              "QKV3 GGUF MMVQ tensors must be on the same device");
  TORCH_CHECK(type01 == 12 || type01 == 13,
              "QKV3 GGUF MMVQ supports Q4_K/Q4_K/Q6_K or Q5_K/Q5_K/Q6_K");
  TORCH_CHECK(col % QK_K == 0, "QKV3 GGUF MMVQ col must be divisible by QK_K");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W0));
  const int rows0 = W0.sizes()[0];
  const int rows1 = W1.sizes()[0];
  const int rows2 = W2.sizes()[0];
  const int total_rows = rows0 + rows1 + rows2;
  const int vecs = quant_X.sizes()[0];
  auto options = torch::TensorOptions().dtype(dtype).device(W0.device());
  at::Tensor Y = torch::empty({vecs, total_rows}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(dtype, "ggml_mul_mat_vec_q8_qkv3", [&] {
    ggml_mul_mat_vec_q8_qkv3_dispatch<scalar_t>(
        (void*)W0.data_ptr(), (void*)W1.data_ptr(), (void*)W2.data_ptr(),
        (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, rows0, rows1,
        rows2, vecs, type01, stream);
  });
  return Y;
}

torch::Tensor ggml_mul_mat_vec_q8_sharded(std::vector<torch::Tensor> W,
                                          torch::Tensor quant_X,
                                          std::vector<int64_t> types,
                                          int64_t col, at::ScalarType dtype) {
  TORCH_CHECK(!W.empty(), "W must have at least one shard");
  TORCH_CHECK(W.size() == types.size(),
              "W and types must have the same number of elements");
  int vecs = quant_X.sizes()[0];
  int64_t total_rows = 0;
  for (const auto& shard : W) {
    total_rows += shard.sizes()[0];
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W[0]));
  auto options = torch::TensorOptions().dtype(dtype).device(W[0].device());
  at::Tensor Y = torch::empty({vecs, total_rows}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(dtype, "ggml_mul_mat_vec_q8_sharded", [&] {
    const bool group_same_type = gguf_sharded_group_same_type_enabled();
    int64_t row_offset = 0;
    for (size_t i = 0; i < W.size();) {
      if (!group_same_type) {
        const auto& shard = W[i];
        const int row = shard.sizes()[0];
        scalar_t* dst = (scalar_t*)Y.data_ptr() + row_offset;
        ggml_mul_mat_vec_q8_dispatch<scalar_t>(
            (void*)shard.data_ptr(), (void*)quant_X.data_ptr(), dst, col, row,
            vecs, types[i], stream, total_rows);
        row_offset += row;
        ++i;
        continue;
      }

      const int64_t type = types[i];
      int rows[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      const void* ptrs[8] = {nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr};
      int group_rows = 0;
      size_t group_size = 0;
      while (i + group_size < W.size() && group_size < 8 &&
             types[i + group_size] == type) {
        const auto& shard = W[i + group_size];
        rows[group_size] = shard.sizes()[0];
        ptrs[group_size] = shard.data_ptr();
        group_rows += rows[group_size];
        ++group_size;
      }

      scalar_t* dst = (scalar_t*)Y.data_ptr() + row_offset;
      if (group_size > 1) {
        ggml_mul_mat_vec_q8_grouped_dispatch<scalar_t>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6],
            ptrs[7], (void*)quant_X.data_ptr(), dst, col, group_rows, vecs,
            type, stream, rows, total_rows);
      } else {
        ggml_mul_mat_vec_q8_dispatch<scalar_t>(
            (void*)ptrs[0], (void*)quant_X.data_ptr(), dst, col, group_rows,
            vecs, type, stream, total_rows);
      }
      row_offset += group_rows;
      i += group_size;
    }
  });
  return Y;
}

void ggml_mul_mat_vec_q8_sharded_out(std::vector<torch::Tensor> W,
                                     torch::Tensor quant_X, torch::Tensor Y,
                                     std::vector<int64_t> types, int64_t col) {
  TORCH_CHECK(!W.empty(), "W must have at least one shard");
  TORCH_CHECK(W.size() == types.size(),
              "W and types must have the same number of elements");
  TORCH_CHECK(quant_X.is_cuda(), "Sharded GGUF MMVQ input must be CUDA");
  TORCH_CHECK(Y.is_cuda(), "Sharded GGUF MMVQ output must be CUDA");

  int vecs = quant_X.sizes()[0];
  int64_t total_rows = 0;
  for (const auto& shard : W) {
    TORCH_CHECK(shard.is_cuda(), "All sharded GGUF shards must be CUDA tensors");
    TORCH_CHECK(shard.device() == W[0].device(),
                "All sharded GGUF shards must be on the same device");
    TORCH_CHECK(shard.device() == quant_X.device() && shard.device() == Y.device(),
                "Sharded GGUF MMVQ tensors must be on the same device");
    total_rows += shard.sizes()[0];
  }
  TORCH_CHECK(Y.dim() == 2 && Y.size(0) == vecs && Y.size(1) == total_rows,
              "Sharded GGUF MMVQ output has incorrect shape");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W[0]));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  VLLM_DISPATCH_FLOATING_TYPES(Y.scalar_type(), "ggml_mul_mat_vec_q8_sharded_out", [&] {
    const bool group_same_type = gguf_sharded_group_same_type_enabled();
    int64_t row_offset = 0;
    for (size_t i = 0; i < W.size();) {
      if (!group_same_type) {
        const auto& shard = W[i];
        const int row = shard.sizes()[0];
        scalar_t* dst = (scalar_t*)Y.data_ptr() + row_offset;
        ggml_mul_mat_vec_q8_dispatch<scalar_t>(
            (void*)shard.data_ptr(), (void*)quant_X.data_ptr(), dst, col, row,
            vecs, types[i], stream, total_rows);
        row_offset += row;
        ++i;
        continue;
      }

      const int64_t type = types[i];
      int rows[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      const void* ptrs[8] = {nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr};
      int group_rows = 0;
      size_t group_size = 0;
      while (i + group_size < W.size() && group_size < 8 &&
             types[i + group_size] == type) {
        const auto& shard = W[i + group_size];
        rows[group_size] = shard.sizes()[0];
        ptrs[group_size] = shard.data_ptr();
        group_rows += rows[group_size];
        ++group_size;
      }

      scalar_t* dst = (scalar_t*)Y.data_ptr() + row_offset;
      if (group_size > 1) {
        ggml_mul_mat_vec_q8_grouped_dispatch<scalar_t>(
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6],
            ptrs[7], (void*)quant_X.data_ptr(), dst, col, group_rows, vecs,
            type, stream, rows, total_rows);
      } else {
        ggml_mul_mat_vec_q8_dispatch<scalar_t>(
            (void*)ptrs[0], (void*)quant_X.data_ptr(), dst, col, group_rows,
            vecs, type, stream, total_rows);
      }
      row_offset += group_rows;
      i += group_size;
    }
  });
}

torch::Tensor ggml_mul_mat_vec_a8_sharded(std::vector<torch::Tensor> W,
                                          torch::Tensor X,
                                          std::vector<int64_t> types) {
  TORCH_CHECK(!W.empty(), "W must have at least one shard");
  TORCH_CHECK(W.size() == types.size(),
              "W and types must have the same number of elements");
  int col = X.sizes()[1];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  at::Tensor quant_X = ggml_quantize_row_q8_1(X);
  return ggml_mul_mat_vec_q8_sharded(W, quant_X, types, col, X.scalar_type());
}

torch::Tensor ggml_mul_mat_a8(torch::Tensor W,  // quant weight
                              torch::Tensor X,  // input
                              int64_t type, int64_t row) {
  int64_t col = X.sizes()[1];
  int64_t padded = (col + 512 - 1) / 512 * 512;
  int64_t batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  Y.zero_();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_moe_a8(torch::Tensor X,  // input
                          torch::Tensor W,  // expert weights
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens) {
  int64_t col = X.sizes()[1];
  int64_t padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  Y.zero_();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
    }
  });
  return Y;
}

#define VLLM_GGUF_MOE_A8_VEC_DISPATCH(SCALAR_T, W_PTR, Q_PTR, Y_PTR, TOPK_PTR, \
                                       TOP_K, TOKENS, COL, ROW, Q_STRIDE,      \
                                       STREAM, TYPE)                           \
  switch (TYPE) {                                                              \
    case 2:                                                                    \
      moe_vec_q4_0_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 3:                                                                    \
      moe_vec_q4_1_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 6:                                                                    \
      moe_vec_q5_0_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 7:                                                                    \
      moe_vec_q5_1_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 8:                                                                    \
      moe_vec_q8_0_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 10:                                                                   \
      moe_vec_q2_K_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 11:                                                                   \
      moe_vec_q3_K_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 12:                                                                   \
      moe_vec_q4_K_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 13:                                                                   \
      moe_vec_q5_K_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 14:                                                                   \
      moe_vec_q6_K_q8_1_cuda<SCALAR_T>(                                        \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 16:                                                                   \
      moe_vec_iq2_xxs_q8_1_cuda<SCALAR_T>(                                     \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 17:                                                                   \
      moe_vec_iq2_xs_q8_1_cuda<SCALAR_T>(                                      \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 18:                                                                   \
      moe_vec_iq3_xxs_q8_1_cuda<SCALAR_T>(                                     \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 19:                                                                   \
      moe_vec_iq1_s_q8_1_cuda<SCALAR_T>(                                       \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 20:                                                                   \
      moe_vec_iq4_nl_q8_1_cuda<SCALAR_T>(                                      \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 21:                                                                   \
      moe_vec_iq3_s_q8_1_cuda<SCALAR_T>(                                       \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 22:                                                                   \
      moe_vec_iq2_s_q8_1_cuda<SCALAR_T>(                                       \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 23:                                                                   \
      moe_vec_iq4_xs_q8_1_cuda<SCALAR_T>(                                      \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    case 29:                                                                   \
      moe_vec_iq1_m_q8_1_cuda<SCALAR_T>(                                       \
          W_PTR, Q_PTR, Y_PTR, TOPK_PTR, TOP_K, TOKENS, COL, ROW, Q_STRIDE,    \
          STREAM);                                                             \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "Unsupported GGUF MoE MMVQ quantization type: ",      \
                  TYPE);                                                       \
  }

torch::Tensor ggml_moe_a8_vec(torch::Tensor X,  // input
                              torch::Tensor W,  // expert weights
                              torch::Tensor topk_ids, int64_t top_k,
                              int64_t type, int64_t row, int64_t tokens) {
  int64_t col = X.sizes()[1];
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  Y.zero_();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                     (void*)quant_X.data_ptr(), col, tokens,
                                     stream);
    VLLM_GGUF_MOE_A8_VEC_DISPATCH(
        scalar_t, (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
        (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col,
        row, quant_X.stride(0), stream, type);
  });
  return Y;
}

void ggml_moe_a8_vec_out(torch::Tensor X, torch::Tensor W,
                         torch::Tensor topk_ids, torch::Tensor output,
                         torch::Tensor quant_X, int64_t top_k, int64_t type,
                         int64_t row, int64_t tokens) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(quant_X.is_cuda(), "quant_X must be a CUDA tensor");
  TORCH_CHECK(output.device() == X.device() && quant_X.device() == X.device(),
              "output and quant_X must be on the same device as X");
  TORCH_CHECK(output.scalar_type() == X.scalar_type(),
              "output must have the same dtype as X");
  TORCH_CHECK(quant_X.scalar_type() == torch::kInt32,
              "quant_X must have dtype int32");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == tokens * top_k &&
                  output.size(1) == row,
              "output shape must be [tokens * top_k, row]");
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  TORCH_CHECK(quant_X.dim() == 2 && quant_X.size(0) == tokens &&
                  quant_X.size(1) == padded / 32 * 9,
              "quant_X has incorrect shape");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_vec_a8_out", [&] {
    quantize_row_q8_1_cuda<scalar_t>(
        (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    VLLM_GGUF_MOE_A8_VEC_DISPATCH(
        scalar_t, (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
        (scalar_t*)output.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
        col, row, quant_X.stride(0), stream, type);
  });
}

#undef VLLM_GGUF_MOE_A8_VEC_DISPATCH

torch::Tensor ggml_moe_q8_vec(torch::Tensor quant_X,  // input
                              torch::Tensor W,        // expert weights
                              torch::Tensor topk_ids, int64_t top_k,
                              int64_t type, int64_t row, int64_t tokens,
                              int64_t col, at::ScalarType dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(quant_X));
  auto options = torch::TensorOptions().dtype(dtype).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  VLLM_DISPATCH_FLOATING_TYPES(dtype, "ggml_moe_q8_vec", [&] {
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      default:
        TORCH_CHECK(false, "Unsupported GGUF MoE MMVQ quantization type: ",
                    type);
    }
  });
  return Y;
}

template <typename scalar_t, typename weight_t, int qk, int qi,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void moe_vec_q_weighted_sum(
    const void* __restrict__ vx, const void* __restrict__ vy,
    const weight_t* __restrict__ weights, scalar_t* __restrict__ dst,
    const int* __restrict__ topk_ids, const int topk, const int ncols,
    const int nrows, const int token_stride) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int token = blockIdx.z;

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;
  scalar_t total = static_cast<scalar_t>(0.0f);

  for (int k = 0; k < topk; ++k) {
    const int flat_idx = token * topk + k;
    const int expert = topk_ids[flat_idx];
    const block_q_t* x =
        ((const block_q_t*)vx) + expert * nrows * blocks_per_row;
    const block_q8_1* y =
        (const block_q8_1*)(((const int*)vy) + flat_idx * token_stride);

    float partial = 0.0f;
    for (int i = threadIdx.x / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
      const int ibx = row * blocks_per_row + i;
      const int iby = i * (qk / QK8_1);
      const int iqs = vdr * (threadIdx.x % (qi / vdr));
      partial += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      partial += VLLM_SHFL_XOR_SYNC(partial, mask);
    }

    const scalar_t value = static_cast<scalar_t>(partial);
    const float weight = static_cast<float>(weights[flat_idx]);
    const scalar_t product =
        static_cast<scalar_t>(static_cast<float>(value) * weight);
    total += product;
  }

  if (threadIdx.x == 0) {
    dst[token * nrows + row] = total;
  }
}

template <typename scalar_t, typename weight_t, int qk, int qi,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static void moe_vec_q_weighted_sum_cuda(
    const void* vx, const void* vy, const weight_t* weights, scalar_t* dst,
    const int* topk_ids, const int top_k, const int tokens, const int ncols,
    const int nrows, const int token_stride, cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q_weighted_sum<scalar_t, weight_t, qk, qi, block_q_t, vdr,
                         vec_dot_q_cuda>
      <<<block_nums, block_dims, 0, stream>>>(
          vx, vy, weights, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t, typename weight_t>
static void ggml_moe_q8_vec_weighted_sum_dispatch(
    const void* W, const void* quant_X, const weight_t* weights, scalar_t* dst,
    const int* topk_ids, int top_k, int tokens, int64_t type, int row, int col,
    int token_stride, cudaStream_t stream) {
  switch (type) {
    case 2:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK4_0, QI4_0,
                                  block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                                  vec_dot_q4_0_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 3:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK4_0, QI4_1,
                                  block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                                  vec_dot_q4_1_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 6:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK5_0, QI5_0,
                                  block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                                  vec_dot_q5_0_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 7:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK5_1, QI5_1,
                                  block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                                  vec_dot_q5_1_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 8:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK8_0, QI8_0,
                                  block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                                  vec_dot_q8_0_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 10:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI2_K,
                                  block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                                  vec_dot_q2_K_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 11:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI3_K,
                                  block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                                  vec_dot_q3_K_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 12:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI4_K,
                                  block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                                  vec_dot_q4_K_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 13:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI5_K,
                                  block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                                  vec_dot_q5_K_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 14:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI6_K,
                                  block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                                  vec_dot_q6_K_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 16:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI2_XXS,
                                  block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 17:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI2_XS,
                                  block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 18:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI3_XXS,
                                  block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 19:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI1_S,
                                  block_iq1_s, 1, vec_dot_iq1_s_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 20:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK4_NL, QI4_NL,
                                  block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ,
                                  vec_dot_iq4_nl_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 21:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI3_XS,
                                  block_iq3_s, 1, vec_dot_iq3_s_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 22:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI2_S,
                                  block_iq2_s, 1, vec_dot_iq2_s_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 23:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI4_XS,
                                  block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ,
                                  vec_dot_iq4_xs_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    case 29:
      moe_vec_q_weighted_sum_cuda<scalar_t, weight_t, QK_K, QI1_M,
                                  block_iq1_m, 1, vec_dot_iq1_m_q8_1>(
          W, quant_X, weights, dst, topk_ids, top_k, tokens, col, row,
          token_stride, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported GGUF MoE weighted MMVQ quantization type: ",
                  type);
  }
}

torch::Tensor ggml_moe_q8_vec_weighted_sum(
    torch::Tensor quant_X, torch::Tensor W, torch::Tensor topk_ids,
    torch::Tensor topk_weights, int64_t top_k, int64_t type, int64_t row,
    int64_t tokens, int64_t col, at::ScalarType dtype) {
  TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float ||
                  topk_weights.scalar_type() == dtype,
              "topk_weights must be float32 or match output dtype");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(quant_X));
  auto options = torch::TensorOptions().dtype(dtype).device(W.device());
  at::Tensor Y = torch::empty({tokens, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

#define VLLM_LAUNCH_GGUF_MOE_Q8_WEIGHTED_SUM(WEIGHT_T)                    \
  VLLM_DISPATCH_FLOATING_TYPES(dtype, "ggml_moe_q8_vec_weighted_sum", [&] { \
    ggml_moe_q8_vec_weighted_sum_dispatch<scalar_t, WEIGHT_T>(             \
        (void*)W.data_ptr(), (void*)quant_X.data_ptr(),                    \
        (const WEIGHT_T*)topk_weights.data_ptr(), (scalar_t*)Y.data_ptr(), \
        (int*)topk_ids.data_ptr(), top_k, tokens, type, row, col,          \
        quant_X.stride(0), stream);                                        \
  })

  if (topk_weights.scalar_type() == at::ScalarType::Float) {
    VLLM_LAUNCH_GGUF_MOE_Q8_WEIGHTED_SUM(float);
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "ggml_moe_q8_vec_weighted_sum", [&] {
      using weight_t = scalar_t;
      ggml_moe_q8_vec_weighted_sum_dispatch<scalar_t, weight_t>(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
          (const weight_t*)topk_weights.data_ptr(), (scalar_t*)Y.data_ptr(),
          (int*)topk_ids.data_ptr(), top_k, tokens, type, row, col,
          quant_X.stride(0), stream);
    });
  }

#undef VLLM_LAUNCH_GGUF_MOE_Q8_WEIGHTED_SUM

  return Y;
}

void ggml_moe_q8_vec_weighted_sum_out(
    torch::Tensor quant_X, torch::Tensor W, torch::Tensor topk_ids,
    torch::Tensor topk_weights, torch::Tensor output, int64_t top_k,
    int64_t type, int64_t row, int64_t tokens, int64_t col) {
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(output.device() == W.device(),
              "output must be on the same device as W");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == tokens &&
                  output.size(1) == row,
              "output shape must be [tokens, row]");
  TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float ||
                  topk_weights.scalar_type() == output.scalar_type(),
              "topk_weights must be float32 or match output dtype");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(quant_X));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

#define VLLM_LAUNCH_GGUF_MOE_Q8_WEIGHTED_SUM_OUT(WEIGHT_T)                  \
  VLLM_DISPATCH_FLOATING_TYPES(                                             \
      output.scalar_type(), "ggml_moe_q8_vec_weighted_sum_out", [&] {       \
        ggml_moe_q8_vec_weighted_sum_dispatch<scalar_t, WEIGHT_T>(           \
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),                  \
            (const WEIGHT_T*)topk_weights.data_ptr(),                        \
            (scalar_t*)output.data_ptr(), (int*)topk_ids.data_ptr(), top_k,  \
            tokens, type, row, col, quant_X.stride(0), stream);              \
      })

  if (topk_weights.scalar_type() == at::ScalarType::Float) {
    VLLM_LAUNCH_GGUF_MOE_Q8_WEIGHTED_SUM_OUT(float);
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "ggml_moe_q8_vec_weighted_sum_out", [&] {
          using weight_t = scalar_t;
          ggml_moe_q8_vec_weighted_sum_dispatch<scalar_t, weight_t>(
              (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
              (const weight_t*)topk_weights.data_ptr(),
              (scalar_t*)output.data_ptr(), (int*)topk_ids.data_ptr(), top_k,
              tokens, type, row, col, quant_X.stride(0), stream);
        });
  }

#undef VLLM_LAUNCH_GGUF_MOE_Q8_WEIGHTED_SUM_OUT
}

void ggml_moe_a8_vec_silu_q8_weighted_sum_out(
    torch::Tensor X, torch::Tensor W1, torch::Tensor W2,
    torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w1_output, torch::Tensor quant_X,
    torch::Tensor quant_w1_output, torch::Tensor output, int64_t top_k,
    int64_t w1_type, int64_t w2_type, int64_t w1_row, int64_t w2_row,
    int64_t tokens) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(W1.is_cuda(), "W1 must be a CUDA tensor");
  TORCH_CHECK(W2.is_cuda(), "W2 must be a CUDA tensor");
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be a CUDA tensor");
  TORCH_CHECK(topk_weights.is_cuda(), "topk_weights must be a CUDA tensor");
  TORCH_CHECK(w1_output.is_cuda(), "w1_output must be a CUDA tensor");
  TORCH_CHECK(quant_X.is_cuda(), "quant_X must be a CUDA tensor");
  TORCH_CHECK(quant_w1_output.is_cuda(),
              "quant_w1_output must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(X.dim() == 2, "X must have shape [tokens, hidden]");
  TORCH_CHECK(w1_row % 2 == 0, "w1_row must be divisible by 2");
  TORCH_CHECK(w1_output.dim() == 2 && w1_output.size(0) == tokens * top_k &&
                  w1_output.size(1) == w1_row,
              "w1_output shape must be [tokens * top_k, w1_row]");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == tokens &&
                  output.size(1) == w2_row,
              "output shape must be [tokens, w2_row]");

  ggml_moe_a8_vec_out(X, W1, topk_ids, w1_output, quant_X, top_k, w1_type,
                      w1_row, tokens);
  ggml_silu_and_mul_quantize_row_q8_1_out(w1_output, quant_w1_output);
  ggml_moe_q8_vec_weighted_sum_out(quant_w1_output, W2, topk_ids,
                                   topk_weights, output, top_k, w2_type,
                                   w2_row, tokens, w1_row / 2);
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
  }
  return 0;
}
