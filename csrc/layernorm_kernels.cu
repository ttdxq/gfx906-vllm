#include "type_convert.cuh"
#include "dispatch_utils.h"
#include "cub_helpers.h"
#include "core/batch_invariant.hpp"
#include "quantization/vectorization_utils.cuh"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const int64_t input_stride_d2,        // input.stride(-2)
    const int64_t input_stride_d3,        // input.stride(-3)
    const int64_t input_stride_d4,        // input.stride(-4)
    const int64_t input_shape_d2,         // input.size(-2)
    const int64_t input_shape_d3,         // input.size(-3)
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  const scalar_t* input_row;
  if constexpr (NUM_DIMS == 2) {
    // 2D for layernorm normal case [batch_size, hidden]
    input_row = input + blockIdx.x * input_stride_d2;
  } else if constexpr (NUM_DIMS == 3) {
    // 3D for q/k norm [batch_size, num_heads, head_size]
    int batch_idx = blockIdx.x / input_shape_d2;
    int head_idx = blockIdx.x % input_shape_d2;
    input_row =
        input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
  } else if constexpr (NUM_DIMS == 4) {
    // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
    int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
    int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
    int seq_idx = remaining / input_shape_d2;
    int head_idx = remaining % input_shape_d2;
    input_row = input + batch_idx * input_stride_d4 +
                seq_idx * input_stride_d3 + head_idx * input_stride_d2;
  }

  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + blockIdx.x * hidden_size;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
    vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      dst.val[j] = ((scalar_t)(x * s_variance)) * src2.val[j];
    }
    v_out[i] = dst;
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * input_stride + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

template <typename scalar_t>
__global__ void rms_norm_gated_gfx906_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight, const scalar_t* __restrict__ gate,
    const float epsilon, const int hidden_size, const int64_t input_stride,
    const int64_t gate_stride, const bool norm_before_gate) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const int64_t row = blockIdx.x;
  const scalar_t* input_row = input + row * input_stride;
  const scalar_t* gate_row = gate + row * gate_stride;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input_row[idx]);
    if (!norm_before_gate) {
      float z = static_cast<float>(gate_row[idx]);
      x *= z / (1.0f + expf(-z));
    }
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + row * hidden_size;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input_row[idx]);
    float z = static_cast<float>(gate_row[idx]);
    const float gate_val = z / (1.0f + expf(-z));
    float val = norm_before_gate ? x * s_variance * gate_val
                                 : x * gate_val * s_variance;
    val *= static_cast<float>(weight[idx]);
    out_row[idx] = static_cast<scalar_t>(val);
  }
}

template <typename scalar_t>
__global__ void gemma_rms_norm_gfx906_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight, const float epsilon,
    const int hidden_size, const int64_t input_stride) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const int64_t row = blockIdx.x;
  const scalar_t* input_row = input + row * input_stride;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input_row[idx]);
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + row * hidden_size;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = static_cast<float>(input_row[idx]);
    const float w = 1.0f + static_cast<float>(weight[idx]);
    out_row[idx] = static_cast<scalar_t>(x * s_variance * w);
  }
}

template <typename scalar_t, typename residual_t>
__global__ void gemma_fused_add_rms_norm_gfx906_kernel(
    scalar_t* __restrict__ out, float* __restrict__ residual_out,
    const scalar_t* __restrict__ input,
    const residual_t* __restrict__ residual,
    const scalar_t* __restrict__ weight, const float epsilon,
    const int hidden_size, const int64_t input_stride,
    const int64_t residual_stride) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const int64_t row = blockIdx.x;
  const scalar_t* input_row = input + row * input_stride;
  const residual_t* residual_row = residual + row * residual_stride;
  float* residual_out_row = residual_out + row * hidden_size;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = static_cast<float>(input_row[idx]) +
                    static_cast<float>(residual_row[idx]);
    variance += x * x;
    residual_out_row[idx] = x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + row * hidden_size;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = residual_out_row[idx];
    const float w = 1.0f + static_cast<float>(weight[idx]);
    out_row[idx] = static_cast<scalar_t>(x * s_variance * w);
  }
}

}  // namespace vllm

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);

  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();
  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_RANK234(num_dims, [&] {
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
      const int calculated_vec_size =
          std::gcd(16 / sizeof(scalar_t), hidden_size);
      const int block_size =
          std::min(hidden_size / calculated_vec_size, max_block_size);
      dim3 block(block_size);
      VLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
        vllm::rms_norm_kernel<scalar_t, vec_size, tensor_rank>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                input_stride_d2, input_stride_d3, input_stride_d4,
                input_shape_d2, input_shape_d3, weight.data_ptr<scalar_t>(),
                epsilon, num_tokens, hidden_size);
      });
    });
  });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                    \
  VLLM_DISPATCH_FLOATING_TYPES(                                             \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {               \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                    \
            <<<grid, block, 0, stream>>>(                                   \
                input.data_ptr<scalar_t>(), input_stride,                   \
                residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), \
                epsilon, num_tokens, hidden_size);                          \
      });

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  TORCH_CHECK(input.scalar_type() == residual.scalar_type());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes =
      vector_width * 2;  // vector_width * sizeof(bfloat16 or float16) (float32
                         // falls back to non-vectorized version anyway)
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
      !batch_invariant_launch) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}

torch::Tensor rms_norm_gated_gfx906(torch::Tensor input,    // [..., hidden_size]
                                    torch::Tensor weight,   // [hidden_size]
                                    torch::Tensor gate,     // [..., hidden_size]
                                    double epsilon,
                                    bool norm_before_gate) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == gate.scalar_type(),
              "input and gate must have the same dtype");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input and weight must have the same dtype");
  TORCH_CHECK(input.sizes() == gate.sizes(),
              "input and gate must have the same shape");
  TORCH_CHECK(input.dim() == 2, "input must have shape [tokens, hidden_size]");
  TORCH_CHECK(input.stride(-1) == 1, "input last dimension must be contiguous");
  TORCH_CHECK(gate.stride(-1) == 1, "gate last dimension must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int hidden_size = input.size(-1);
  TORCH_CHECK(weight.numel() == hidden_size,
              "weight size must match input hidden size");
  const int num_tokens = input.numel() / hidden_size;
  const int64_t input_stride = input.stride(-2);
  const int64_t gate_stride = gate.stride(-2);
  auto out = torch::empty_like(input, input.options().memory_format(
                                          c10::MemoryFormat::Contiguous));

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(),
                               "rms_norm_gated_gfx906_kernel", [&] {
                                 vllm::rms_norm_gated_gfx906_kernel<scalar_t>
                                     <<<grid, block, 0, stream>>>(
                                         out.data_ptr<scalar_t>(),
                                         input.data_ptr<scalar_t>(),
                                         weight.data_ptr<scalar_t>(),
                                         gate.data_ptr<scalar_t>(), epsilon,
                                         hidden_size, input_stride, gate_stride,
                                         norm_before_gate);
                               });
  return out;
}

torch::Tensor gemma_rms_norm_gfx906(torch::Tensor input,
                                    torch::Tensor weight,
                                    double epsilon) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input and weight must have the same dtype");
  TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
  if (input.stride(-1) != 1 || !input.is_contiguous()) {
    input = input.contiguous();
  }
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int hidden_size = input.size(-1);
  TORCH_CHECK(weight.numel() == hidden_size,
              "weight size must match input hidden size");
  const int num_tokens = input.numel() / hidden_size;
  const int64_t input_stride = input.stride(-2);
  auto out = torch::empty_like(input, input.options().memory_format(
                                          c10::MemoryFormat::Contiguous));

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(),
                               "gemma_rms_norm_gfx906_kernel", [&] {
                                 vllm::gemma_rms_norm_gfx906_kernel<scalar_t>
                                     <<<grid, block, 0, stream>>>(
                                         out.data_ptr<scalar_t>(),
                                         input.data_ptr<scalar_t>(),
                                         weight.data_ptr<scalar_t>(), epsilon,
                                         hidden_size, input_stride);
                               });
  return out;
}

std::vector<torch::Tensor> gemma_fused_add_rms_norm_gfx906(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
    double epsilon) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input and weight must have the same dtype");
  TORCH_CHECK(residual.scalar_type() == input.scalar_type() ||
                  residual.scalar_type() == at::ScalarType::Float,
              "residual must be float32 or have the same dtype as input");
  TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
  TORCH_CHECK(residual.sizes() == input.sizes(),
              "residual shape must match input shape");
  if (input.stride(-1) != 1 || !input.is_contiguous()) {
    input = input.contiguous();
  }
  if (residual.stride(-1) != 1 || !residual.is_contiguous()) {
    residual = residual.contiguous();
  }
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int hidden_size = input.size(-1);
  TORCH_CHECK(weight.numel() == hidden_size,
              "weight size must match input hidden size");
  const int num_tokens = input.numel() / hidden_size;
  const int64_t input_stride = input.stride(-2);
  const int64_t residual_stride = residual.stride(-2);
  auto out = torch::empty_like(input, input.options().memory_format(
                                          c10::MemoryFormat::Contiguous));
  auto residual_out =
      torch::empty(input.sizes(),
                   torch::TensorOptions().dtype(torch::kFloat32).device(
                       input.device()));

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (residual.scalar_type() == at::ScalarType::Float) {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "gemma_fused_add_rms_norm_gfx906_kernel", [&] {
          vllm::gemma_fused_add_rms_norm_gfx906_kernel<scalar_t, float>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), residual_out.data_ptr<float>(),
                  input.data_ptr<scalar_t>(), residual.data_ptr<float>(),
                  weight.data_ptr<scalar_t>(), epsilon, hidden_size,
                  input_stride, residual_stride);
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "gemma_fused_add_rms_norm_gfx906_kernel", [&] {
          vllm::gemma_fused_add_rms_norm_gfx906_kernel<scalar_t, scalar_t>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), residual_out.data_ptr<float>(),
                  input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
                  weight.data_ptr<scalar_t>(), epsilon, hidden_size,
                  input_stride, residual_stride);
        });
  }

  return {out, residual_out};
}
