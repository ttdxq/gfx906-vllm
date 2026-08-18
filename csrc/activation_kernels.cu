#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}
// Activation and gating kernel template.

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename scalar_t>
__global__ void shared_expert_gate_mul_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight, const int64_t tokens,
    const int64_t input_hidden, const int64_t output_hidden,
    const int64_t input_stride_token, const int64_t input_stride_hidden,
    const int64_t out_stride_token, const int64_t out_stride_hidden,
    const int64_t weight_stride_hidden) {
  extern __shared__ float partial_sums[];
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= tokens) {
    return;
  }

  float sum = 0.0f;
  for (int64_t idx = threadIdx.x; idx < input_hidden; idx += blockDim.x) {
    const float x =
        static_cast<float>(VLLM_LDG(&input[token_idx * input_stride_token +
                                           idx * input_stride_hidden]));
    const float w =
        static_cast<float>(VLLM_LDG(&weight[idx * weight_stride_hidden]));
    sum += x * w;
  }
  partial_sums[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float gate = 1.0f / (1.0f + expf(-partial_sums[0]));
  for (int64_t idx = threadIdx.x; idx < output_hidden; idx += blockDim.x) {
    scalar_t* out_ptr =
        &out[token_idx * out_stride_token + idx * out_stride_hidden];
    const float value = static_cast<float>(VLLM_LDG(out_ptr));
    *out_ptr = static_cast<scalar_t>(value * gate);
  }
}

template <typename scalar_t>
__global__ void shared_expert_gate_add_kernel(
    scalar_t* __restrict__ routed_out, const scalar_t* __restrict__ shared_out,
    const scalar_t* __restrict__ input, const scalar_t* __restrict__ weight,
    const int64_t tokens, const int64_t input_hidden,
    const int64_t output_hidden, const int64_t input_stride_token,
    const int64_t input_stride_hidden, const int64_t shared_stride_token,
    const int64_t shared_stride_hidden, const int64_t routed_stride_token,
    const int64_t routed_stride_hidden, const int64_t weight_stride_hidden) {
  extern __shared__ float partial_sums[];
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= tokens) {
    return;
  }

  float sum = 0.0f;
  for (int64_t idx = threadIdx.x; idx < input_hidden; idx += blockDim.x) {
    const float x =
        static_cast<float>(VLLM_LDG(&input[token_idx * input_stride_token +
                                           idx * input_stride_hidden]));
    const float w =
        static_cast<float>(VLLM_LDG(&weight[idx * weight_stride_hidden]));
    sum += x * w;
  }
  partial_sums[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float gate = 1.0f / (1.0f + expf(-partial_sums[0]));
  for (int64_t idx = threadIdx.x; idx < output_hidden; idx += blockDim.x) {
    scalar_t* routed_ptr =
        &routed_out[token_idx * routed_stride_token + idx * routed_stride_hidden];
    const scalar_t* shared_ptr =
        &shared_out[token_idx * shared_stride_token + idx * shared_stride_hidden];
    const float routed = static_cast<float>(VLLM_LDG(routed_ptr));
    const float shared = static_cast<float>(VLLM_LDG(shared_ptr));
    *routed_ptr = static_cast<scalar_t>(routed + shared * gate);
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                 \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  dim3 grid(num_tokens);                                                 \
  dim3 block(std::min(d, 1024));                                         \
  if (num_tokens == 0) {                                                 \
    return;                                                              \
  }                                                                      \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(), "act_and_mul_kernel", [&] {                   \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>  \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
}

void mul_and_silu(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
  // applies the silu to the latter half of the input.
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
}

void shared_expert_gate_mul(torch::Tensor& out,     // [tokens, output_hidden]
                            torch::Tensor& input,   // [tokens, input_hidden]
                            torch::Tensor& weight)  // [1, input_hidden]
{
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(out.dim() == 2, "out must have shape [tokens, output_hidden]");
  TORCH_CHECK(input.dim() == 2, "input must have shape [tokens, input_hidden]");
  TORCH_CHECK(weight.numel() == input.size(1),
              "weight must have input_hidden elements");
  TORCH_CHECK(out.size(0) == input.size(0),
              "out and input token dimensions must match");
  TORCH_CHECK(out.scalar_type() == input.scalar_type() &&
                  input.scalar_type() == weight.scalar_type(),
              "out, input, and weight dtypes must match");

  const int64_t tokens = input.size(0);
  if (tokens == 0) {
    return;
  }

  const int64_t input_hidden = input.size(1);
  const int64_t output_hidden = out.size(1);
  constexpr int threads = 256;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t weight_stride_hidden =
      weight.dim() == 1 ? weight.stride(0) : weight.stride(weight.dim() - 1);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "shared_expert_gate_mul_kernel", [&] {
        vllm::shared_expert_gate_mul_kernel<scalar_t>
            <<<tokens, threads, threads * sizeof(float), stream>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(), tokens, input_hidden, output_hidden,
                input.stride(0), input.stride(1), out.stride(0), out.stride(1),
                weight_stride_hidden);
      });
}

void shared_expert_gate_add(torch::Tensor& routed_out,  // [tokens, output_hidden]
                            torch::Tensor& shared_out,  // [tokens, output_hidden]
                            torch::Tensor& input,       // [tokens, input_hidden]
                            torch::Tensor& weight)      // [1, input_hidden]
{
  TORCH_CHECK(routed_out.is_cuda(), "routed_out must be a CUDA tensor");
  TORCH_CHECK(shared_out.is_cuda(), "shared_out must be a CUDA tensor");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(routed_out.dim() == 2,
              "routed_out must have shape [tokens, output_hidden]");
  TORCH_CHECK(shared_out.dim() == 2,
              "shared_out must have shape [tokens, output_hidden]");
  TORCH_CHECK(input.dim() == 2, "input must have shape [tokens, input_hidden]");
  TORCH_CHECK(weight.numel() == input.size(1),
              "weight must have input_hidden elements");
  TORCH_CHECK(routed_out.sizes() == shared_out.sizes(),
              "routed_out and shared_out shapes must match");
  TORCH_CHECK(routed_out.size(0) == input.size(0),
              "output and input token dimensions must match");
  TORCH_CHECK(routed_out.scalar_type() == shared_out.scalar_type() &&
                  shared_out.scalar_type() == input.scalar_type() &&
                  input.scalar_type() == weight.scalar_type(),
              "routed_out, shared_out, input, and weight dtypes must match");

  const int64_t tokens = input.size(0);
  if (tokens == 0) {
    return;
  }

  const int64_t input_hidden = input.size(1);
  const int64_t output_hidden = routed_out.size(1);
  constexpr int threads = 256;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t weight_stride_hidden =
      weight.dim() == 1 ? weight.stride(0) : weight.stride(weight.dim() - 1);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "shared_expert_gate_add_kernel", [&] {
        vllm::shared_expert_gate_add_kernel<scalar_t>
            <<<tokens, threads, threads * sizeof(float), stream>>>(
                routed_out.data_ptr<scalar_t>(), shared_out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), tokens,
                input_hidden, output_hidden, input.stride(0), input.stride(1),
                shared_out.stride(0), shared_out.stride(1), routed_out.stride(0),
                routed_out.stride(1), weight_stride_hidden);
      });
}

namespace vllm {

template <typename T>
__device__ __forceinline__ T fatrelu_kernel(const T& x, const float threshold) {
  const float f = (float)x;
  return (T)(f > threshold ? f : 0.0f);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&, const float)>
__global__ void act_and_mul_kernel_with_param(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const int d,
    const float param) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = ACT_FN(x, param) * y;
  }
}

template <typename T>
__device__ __forceinline__ T swigluoai_and_mul(const T& gate, const T& up,
                                               float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + expf(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
__global__ void swigluoai_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const float alpha, const float limit) {
  const int64_t token_idx = blockIdx.x;
  // TODO: Vectorize loads and stores.
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // gate = x[..., ::2]  (even indices)
    const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx]);
    // up = x[..., 1::2]   (odd indices)
    const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx + 1]);

    out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
  }
}

}  // namespace vllm

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(KERNEL, PARAM)         \
  int d = input.size(-1) / 2;                                           \
  int64_t num_tokens = input.numel() / input.size(-1);                  \
  dim3 grid(num_tokens);                                                \
  dim3 block(std::min(d, 1024));                                        \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));     \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();         \
  VLLM_DISPATCH_FLOATING_TYPES(                                         \
      input.scalar_type(), "act_and_mul_kernel_with_param", [&] {       \
        vllm::act_and_mul_kernel_with_param<scalar_t, KERNEL<scalar_t>> \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),      \
                                         input.data_ptr<scalar_t>(), d, \
                                         PARAM);                        \
      });

#define LAUNCH_SIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                          \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {            \
        vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>             \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),             \
                                         input.data_ptr<scalar_t>(), d, ALPHA, \
                                         LIMIT);                               \
      });

void fatrelu_and_mul(torch::Tensor& out,    // [..., d],
                     torch::Tensor& input,  // [..., 2 * d]
                     double threshold) {
  LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(vllm::fatrelu_kernel, threshold);
}
void swigluoai_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}
namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

}  // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / d;                                      \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] { \
    vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                        \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                 \
                                     input.data_ptr<scalar_t>(), d);           \
  });

namespace vllm {

template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

}  // namespace vllm

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
}
