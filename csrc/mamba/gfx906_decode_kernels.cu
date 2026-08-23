#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdio>
#include <cstdlib>

#include "../dispatch_utils.h"

namespace {

static int gfx906_gdn_packed_decode_threads() {
  static const int threads = [] {
    const char* raw_threads = std::getenv("VLLM_GDN_PACKED_DECODE_THREADS");
    if (raw_threads == nullptr) {
      return 256;
    }
    const int requested_threads = std::atoi(raw_threads);
    return (requested_threads == 128 || requested_threads == 256)
               ? requested_threads
               : 256;
  }();
  return threads;
}

static bool gfx906_gdn_specialized_packed_decode_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("VLLM_GDN_SPECIALIZED_PACKED_DECODE");
    return env == nullptr || env[0] != '0';
  }();
  return enabled;
}

template <typename scalar_t, typename state_t>
__global__ void causal_conv1d_gfx906_decode_update_kernel(
    const scalar_t* __restrict__ x, state_t* __restrict__ conv_state,
    const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
    const int32_t* __restrict__ conv_state_indices, scalar_t* __restrict__ out,
    int64_t batch, int64_t dim, int64_t width, int64_t state_len,
    int64_t x_stride_b, int64_t x_stride_d, int64_t state_stride_b,
    int64_t state_stride_d, int64_t state_stride_s, int64_t weight_stride_d,
    int64_t weight_stride_w, int64_t out_stride_b, int64_t out_stride_d,
    int64_t pad_slot_id, bool has_bias, bool silu_activation) {
  const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * dim;
  if (linear_idx >= total) {
    return;
  }

  const int64_t batch_idx = linear_idx / dim;
  const int64_t dim_idx = linear_idx - batch_idx * dim;
  const int64_t raw_state_idx =
      conv_state_indices == nullptr ? batch_idx : conv_state_indices[batch_idx];
  if (raw_state_idx == pad_slot_id) {
    out[batch_idx * out_stride_b + dim_idx * out_stride_d] =
        static_cast<scalar_t>(0.0f);
    return;
  }
  const int64_t state_idx = raw_state_idx;

  float acc = has_bias ? static_cast<float>(bias[dim_idx]) : 0.0f;
  const float x_val =
      static_cast<float>(x[batch_idx * x_stride_b + dim_idx * x_stride_d]);
  for (int64_t w_idx = 0; w_idx < width; ++w_idx) {
    const int64_t source_idx = state_len - (width - 1) + w_idx;
    const float source =
        (w_idx == width - 1)
            ? x_val
            : static_cast<float>(
                  conv_state[state_idx * state_stride_b +
                             dim_idx * state_stride_d +
                             source_idx * state_stride_s]);
    acc += source *
           static_cast<float>(
               weight[dim_idx * weight_stride_d + w_idx * weight_stride_w]);
  }
  if (silu_activation) {
    acc = acc / (1.0f + expf(-acc));
  }
  out[batch_idx * out_stride_b + dim_idx * out_stride_d] =
      static_cast<scalar_t>(acc);

  if (state_len <= 0) {
    return;
  }
  const int64_t keep_old = min(width - 1, state_len - 1);
  const int64_t target_start = state_len - 1 - keep_old;
  for (int64_t s_idx = 0; s_idx < target_start; ++s_idx) {
    conv_state[state_idx * state_stride_b + dim_idx * state_stride_d +
               s_idx * state_stride_s] = static_cast<state_t>(0.0f);
  }
  for (int64_t s_idx = 0; s_idx < keep_old; ++s_idx) {
    const int64_t source_idx = state_len - keep_old + s_idx;
    const int64_t target_idx = target_start + s_idx;
    conv_state[state_idx * state_stride_b + dim_idx * state_stride_d +
               target_idx * state_stride_s] =
        conv_state[state_idx * state_stride_b + dim_idx * state_stride_d +
                   source_idx * state_stride_s];
  }
  conv_state[state_idx * state_stride_b + dim_idx * state_stride_d +
             (state_len - 1) * state_stride_s] = static_cast<state_t>(x_val);
}

template <typename scalar_t, typename state_t, typename accepted_t>
__global__ void causal_conv1d_gfx906_mtp_update_width4_kernel(
    const scalar_t* __restrict__ x, state_t* __restrict__ conv_state,
    const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ cu_seqlens,
    const accepted_t* __restrict__ num_accepted_tokens,
    scalar_t* __restrict__ out, int64_t dim, int64_t state_len,
    int64_t x_stride_t, int64_t x_stride_d, int64_t state_stride_slot,
    int64_t state_stride_d, int64_t state_stride_s,
    int64_t weight_stride_d, int64_t weight_stride_w,
    int64_t indices_stride_req, int64_t out_stride_t,
    int64_t out_stride_d, int64_t pad_slot_id, bool has_bias,
    bool silu_activation) {
  const int64_t request_idx = blockIdx.x;
  const int64_t dim_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (dim_idx >= dim) {
    return;
  }

  const int64_t seq_start = cu_seqlens[request_idx];
  const int64_t seq_end = cu_seqlens[request_idx + 1];
  const int64_t query_len = seq_end - seq_start;
  if (query_len <= 0) {
    return;
  }
  const int64_t state_offset =
      static_cast<int64_t>(num_accepted_tokens[request_idx]) - 1;
  if (state_offset < 0 || state_offset + 2 >= state_len ||
      query_len > state_len - 2) {
    return;
  }
  const int64_t state_idx = state_indices[request_idx * indices_stride_req];
  if (state_idx < 0 || state_idx == pad_slot_id) {
    return;
  }

  float col0 = static_cast<float>(
      conv_state[state_idx * state_stride_slot + dim_idx * state_stride_d +
                 state_offset * state_stride_s]);
  float col1 = static_cast<float>(
      conv_state[state_idx * state_stride_slot + dim_idx * state_stride_d +
                 (state_offset + 1) * state_stride_s]);
  float col2 = static_cast<float>(
      conv_state[state_idx * state_stride_slot + dim_idx * state_stride_d +
                 (state_offset + 2) * state_stride_s]);
  const float rolled_col0 = col1;
  const float rolled_col1 = col2;
  const float w0 = static_cast<float>(
      weight[dim_idx * weight_stride_d + 0 * weight_stride_w]);
  const float w1 = static_cast<float>(
      weight[dim_idx * weight_stride_d + 1 * weight_stride_w]);
  const float w2 = static_cast<float>(
      weight[dim_idx * weight_stride_d + 2 * weight_stride_w]);
  const float w3 = static_cast<float>(
      weight[dim_idx * weight_stride_d + 3 * weight_stride_w]);
  const float bias_val =
      has_bias ? static_cast<float>(bias[dim_idx]) : 0.0f;
  float candidate_tokens[64];

  for (int64_t token_position = 0; token_position < query_len;
       ++token_position) {
    const int64_t token_idx = seq_start + token_position;
    const float x_val = static_cast<float>(
        x[token_idx * x_stride_t + dim_idx * x_stride_d]);
    candidate_tokens[token_position] = x_val;
    float acc = bias_val + col0 * w0 + col1 * w1 + col2 * w2 + x_val * w3;
    if (silu_activation) {
      acc = acc / (1.0f + expf(-acc));
    }
    out[token_idx * out_stride_t + dim_idx * out_stride_d] =
        static_cast<scalar_t>(acc);
    col0 = col1;
    col1 = col2;
    col2 = x_val;
  }

  const int64_t state_base = state_idx * state_stride_slot +
                             dim_idx * state_stride_d;
  conv_state[state_base] = static_cast<state_t>(rolled_col0);
  conv_state[state_base + state_stride_s] = static_cast<state_t>(rolled_col1);
  for (int64_t token_position = 0; token_position < query_len;
       ++token_position) {
    conv_state[state_base + (token_position + 2) * state_stride_s] =
        static_cast<state_t>(candidate_tokens[token_position]);
  }
}

template <typename scalar_t, typename state_t, typename accepted_t>
__global__ void causal_conv1d_gfx906_mtp_update_kernel(
    const scalar_t* __restrict__ x, state_t* __restrict__ conv_state,
    const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ cu_seqlens,
    const accepted_t* __restrict__ num_accepted_tokens,
    scalar_t* __restrict__ out, int64_t dim, int64_t width,
    int64_t state_len, int64_t x_stride_t, int64_t x_stride_d,
    int64_t state_stride_slot,
    int64_t state_stride_d, int64_t state_stride_s,
    int64_t weight_stride_d, int64_t weight_stride_w,
    int64_t indices_stride_req, int64_t out_stride_t,
    int64_t out_stride_d, int64_t pad_slot_id, bool has_bias,
    bool silu_activation) {
  const int64_t request_idx = blockIdx.x;
  const int64_t dim_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (dim_idx >= dim) {
    return;
  }

  const int64_t seq_start = cu_seqlens[request_idx];
  const int64_t seq_end = cu_seqlens[request_idx + 1];
  const int64_t query_len = seq_end - seq_start;
  if (query_len <= 0) {
    return;
  }
  const int64_t state_offset =
      static_cast<int64_t>(num_accepted_tokens[request_idx]) - 1;
  if (state_offset < 0 || state_offset + width - 2 >= state_len ||
      query_len > state_len - (width - 2)) {
    return;
  }
  const int64_t state_idx = state_indices[request_idx * indices_stride_req];
  if (state_idx < 0 || state_idx == pad_slot_id) {
    return;
  }

  float history[5];
  float candidate_tokens[64];
  for (int64_t history_position = 0; history_position < width - 1;
       ++history_position) {
    history[history_position] = static_cast<float>(conv_state[
        state_idx * state_stride_slot + dim_idx * state_stride_d +
        (state_offset + history_position) * state_stride_s]);
  }

  for (int64_t token_position = 0; token_position < query_len;
       ++token_position) {
    const int64_t token_idx = seq_start + token_position;
    const float x_val = static_cast<float>(
        x[token_idx * x_stride_t + dim_idx * x_stride_d]);
    candidate_tokens[token_position] = x_val;
    float acc = has_bias ? static_cast<float>(bias[dim_idx]) : 0.0f;
    for (int64_t width_position = 0; width_position < width; ++width_position) {
      const float input_val =
          width_position == width - 1
              ? x_val
              : history[width_position];
      acc += input_val * static_cast<float>(
                             weight[dim_idx * weight_stride_d +
                                    width_position * weight_stride_w]);
    }
    if (silu_activation) {
      acc = acc / (1.0f + expf(-acc));
    }
    out[token_idx * out_stride_t + dim_idx * out_stride_d] =
        static_cast<scalar_t>(acc);
    for (int64_t history_position = 0; history_position < width - 2;
         ++history_position) {
      history[history_position] = history[history_position + 1];
    }
    history[width - 2] = x_val;
  }

  const int64_t state_base =
      state_idx * state_stride_slot + dim_idx * state_stride_d;
  for (int64_t history_position = 0; history_position < width - 2;
       ++history_position) {
    conv_state[state_base + history_position * state_stride_s] =
        conv_state[state_base +
                   (state_offset + history_position + 1) * state_stride_s];
  }
  for (int64_t token_position = 0; token_position < query_len;
       ++token_position) {
    conv_state[state_base + (width - 2 + token_position) * state_stride_s] =
        static_cast<state_t>(candidate_tokens[token_position]);
  }
}

template <typename scalar_t, typename state_t>
__global__ void fused_sigmoid_gating_delta_rule_gfx906_decode_kernel(
    const scalar_t* __restrict__ A_log, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ dt_bias,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, state_t* __restrict__ state,
    scalar_t* __restrict__ out, int64_t tokens, int64_t heads, int64_t kv_heads,
    int64_t key_dim, int64_t value_dim, int64_t q_stride_t,
    int64_t q_stride_h, int64_t q_stride_k, int64_t k_stride_t,
    int64_t k_stride_h, int64_t k_stride_k, int64_t v_stride_t,
    int64_t v_stride_h, int64_t v_stride_v, int64_t a_stride_t,
    int64_t a_stride_h, int64_t b_stride_t, int64_t b_stride_h,
    int64_t state_stride_t, int64_t state_stride_h, int64_t state_stride_v,
    int64_t state_stride_k, int64_t out_stride_t, int64_t out_stride_h,
    int64_t out_stride_v, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel) {
  const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = tokens * kv_heads * value_dim;
  if (linear_idx >= total) {
    return;
  }

  const int64_t value_idx = linear_idx % value_dim;
  const int64_t hv_idx = (linear_idx / value_dim) % kv_heads;
  const int64_t token_idx = linear_idx / (value_dim * kv_heads);
  const int64_t head_ratio = kv_heads / heads;
  const int64_t q_head_idx = hv_idx / head_ratio;

  float q_norm = 0.0f;
  float k_norm = 0.0f;
  if (use_qk_l2norm_in_kernel) {
    for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
      const float q_val = static_cast<float>(
          q[token_idx * q_stride_t + q_head_idx * q_stride_h +
            k_idx * q_stride_k]);
      const float k_val = static_cast<float>(
          k[token_idx * k_stride_t + q_head_idx * k_stride_h +
            k_idx * k_stride_k]);
      q_norm += q_val * q_val;
      k_norm += k_val * k_val;
    }
    q_norm = rsqrtf(q_norm + 1e-6f);
    k_norm = rsqrtf(k_norm + 1e-6f);
  } else {
    q_norm = 1.0f;
    k_norm = 1.0f;
  }

  const float x = static_cast<float>(
                      a[token_idx * a_stride_t + hv_idx * a_stride_h]) +
                  static_cast<float>(dt_bias[hv_idx]);
  const float beta_x = static_cast<float>(beta) * x;
  const float softplus_x =
      beta_x <= static_cast<float>(threshold)
          ? static_cast<float>(1.0 / beta) * log1pf(expf(beta_x))
          : x;
  const float g =
      -expf(static_cast<float>(A_log[hv_idx])) * softplus_x;
  const float beta_t =
      1.0f / (1.0f + expf(-static_cast<float>(
                            b[token_idx * b_stride_t + hv_idx * b_stride_h])));
  const float decay = expf(g);

  float state_dot_k = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val =
        static_cast<float>(k[token_idx * k_stride_t + q_head_idx * k_stride_h +
                             k_idx * k_stride_k]) *
        k_norm;
    const int64_t state_offset =
        token_idx * state_stride_t + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const float decayed_state = static_cast<float>(state[state_offset]) * decay;
    state_dot_k += decayed_state * k_val;
  }

  const float v_residual =
      (static_cast<float>(v[token_idx * v_stride_t + hv_idx * v_stride_h +
                            value_idx * v_stride_v]) -
       state_dot_k) *
      beta_t;

  float out_val = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val =
        static_cast<float>(k[token_idx * k_stride_t + q_head_idx * k_stride_h +
                             k_idx * k_stride_k]) *
        k_norm;
    const int64_t state_offset =
        token_idx * state_stride_t + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const float new_state =
        static_cast<float>(state[state_offset]) * decay + v_residual * k_val;
    state[state_offset] = static_cast<state_t>(new_state);

    const float q_val =
        static_cast<float>(q[token_idx * q_stride_t + q_head_idx * q_stride_h +
                             k_idx * q_stride_k]) *
        q_norm * static_cast<float>(scale);
    out_val += new_state * q_val;
  }

  out[token_idx * out_stride_t + hv_idx * out_stride_h +
      value_idx * out_stride_v] = static_cast<scalar_t>(out_val);
}

template <typename scalar_t, typename state_t>
__global__ void fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kernel(
    const scalar_t* __restrict__ A_log, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ dt_bias,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices, scalar_t* __restrict__ out,
    int64_t tokens, int64_t heads, int64_t kv_heads, int64_t key_dim,
    int64_t value_dim, int64_t q_stride_t, int64_t q_stride_h,
    int64_t q_stride_k, int64_t k_stride_t, int64_t k_stride_h,
    int64_t k_stride_k, int64_t v_stride_t, int64_t v_stride_h,
    int64_t v_stride_v, int64_t a_stride_t, int64_t a_stride_h,
    int64_t b_stride_t, int64_t b_stride_h, int64_t state_stride_slot,
    int64_t state_stride_h, int64_t state_stride_v, int64_t state_stride_k,
    int64_t indices_stride, int64_t out_stride_t, int64_t out_stride_h,
    int64_t out_stride_v, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel) {
  const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = tokens * kv_heads * value_dim;
  if (linear_idx >= total) {
    return;
  }

  const int64_t value_idx = linear_idx % value_dim;
  const int64_t hv_idx = (linear_idx / value_dim) % kv_heads;
  const int64_t token_idx = linear_idx / (value_dim * kv_heads);
  const int64_t state_idx = state_indices[token_idx * indices_stride];
  if (state_idx < 0) {
    out[token_idx * out_stride_t + hv_idx * out_stride_h +
        value_idx * out_stride_v] = static_cast<scalar_t>(0.0f);
    return;
  }
  const int64_t head_ratio = kv_heads / heads;
  const int64_t q_head_idx = hv_idx / head_ratio;

  float q_norm = 0.0f;
  float k_norm = 0.0f;
  if (use_qk_l2norm_in_kernel) {
    for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
      const float q_val = static_cast<float>(
          q[token_idx * q_stride_t + q_head_idx * q_stride_h +
            k_idx * q_stride_k]);
      const float k_val = static_cast<float>(
          k[token_idx * k_stride_t + q_head_idx * k_stride_h +
            k_idx * k_stride_k]);
      q_norm += q_val * q_val;
      k_norm += k_val * k_val;
    }
    q_norm = rsqrtf(q_norm + 1e-6f);
    k_norm = rsqrtf(k_norm + 1e-6f);
  } else {
    q_norm = 1.0f;
    k_norm = 1.0f;
  }

  const float x = static_cast<float>(
                      a[token_idx * a_stride_t + hv_idx * a_stride_h]) +
                  static_cast<float>(dt_bias[hv_idx]);
  const float beta_x = static_cast<float>(beta) * x;
  const float softplus_x =
      beta_x <= static_cast<float>(threshold)
          ? static_cast<float>(1.0 / beta) * log1pf(expf(beta_x))
          : x;
  const float g =
      -expf(static_cast<float>(A_log[hv_idx])) * softplus_x;
  const float beta_t =
      1.0f / (1.0f + expf(-static_cast<float>(
                            b[token_idx * b_stride_t + hv_idx * b_stride_h])));
  const float decay = expf(g);

  float state_dot_k = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val =
        static_cast<float>(k[token_idx * k_stride_t + q_head_idx * k_stride_h +
                             k_idx * k_stride_k]) *
        k_norm;
    const int64_t state_offset =
        state_idx * state_stride_slot + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const float decayed_state = static_cast<float>(state[state_offset]) * decay;
    state_dot_k += decayed_state * k_val;
  }

  const float v_residual =
      (static_cast<float>(v[token_idx * v_stride_t + hv_idx * v_stride_h +
                            value_idx * v_stride_v]) -
       state_dot_k) *
      beta_t;

  float out_val = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val =
        static_cast<float>(k[token_idx * k_stride_t + q_head_idx * k_stride_h +
                             k_idx * k_stride_k]) *
        k_norm;
    const int64_t state_offset =
        state_idx * state_stride_slot + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const float new_state =
        static_cast<float>(state[state_offset]) * decay + v_residual * k_val;
    state[state_offset] = static_cast<state_t>(new_state);

    const float q_val =
        static_cast<float>(q[token_idx * q_stride_t + q_head_idx * q_stride_h +
                             k_idx * q_stride_k]) *
        q_norm * static_cast<float>(scale);
    out_val += new_state * q_val;
  }

  out[token_idx * out_stride_t + hv_idx * out_stride_h +
      value_idx * out_stride_v] = static_cast<scalar_t>(out_val);
}

template <typename scalar_t, typename state_t, typename accepted_t>
__global__ void fused_sigmoid_gating_delta_rule_gfx906_mtp_update_kernel(
    const scalar_t* __restrict__ A_log, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ dt_bias,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ cu_seqlens,
    const accepted_t* __restrict__ num_accepted_tokens,
    scalar_t* __restrict__ out, int64_t requests, int64_t max_query_len,
    int64_t heads, int64_t kv_heads, int64_t key_dim, int64_t value_dim,
    int64_t q_stride_t, int64_t q_stride_h, int64_t q_stride_k,
    int64_t k_stride_t, int64_t k_stride_h, int64_t k_stride_k,
    int64_t v_stride_t, int64_t v_stride_h, int64_t v_stride_v,
    int64_t a_stride_t, int64_t a_stride_h, int64_t b_stride_t,
    int64_t b_stride_h, int64_t state_stride_slot,
    int64_t state_stride_h, int64_t state_stride_v,
    int64_t state_stride_k, int64_t indices_stride_req,
    int64_t indices_stride_tok, int64_t out_stride_t,
    int64_t out_stride_h, int64_t out_stride_v, double beta,
    double threshold, double scale, bool use_qk_l2norm_in_kernel) {
  const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = requests * kv_heads * value_dim;
  if (linear_idx >= total) {
    return;
  }

  const int64_t value_idx = linear_idx % value_dim;
  const int64_t hv_idx = (linear_idx / value_dim) % kv_heads;
  const int64_t request_idx = linear_idx / (value_dim * kv_heads);
  const int64_t seq_start = cu_seqlens[request_idx];
  const int64_t seq_end = cu_seqlens[request_idx + 1];
  const int64_t seq_len = seq_end - seq_start;

  // Walk the whole speculative token chain inside one launch. The previous
  // per-token launcher issued max_query_len sequential kernel launches on the
  // same stream; each launch had to drain before the next could start (state
  // dependency), tripling the launch/drain cost and shrinking the active
  // grid. Here each thread processes its (request, head, value) column
  // through the chain; token t > 0 reads the state slot that token t - 1 of
  // the same thread just wrote, so the source column stays hot in L1.
  const int64_t head_ratio = kv_heads / heads;
  const int64_t q_head_idx = hv_idx / head_ratio;
  const int64_t initial_position =
      static_cast<int64_t>(num_accepted_tokens[request_idx]) - 1;

  for (int64_t token_position = 0; token_position < max_query_len;
       ++token_position) {
    if (token_position >= seq_len) {
      break;
    }

    const int64_t token_idx = seq_start + token_position;
    const int64_t source_position =
        token_position == 0 ? initial_position : token_position - 1;
    const int64_t source_state_idx =
        source_position >= 0
            ? state_indices[request_idx * indices_stride_req +
                            source_position * indices_stride_tok]
            : -1;
    const int64_t destination_state_idx =
        state_indices[request_idx * indices_stride_req +
                      token_position * indices_stride_tok];
    if (source_state_idx < 0 || destination_state_idx < 0) {
      out[token_idx * out_stride_t + hv_idx * out_stride_h +
          value_idx * out_stride_v] = static_cast<scalar_t>(0.0f);
      continue;
    }

    float q_norm = 1.0f;
    float k_norm = 1.0f;
  if (use_qk_l2norm_in_kernel) {
    float q_norm_sq = 0.0f;
    float k_norm_sq = 0.0f;
    for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
      const float q_val = static_cast<float>(
          q[token_idx * q_stride_t + q_head_idx * q_stride_h +
            k_idx * q_stride_k]);
      const float k_val = static_cast<float>(
          k[token_idx * k_stride_t + q_head_idx * k_stride_h +
            k_idx * k_stride_k]);
      q_norm_sq += q_val * q_val;
      k_norm_sq += k_val * k_val;
    }
    q_norm = rsqrtf(q_norm_sq + 1e-6f);
    k_norm = rsqrtf(k_norm_sq + 1e-6f);
  }

  const float x = static_cast<float>(
                      a[token_idx * a_stride_t + hv_idx * a_stride_h]) +
                  static_cast<float>(dt_bias[hv_idx]);
  const float beta_x = static_cast<float>(beta) * x;
  const float softplus_x =
      beta_x <= static_cast<float>(threshold)
          ? static_cast<float>(1.0 / beta) * log1pf(expf(beta_x))
          : x;
  const float decay =
      expf(-expf(static_cast<float>(A_log[hv_idx])) * softplus_x);
  const float beta_t =
      1.0f / (1.0f + expf(-static_cast<float>(
                            b[token_idx * b_stride_t + hv_idx * b_stride_h])));

  float state_dot_k = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val =
        static_cast<float>(k[token_idx * k_stride_t +
                             q_head_idx * k_stride_h + k_idx * k_stride_k]) *
        k_norm;
    const int64_t source_offset =
        source_state_idx * state_stride_slot + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    state_dot_k +=
        static_cast<float>(state[source_offset]) * decay * k_val;
  }

  const float v_residual =
      (static_cast<float>(v[token_idx * v_stride_t + hv_idx * v_stride_h +
                            value_idx * v_stride_v]) -
       state_dot_k) *
      beta_t;
  float out_val = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val =
        static_cast<float>(k[token_idx * k_stride_t +
                             q_head_idx * k_stride_h + k_idx * k_stride_k]) *
        k_norm;
    const int64_t source_offset =
        source_state_idx * state_stride_slot + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const int64_t destination_offset =
        destination_state_idx * state_stride_slot + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const float new_state =
        static_cast<float>(state[source_offset]) * decay + v_residual * k_val;
    state[destination_offset] = static_cast<state_t>(new_state);
    const float q_val =
        static_cast<float>(q[token_idx * q_stride_t +
                             q_head_idx * q_stride_h + k_idx * q_stride_k]) *
        q_norm * static_cast<float>(scale);
    out_val += new_state * q_val;
  }
    out[token_idx * out_stride_t + hv_idx * out_stride_h +
        value_idx * out_stride_v] = static_cast<scalar_t>(out_val);
  }
}

// Warp-cooperative MTP update for gfx906, targeting the transposed state
// pool layout the server actually uses: state[slot][head][v][k] with v stride
// 1 and k stride value_dim, i.e. physically [slot][head][k][v] where v is
// contiguous. Decomposition (adapted from the mainline fused_gdn_decode
// kernel): grid = (kv_heads, requests), 256 threads = 4 warps x 64 lanes.
// Warps take turns preparing each token (q/k l2norm via full-wave reduce, v,
// decay/beta) into shared memory; the token chain then advances in registers
// with keys chunked 32 at a time (8 k rows per warp, each lane holding two
// contiguous v elements) while the next chunk is prefetched. The per-value
// dot products sum each thread's k rows locally and exchange one partial per
// lane through shared memory, so the whole step is shuffle-free and every
// state load/store is a coalesced float2. key_dim and value_dim are fixed at
// 128 by the launcher guard.
struct GdnMtpV2Strides {
  int64_t q_t;
  int64_t q_h;
  int64_t k_t;
  int64_t k_h;
  int64_t v_t;
  int64_t v_h;
  int64_t a_t;
  int64_t a_h;
  int64_t b_t;
  int64_t b_h;
  int64_t state_slot;
  int64_t state_h;
  int64_t out_t;
  int64_t out_h;
  int64_t si_req;
  int64_t si_tok;
};

__device__ __forceinline__ float2 gdn_mtp_v2_reduce_pair(float x, float y) {
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffffffffffull, x, offset);
    y += __shfl_xor_sync(0xffffffffffffffffull, y, offset);
  }
  return make_float2(x, y);
}

// Loads/stores 8 k rows of one 32-key chunk. In the transposed pool the
// (k, v) element lives at k * value_dim + v, so a lane's v pair is a
// contiguous float2.
template <typename state_t>
__device__ __forceinline__ void gdn_mtp_v2_load_kchunk(
    float (&h)[8][2], const state_t* __restrict__ state_head_base, int pass,
    int warp, int lane) {
#pragma unroll
  for (int row = 0; row < 8; ++row) {
    const state_t* src =
        state_head_base + (pass * 32 + warp + row * 4) * 128 + lane * 2;
    if constexpr (std::is_same<state_t, float>::value) {
      const float2 packed = *reinterpret_cast<const float2*>(src);
      h[row][0] = packed.x;
      h[row][1] = packed.y;
    } else {
      h[row][0] = static_cast<float>(src[0]);
      h[row][1] = static_cast<float>(src[1]);
    }
  }
}

template <typename state_t>
__device__ __forceinline__ void gdn_mtp_v2_store_kchunk(
    const float (&h)[8][2], state_t* __restrict__ state_head_base, int pass,
    int warp, int lane) {
#pragma unroll
  for (int row = 0; row < 8; ++row) {
    state_t* dst =
        state_head_base + (pass * 32 + warp + row * 4) * 128 + lane * 2;
    if constexpr (std::is_same<state_t, float>::value) {
      *reinterpret_cast<float2*>(dst) = make_float2(h[row][0], h[row][1]);
    } else {
      dst[0] = static_cast<state_t>(h[row][0]);
      dst[1] = static_cast<state_t>(h[row][1]);
    }
  }
}

template <typename scalar_t, typename state_t, typename accepted_t>
__global__ __launch_bounds__(256, 2)
void fused_sigmoid_gating_delta_rule_gfx906_mtp_v2_kernel(
    const scalar_t* __restrict__ A_log, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ dt_bias,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ cu_seqlens,
    const accepted_t* __restrict__ num_accepted_tokens,
    scalar_t* __restrict__ out, int64_t max_query_len, int64_t heads,
    double beta, double threshold, double scale, bool use_qk_l2norm_in_kernel,
    bool use_tiled_qk_head_mapping, GdnMtpV2Strides s) {
  constexpr int kThreads = 256;
  constexpr int kWaveSize = 64;  // gfx906 GCN wave64
  constexpr int kWarps = kThreads / kWaveSize;
  constexpr int kRowsPerWarp = 32 / kWarps;  // 8 k rows per warp per pass
  constexpr int kPasses = 128 / 32;          // key_dim / keys-per-pass
  constexpr int kMaxTokens = 4;              // launcher caps max_query_len
  constexpr int kDimV = 128;

  const int hv = blockIdx.x;
  const int request = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWaveSize;
  const int warp = tid / kWaveSize;
  const int head_ratio = static_cast<int>(gridDim.x / heads);
  const int q_head = use_tiled_qk_head_mapping
                         ? (hv % static_cast<int>(heads))
                         : (hv / head_ratio);

  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }

  const int initial_position =
      static_cast<int>(num_accepted_tokens[request]) - 1;
  const int source_slot =
      initial_position >= 0 &&
              initial_position < static_cast<int>(max_query_len)
          ? state_indices[request * s.si_req + initial_position * s.si_tok]
          : -1;
  if (source_slot < 0 || num_tokens > kMaxTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int t = linear >> 7;
      const int value = linear & (kDimV - 1);
      out[(bos + t) * s.out_t + hv * s.out_h + value] =
          static_cast<scalar_t>(0.0f);
    }
    return;
  }

  __shared__ float smem_q[kMaxTokens][128];
  __shared__ float smem_k[kMaxTokens][128];
  __shared__ float smem_v[kMaxTokens][128];
  __shared__ float smem_decay[kMaxTokens];
  __shared__ float smem_beta[kMaxTokens];
  __shared__ float smem_partial[kWarps][128];

  for (int t = warp; t < num_tokens; t += kWarps) {
    const int64_t token = bos + t;
    const scalar_t* q_src = q + token * s.q_t + q_head * s.q_h + lane * 2;
    const scalar_t* k_src = k + token * s.k_t + q_head * s.k_h + lane * 2;
    const float q0 = static_cast<float>(q_src[0]);
    const float q1 = static_cast<float>(q_src[1]);
    const float k0 = static_cast<float>(k_src[0]);
    const float k1 = static_cast<float>(k_src[1]);
    const float2 qk = gdn_mtp_v2_reduce_pair(q0 * q0 + q1 * q1, k0 * k0 + k1 * k1);
    float q_scale = static_cast<float>(scale);
    float k_scale = 1.0f;
    if (use_qk_l2norm_in_kernel) {
      q_scale = rsqrtf(qk.x + 1e-6f) * static_cast<float>(scale);
      k_scale = rsqrtf(qk.y + 1e-6f);
    }
    *reinterpret_cast<float2*>(&smem_q[t][lane * 2]) =
        make_float2(q0 * q_scale, q1 * q_scale);
    *reinterpret_cast<float2*>(&smem_k[t][lane * 2]) =
        make_float2(k0 * k_scale, k1 * k_scale);
    const scalar_t* v_src = v + token * s.v_t + hv * s.v_h + lane * 2;
    *reinterpret_cast<float2*>(&smem_v[t][lane * 2]) =
        make_float2(static_cast<float>(v_src[0]),
                    static_cast<float>(v_src[1]));
    if (lane == 0) {
      const float x = static_cast<float>(a[token * s.a_t + hv * s.a_h]) +
                      static_cast<float>(dt_bias[hv]);
      const float beta_x = static_cast<float>(beta) * x;
      const float softplus_x =
          beta_x <= static_cast<float>(threshold)
              ? static_cast<float>(1.0 / beta) * log1pf(expf(beta_x))
              : x;
      smem_decay[t] =
          expf(-expf(static_cast<float>(A_log[hv])) * softplus_x);
      smem_beta[t] = 1.0f /
                     (1.0f + expf(-static_cast<float>(
                                  b[token * s.b_t + hv * s.b_h])));
    }
  }
  __syncthreads();

  const state_t* source_state = state +
      static_cast<int64_t>(source_slot) * s.state_slot + hv * s.state_h;
  // The full key range is split across warps (8 k rows each), so each warp
  // keeps its share of ALL four 32-key chunks in registers and every dot is
  // reduced across warps BEFORE the delta update: per token we do one full-k
  // dot exchange, then update/store. (A per-chunk partial dot would be
  // mathematically wrong.)
  float h_all[kPasses][kRowsPerWarp][2];
#pragma unroll
  for (int pass = 0; pass < kPasses; ++pass) {
    gdn_mtp_v2_load_kchunk(h_all[pass], source_state, pass, warp, lane);
  }
#pragma unroll
  for (int t = 0; t < kMaxTokens; ++t) {
    if (t >= num_tokens) {
      continue;
    }
    const float decay = smem_decay[t];
    const float beta_t = smem_beta[t];
    const int64_t token = bos + t;
    const int32_t dst_slot =
        state_indices[request * s.si_req + t * s.si_tok];

    // Full-k dot for lane's two values: apply decay and sum this warp's
    // share over all passes, then exchange one partial per lane.
    float warp_dot[2] = {0.0f, 0.0f};
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int k_idx = pass * 32 + warp + row * kWarps;
        const float k_val = smem_k[t][k_idx];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          h_all[pass][row][i] *= decay;
          warp_dot[i] += h_all[pass][row][i] * k_val;
        }
      }
    }
    __syncthreads();
    smem_partial[warp][lane * 2] = warp_dot[0];
    smem_partial[warp][lane * 2 + 1] = warp_dot[1];
    __syncthreads();
    float dot[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float acc = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        acc += smem_partial[w][lane * 2 + i];
      }
      dot[i] = acc;
    }

    const float2 vin =
        *reinterpret_cast<const float2*>(&smem_v[t][lane * 2]);
    float delta[2];
    delta[0] = (vin.x - dot[0]) * beta_t;
    delta[1] = (vin.y - dot[1]) * beta_t;

    // Update this warp's k rows and accumulate the output partials.
    float warp_out[2] = {0.0f, 0.0f};
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int k_idx = pass * 32 + warp + row * kWarps;
        const float k_val = smem_k[t][k_idx];
        const float q_val = smem_q[t][k_idx];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          h_all[pass][row][i] += k_val * delta[i];
          warp_out[i] += h_all[pass][row][i] * q_val;
        }
      }
    }

    if (dst_slot >= 0) {
      state_t* dst_state = state +
          static_cast<int64_t>(dst_slot) * s.state_slot + hv * s.state_h;
#pragma unroll
      for (int pass = 0; pass < kPasses; ++pass) {
        gdn_mtp_v2_store_kchunk(h_all[pass], dst_state, pass, warp, lane);
      }
    }

    // Cross-warp output reduction for lane's two values.
    __syncthreads();
    smem_partial[warp][lane * 2] = warp_out[0];
    smem_partial[warp][lane * 2 + 1] = warp_out[1];
    __syncthreads();
    float o0 = 0.0f;
    float o1 = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      o0 += smem_partial[w][lane * 2];
      o1 += smem_partial[w][lane * 2 + 1];
    }
    out[token * s.out_t + hv * s.out_h + lane * 2] = static_cast<scalar_t>(
        dst_slot >= 0 ? o0 : 0.0f);
    out[token * s.out_t + hv * s.out_h + lane * 2 + 1] =
        static_cast<scalar_t>(dst_slot >= 0 ? o1 : 0.0f);
    __syncthreads();
  }
}

template <typename scalar_t, typename state_t, typename index_t>
__global__ void fused_sigmoid_gating_delta_rule_gfx906_prefill_kernel(
    const scalar_t* __restrict__ A_log, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ dt_bias,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, state_t* __restrict__ state,
    const index_t* __restrict__ cu_seqlens, scalar_t* __restrict__ out,
    int64_t num_seqs, int64_t heads, int64_t kv_heads, int64_t key_dim,
    int64_t value_dim, int64_t q_stride_t, int64_t q_stride_h,
    int64_t q_stride_k, int64_t k_stride_t, int64_t k_stride_h,
    int64_t k_stride_k, int64_t v_stride_t, int64_t v_stride_h,
    int64_t v_stride_v, int64_t a_stride_t, int64_t a_stride_h,
    int64_t b_stride_t, int64_t b_stride_h, int64_t state_stride_slot,
    int64_t state_stride_h, int64_t state_stride_v, int64_t state_stride_k,
    int64_t out_stride_t, int64_t out_stride_h, int64_t out_stride_v,
    double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel) {
  const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = num_seqs * kv_heads * value_dim;
  if (linear_idx >= total) {
    return;
  }

  const int64_t value_idx = linear_idx % value_dim;
  const int64_t hv_idx = (linear_idx / value_dim) % kv_heads;
  const int64_t seq_idx = linear_idx / (value_dim * kv_heads);
  const int64_t token_start = static_cast<int64_t>(cu_seqlens[seq_idx]);
  const int64_t token_end = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
  if (token_end <= token_start) {
    return;
  }

  const int64_t head_ratio = kv_heads / heads;
  const int64_t q_head_idx = hv_idx / head_ratio;

  for (int64_t token_idx = token_start; token_idx < token_end; ++token_idx) {
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    if (use_qk_l2norm_in_kernel) {
      for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
        const float q_val = static_cast<float>(
            q[token_idx * q_stride_t + q_head_idx * q_stride_h +
              k_idx * q_stride_k]);
        const float k_val = static_cast<float>(
            k[token_idx * k_stride_t + q_head_idx * k_stride_h +
              k_idx * k_stride_k]);
        q_norm += q_val * q_val;
        k_norm += k_val * k_val;
      }
      q_norm = rsqrtf(q_norm + 1e-6f);
      k_norm = rsqrtf(k_norm + 1e-6f);
    } else {
      q_norm = 1.0f;
      k_norm = 1.0f;
    }

    const float x = static_cast<float>(
                        a[token_idx * a_stride_t + hv_idx * a_stride_h]) +
                    static_cast<float>(dt_bias[hv_idx]);
    const float beta_x = static_cast<float>(beta) * x;
    const float softplus_x =
        beta_x <= static_cast<float>(threshold)
            ? static_cast<float>(1.0 / beta) * log1pf(expf(beta_x))
            : x;
    const float g =
        -expf(static_cast<float>(A_log[hv_idx])) * softplus_x;
    const float beta_t =
        1.0f / (1.0f + expf(-static_cast<float>(
                              b[token_idx * b_stride_t + hv_idx * b_stride_h])));
    const float decay = expf(g);

    float state_dot_k = 0.0f;
    for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
      const float k_val =
          static_cast<float>(k[token_idx * k_stride_t + q_head_idx * k_stride_h +
                               k_idx * k_stride_k]) *
          k_norm;
      const int64_t state_offset =
          seq_idx * state_stride_slot + hv_idx * state_stride_h +
          value_idx * state_stride_v + k_idx * state_stride_k;
      const float decayed_state = static_cast<float>(state[state_offset]) * decay;
      state_dot_k += decayed_state * k_val;
    }

    const float v_residual =
        (static_cast<float>(v[token_idx * v_stride_t + hv_idx * v_stride_h +
                              value_idx * v_stride_v]) -
         state_dot_k) *
        beta_t;

    float out_val = 0.0f;
    for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
      const float k_val =
          static_cast<float>(k[token_idx * k_stride_t + q_head_idx * k_stride_h +
                               k_idx * k_stride_k]) *
          k_norm;
      const int64_t state_offset =
          seq_idx * state_stride_slot + hv_idx * state_stride_h +
          value_idx * state_stride_v + k_idx * state_stride_k;
      const float new_state =
          static_cast<float>(state[state_offset]) * decay + v_residual * k_val;
      state[state_offset] = static_cast<state_t>(new_state);

      const float q_val =
          static_cast<float>(q[token_idx * q_stride_t + q_head_idx * q_stride_h +
                               k_idx * q_stride_k]) *
          q_norm * static_cast<float>(scale);
      out_val += new_state * q_val;
    }

    out[token_idx * out_stride_t + hv_idx * out_stride_h +
        value_idx * out_stride_v] = static_cast<scalar_t>(out_val);
  }
}

template <typename scalar_t, typename state_t, typename index_t>
__global__ void fused_sigmoid_gating_delta_rule_gfx906_prefill_parallel_kernel(
    const scalar_t* __restrict__ A_log, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ dt_bias,
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, state_t* __restrict__ state,
    const index_t* __restrict__ cu_seqlens, scalar_t* __restrict__ out,
    int64_t num_seqs, int64_t heads, int64_t kv_heads, int64_t key_dim,
    int64_t value_dim, int64_t q_stride_t, int64_t q_stride_h,
    int64_t q_stride_k, int64_t k_stride_t, int64_t k_stride_h,
    int64_t k_stride_k, int64_t v_stride_t, int64_t v_stride_h,
    int64_t v_stride_v, int64_t a_stride_t, int64_t a_stride_h,
    int64_t b_stride_t, int64_t b_stride_h, int64_t state_stride_slot,
    int64_t state_stride_h, int64_t state_stride_v, int64_t state_stride_k,
    int64_t out_stride_t, int64_t out_stride_h, int64_t out_stride_v,
    double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel) {
  const int64_t linear_idx = blockIdx.x;
  const int64_t total = num_seqs * kv_heads * value_dim;
  if (linear_idx >= total) {
    return;
  }

  extern __shared__ float smem[];
  __shared__ float q_norm_shared;
  __shared__ float k_norm_shared;
  __shared__ float v_residual_shared;

  const int64_t value_idx = linear_idx % value_dim;
  const int64_t hv_idx = (linear_idx / value_dim) % kv_heads;
  const int64_t seq_idx = linear_idx / (value_dim * kv_heads);
  const int64_t token_start = static_cast<int64_t>(cu_seqlens[seq_idx]);
  const int64_t token_end = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
  if (token_end <= token_start) {
    return;
  }

  const int64_t head_ratio = kv_heads / heads;
  const int64_t q_head_idx = hv_idx / head_ratio;
  const float scale_f = static_cast<float>(scale);
  const float beta_f = static_cast<float>(beta);
  const float threshold_f = static_cast<float>(threshold);

  for (int64_t token_idx = token_start; token_idx < token_end; ++token_idx) {
    float q_norm = 1.0f;
    float k_norm = 1.0f;

    if (use_qk_l2norm_in_kernel) {
      float q_sum = 0.0f;
      float k_sum = 0.0f;
      for (int64_t k_idx = threadIdx.x; k_idx < key_dim;
           k_idx += blockDim.x) {
        const float q_val = static_cast<float>(
            q[token_idx * q_stride_t + q_head_idx * q_stride_h +
              k_idx * q_stride_k]);
        const float k_val = static_cast<float>(
            k[token_idx * k_stride_t + q_head_idx * k_stride_h +
              k_idx * k_stride_k]);
        q_sum += q_val * q_val;
        k_sum += k_val * k_val;
      }

      smem[threadIdx.x] = q_sum;
      __syncthreads();
      for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        q_norm_shared = rsqrtf(smem[0] + 1e-6f);
      }

      smem[threadIdx.x] = k_sum;
      __syncthreads();
      for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        k_norm_shared = rsqrtf(smem[0] + 1e-6f);
      }
      __syncthreads();
      q_norm = q_norm_shared;
      k_norm = k_norm_shared;
    }

    float decay = 1.0f;
    float beta_t = 1.0f;
    if (threadIdx.x == 0) {
      const float x = static_cast<float>(
                          a[token_idx * a_stride_t + hv_idx * a_stride_h]) +
                      static_cast<float>(dt_bias[hv_idx]);
      const float beta_x = beta_f * x;
      const float softplus_x =
          beta_x <= threshold_f ? (1.0f / beta_f) * log1pf(expf(beta_x)) : x;
      const float g = -expf(static_cast<float>(A_log[hv_idx])) * softplus_x;
      decay = expf(g);
      beta_t =
          1.0f /
          (1.0f + expf(-static_cast<float>(
                      b[token_idx * b_stride_t + hv_idx * b_stride_h])));
      q_norm_shared = decay;
      k_norm_shared = beta_t;
    }
    __syncthreads();
    decay = q_norm_shared;
    beta_t = k_norm_shared;

    float state_dot_k = 0.0f;
    for (int64_t k_idx = threadIdx.x; k_idx < key_dim;
         k_idx += blockDim.x) {
      const float k_val =
          static_cast<float>(k[token_idx * k_stride_t + q_head_idx * k_stride_h +
                               k_idx * k_stride_k]) *
          k_norm;
      const int64_t state_offset =
          seq_idx * state_stride_slot + hv_idx * state_stride_h +
          value_idx * state_stride_v + k_idx * state_stride_k;
      state_dot_k += static_cast<float>(state[state_offset]) * decay * k_val;
    }

    smem[threadIdx.x] = state_dot_k;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        smem[threadIdx.x] += smem[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      v_residual_shared =
          (static_cast<float>(v[token_idx * v_stride_t + hv_idx * v_stride_h +
                                value_idx * v_stride_v]) -
           smem[0]) *
          beta_t;
    }
    __syncthreads();
    const float v_residual = v_residual_shared;

    float out_val = 0.0f;
    for (int64_t k_idx = threadIdx.x; k_idx < key_dim;
         k_idx += blockDim.x) {
      const float k_val =
          static_cast<float>(k[token_idx * k_stride_t + q_head_idx * k_stride_h +
                               k_idx * k_stride_k]) *
          k_norm;
      const int64_t state_offset =
          seq_idx * state_stride_slot + hv_idx * state_stride_h +
          value_idx * state_stride_v + k_idx * state_stride_k;
      const float new_state =
          static_cast<float>(state[state_offset]) * decay + v_residual * k_val;
      state[state_offset] = static_cast<state_t>(new_state);

      const float q_val =
          static_cast<float>(q[token_idx * q_stride_t + q_head_idx * q_stride_h +
                               k_idx * q_stride_k]) *
          q_norm * scale_f;
      out_val += new_state * q_val;
    }

    smem[threadIdx.x] = out_val;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        smem[threadIdx.x] += smem[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      out[token_idx * out_stride_t + hv_idx * out_stride_h +
          value_idx * out_stride_v] = static_cast<scalar_t>(smem[0]);
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename state_t>
__global__ void fused_recurrent_gated_delta_rule_gfx906_packed_decode_kernel(
    const scalar_t* __restrict__ mixed_qkv, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ A_log,
    const scalar_t* __restrict__ dt_bias, state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices, scalar_t* __restrict__ out,
    int64_t tokens, int64_t heads, int64_t kv_heads, int64_t key_dim,
    int64_t value_dim, int64_t mixed_stride_t, int64_t mixed_stride_d,
    int64_t a_stride_t, int64_t a_stride_h, int64_t b_stride_t,
    int64_t b_stride_h, int64_t state_stride_slot, int64_t state_stride_h,
    int64_t state_stride_v, int64_t state_stride_k, int64_t indices_stride,
    int64_t out_stride_t, int64_t out_stride_one, int64_t out_stride_h,
    int64_t out_stride_v, double scale, bool use_qk_l2norm_in_kernel,
    bool use_tiled_qk_head_mapping) {
  const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = tokens * kv_heads * value_dim;
  if (linear_idx >= total) {
    return;
  }

  const int64_t value_idx = linear_idx % value_dim;
  const int64_t hv_idx = (linear_idx / value_dim) % kv_heads;
  const int64_t token_idx = linear_idx / (value_dim * kv_heads);
  const int64_t state_idx = state_indices[token_idx * indices_stride];
  constexpr int shared_norm_width = 128;
  const bool use_shared_qk_norm =
      use_qk_l2norm_in_kernel && key_dim == shared_norm_width &&
      value_dim == shared_norm_width &&
      (blockDim.x == shared_norm_width ||
       blockDim.x == 2 * shared_norm_width) &&
      total % blockDim.x == 0;
  if (state_idx < 0 && !use_shared_qk_norm) {
    out[token_idx * out_stride_t + hv_idx * out_stride_h +
        value_idx * out_stride_v] = static_cast<scalar_t>(0.0f);
    return;
  }

  const int64_t head_ratio = kv_heads / heads;
  const int64_t q_head_idx =
      use_tiled_qk_head_mapping ? (hv_idx % heads) : (hv_idx / head_ratio);
  const int64_t q_offset = q_head_idx * key_dim;
  const int64_t k_offset = heads * key_dim + q_head_idx * key_dim;
  const int64_t v_offset = 2 * heads * key_dim + hv_idx * value_dim;

  float q_norm = 0.0f;
  float k_norm = 0.0f;
  __shared__ float q_norm_sums[2 * shared_norm_width];
  __shared__ float k_norm_sums[2 * shared_norm_width];
  __shared__ float q_shared[2 * shared_norm_width];
  __shared__ float k_shared[2 * shared_norm_width];
  __shared__ float q_norm_scales[2];
  __shared__ float k_norm_scales[2];
  __shared__ float decay_scales[2];
  __shared__ float beta_scales[2];
  if (use_qk_l2norm_in_kernel) {
    if (use_shared_qk_norm) {
      const int64_t group_idx = threadIdx.x / shared_norm_width;
      const int64_t group_lane = threadIdx.x - group_idx * shared_norm_width;
      const int64_t group_linear_idx =
          blockIdx.x * blockDim.x + group_idx * shared_norm_width;
      const int64_t group_hv_idx =
          (group_linear_idx / value_dim) % kv_heads;
      const int64_t group_token_idx =
          group_linear_idx / (value_dim * kv_heads);
      const int64_t group_q_head_idx =
          use_tiled_qk_head_mapping ? (group_hv_idx % heads)
                                    : (group_hv_idx / head_ratio);
      const int64_t group_q_offset = group_q_head_idx * key_dim;
      const int64_t group_k_offset =
          heads * key_dim + group_q_head_idx * key_dim;

      const float q_val = static_cast<float>(
          mixed_qkv[group_token_idx * mixed_stride_t +
                    (group_q_offset + group_lane) * mixed_stride_d]);
      const float k_val = static_cast<float>(
          mixed_qkv[group_token_idx * mixed_stride_t +
                    (group_k_offset + group_lane) * mixed_stride_d]);
      const int64_t shared_offset = group_idx * shared_norm_width + group_lane;
      q_shared[shared_offset] = q_val;
      k_shared[shared_offset] = k_val;
      q_norm_sums[shared_offset] = q_val * q_val;
      k_norm_sums[shared_offset] = k_val * k_val;
      __syncthreads();

      for (int stride = shared_norm_width / 2; stride > 0; stride >>= 1) {
        if (group_lane < stride) {
          q_norm_sums[shared_offset] += q_norm_sums[shared_offset + stride];
          k_norm_sums[shared_offset] += k_norm_sums[shared_offset + stride];
        }
        __syncthreads();
      }

      if (group_lane == 0) {
        q_norm_scales[group_idx] =
            rsqrtf(q_norm_sums[group_idx * shared_norm_width] + 1e-6f);
        k_norm_scales[group_idx] =
            rsqrtf(k_norm_sums[group_idx * shared_norm_width] + 1e-6f);
      }
      __syncthreads();

      q_norm = q_norm_scales[group_idx];
      k_norm = k_norm_scales[group_idx];
      q_shared[shared_offset] = q_val * q_norm * static_cast<float>(scale);
      k_shared[shared_offset] = k_val * k_norm;
      __syncthreads();
    } else {
      for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
        const float q_val = static_cast<float>(
            mixed_qkv[token_idx * mixed_stride_t +
                      (q_offset + k_idx) * mixed_stride_d]);
        const float k_val = static_cast<float>(
            mixed_qkv[token_idx * mixed_stride_t +
                      (k_offset + k_idx) * mixed_stride_d]);
        q_norm += q_val * q_val;
        k_norm += k_val * k_val;
      }
      q_norm = rsqrtf(q_norm + 1e-6f);
      k_norm = rsqrtf(k_norm + 1e-6f);
    }
  } else {
    q_norm = 1.0f;
    k_norm = 1.0f;
  }

  if (state_idx < 0) {
    out[token_idx * out_stride_t + hv_idx * out_stride_h +
        value_idx * out_stride_v] = static_cast<scalar_t>(0.0f);
    return;
  }

  float beta_t;
  float decay;
  if (use_shared_qk_norm) {
    const int64_t group_idx = threadIdx.x / shared_norm_width;
    const int64_t group_lane = threadIdx.x - group_idx * shared_norm_width;
    if (group_lane == 0) {
      const float x = static_cast<float>(
                          a[token_idx * a_stride_t + hv_idx * a_stride_h]) +
                      static_cast<float>(dt_bias[hv_idx]);
      const float softplus_x = x <= 20.0f ? log1pf(expf(x)) : x;
      const float g =
          -expf(static_cast<float>(A_log[hv_idx])) * softplus_x;
      beta_scales[group_idx] =
          1.0f /
          (1.0f + expf(-static_cast<float>(
                      b[token_idx * b_stride_t + hv_idx * b_stride_h])));
      decay_scales[group_idx] = expf(g);
    }
    __syncthreads();
    beta_t = beta_scales[group_idx];
    decay = decay_scales[group_idx];
  } else {
    const float x = static_cast<float>(
                        a[token_idx * a_stride_t + hv_idx * a_stride_h]) +
                    static_cast<float>(dt_bias[hv_idx]);
    const float softplus_x = x <= 20.0f ? log1pf(expf(x)) : x;
    const float g =
        -expf(static_cast<float>(A_log[hv_idx])) * softplus_x;
    beta_t =
        1.0f / (1.0f + expf(-static_cast<float>(
                              b[token_idx * b_stride_t + hv_idx * b_stride_h])));
    decay = expf(g);
  }

  float state_dot_k = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val =
        use_shared_qk_norm
            ? k_shared[(threadIdx.x / shared_norm_width) * shared_norm_width +
                       k_idx]
            : static_cast<float>(
                  mixed_qkv[token_idx * mixed_stride_t +
                            (k_offset + k_idx) * mixed_stride_d]) *
                  k_norm;
    const int64_t state_offset =
        state_idx * state_stride_slot + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const float decayed_state = static_cast<float>(state[state_offset]) * decay;
    state_dot_k += decayed_state * k_val;
  }

  const float v_val = static_cast<float>(
      mixed_qkv[token_idx * mixed_stride_t +
                (v_offset + value_idx) * mixed_stride_d]);
  const float v_residual = (v_val - state_dot_k) * beta_t;

  float out_val = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val =
        use_shared_qk_norm
            ? k_shared[(threadIdx.x / shared_norm_width) * shared_norm_width +
                       k_idx]
            : static_cast<float>(
                  mixed_qkv[token_idx * mixed_stride_t +
                            (k_offset + k_idx) * mixed_stride_d]) *
                  k_norm;
    const int64_t state_offset =
        state_idx * state_stride_slot + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const float new_state =
        static_cast<float>(state[state_offset]) * decay + v_residual * k_val;
    state[state_offset] = static_cast<state_t>(new_state);

    const float q_val =
        use_shared_qk_norm
            ? q_shared[(threadIdx.x / shared_norm_width) * shared_norm_width +
                       k_idx]
            : static_cast<float>(
                  mixed_qkv[token_idx * mixed_stride_t +
                            (q_offset + k_idx) * mixed_stride_d]) *
                  q_norm * static_cast<float>(scale);
    out_val += new_state * q_val;
  }

  out[token_idx * out_stride_t + 0 * out_stride_one + hv_idx * out_stride_h +
      value_idx * out_stride_v] = static_cast<scalar_t>(out_val);
}

template <typename scalar_t, typename state_t>
__global__ void
fused_recurrent_gated_delta_rule_gfx906_packed_decode_tiled128_kernel(
    const scalar_t* __restrict__ mixed_qkv, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ A_log,
    const scalar_t* __restrict__ dt_bias, state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices, scalar_t* __restrict__ out,
    int64_t tokens, int64_t heads, int64_t kv_heads, int64_t mixed_stride_t,
    int64_t mixed_stride_d, int64_t a_stride_t, int64_t a_stride_h,
    int64_t b_stride_t, int64_t b_stride_h, int64_t state_stride_slot,
    int64_t state_stride_h, int64_t state_stride_v, int64_t state_stride_k,
    int64_t indices_stride, int64_t out_stride_t, int64_t out_stride_one,
    int64_t out_stride_h, int64_t out_stride_v, double scale) {
  constexpr int width = 128;
  const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = tokens * kv_heads * width;
  if (linear_idx >= total) {
    return;
  }

  const int64_t value_idx = linear_idx % width;
  const int64_t hv_idx = (linear_idx / width) % kv_heads;
  const int64_t token_idx = linear_idx / (width * kv_heads);
  const int64_t state_idx = state_indices[token_idx * indices_stride];
  const int64_t v_offset = 2 * heads * width + hv_idx * width;

  const int64_t group_idx = threadIdx.x / width;
  const int64_t group_lane = threadIdx.x - group_idx * width;
  const int64_t group_linear_idx = blockIdx.x * blockDim.x + group_idx * width;
  const int64_t group_hv_idx = (group_linear_idx / width) % kv_heads;
  const int64_t group_token_idx = group_linear_idx / (width * kv_heads);
  const int64_t group_q_head_idx = group_hv_idx % heads;
  const int64_t group_q_offset = group_q_head_idx * width;
  const int64_t group_k_offset = heads * width + group_q_head_idx * width;
  const int64_t shared_offset = group_idx * width + group_lane;

  __shared__ float q_norm_sums[2 * width];
  __shared__ float k_norm_sums[2 * width];
  __shared__ float q_shared[2 * width];
  __shared__ float k_shared[2 * width];
  __shared__ float q_norm_scales[2];
  __shared__ float k_norm_scales[2];
  __shared__ float decay_scales[2];
  __shared__ float beta_scales[2];

  const float q_val = static_cast<float>(
      mixed_qkv[group_token_idx * mixed_stride_t +
                (group_q_offset + group_lane) * mixed_stride_d]);
  const float k_val = static_cast<float>(
      mixed_qkv[group_token_idx * mixed_stride_t +
                (group_k_offset + group_lane) * mixed_stride_d]);
  q_shared[shared_offset] = q_val;
  k_shared[shared_offset] = k_val;
  q_norm_sums[shared_offset] = q_val * q_val;
  k_norm_sums[shared_offset] = k_val * k_val;
  __syncthreads();

  for (int stride = width / 2; stride > 0; stride >>= 1) {
    if (group_lane < stride) {
      q_norm_sums[shared_offset] += q_norm_sums[shared_offset + stride];
      k_norm_sums[shared_offset] += k_norm_sums[shared_offset + stride];
    }
    __syncthreads();
  }

  if (group_lane == 0) {
    q_norm_scales[group_idx] =
        rsqrtf(q_norm_sums[group_idx * width] + 1e-6f);
    k_norm_scales[group_idx] =
        rsqrtf(k_norm_sums[group_idx * width] + 1e-6f);
  }
  __syncthreads();

  q_shared[shared_offset] =
      q_val * q_norm_scales[group_idx] * static_cast<float>(scale);
  k_shared[shared_offset] = k_val * k_norm_scales[group_idx];

  const bool valid_state = state_idx >= 0;

  if (group_lane == 0) {
    const float x = static_cast<float>(
                        a[token_idx * a_stride_t + hv_idx * a_stride_h]) +
                    static_cast<float>(dt_bias[hv_idx]);
    const float softplus_x = x <= 20.0f ? log1pf(expf(x)) : x;
    const float g = -expf(static_cast<float>(A_log[hv_idx])) * softplus_x;
    beta_scales[group_idx] =
        1.0f /
        (1.0f + expf(-static_cast<float>(
                    b[token_idx * b_stride_t + hv_idx * b_stride_h])));
    decay_scales[group_idx] = expf(g);
  }
  __syncthreads();
  const float beta_t = beta_scales[group_idx];
  const float decay = decay_scales[group_idx];

  float state_dot_k = 0.0f;
  if (valid_state) {
    for (int64_t k_idx = 0; k_idx < width; ++k_idx) {
      const int64_t state_offset =
          state_idx * state_stride_slot + hv_idx * state_stride_h +
          value_idx * state_stride_v + k_idx * state_stride_k;
      state_dot_k +=
          static_cast<float>(state[state_offset]) * decay *
          k_shared[group_idx * width + k_idx];
    }
  }

  const float v_val = static_cast<float>(
      mixed_qkv[token_idx * mixed_stride_t +
                (v_offset + value_idx) * mixed_stride_d]);
  const float v_residual = (v_val - state_dot_k) * beta_t;

  float out_val = 0.0f;
  if (valid_state) {
    for (int64_t k_idx = 0; k_idx < width; ++k_idx) {
      const float k_val = k_shared[group_idx * width + k_idx];
      const int64_t state_offset =
          state_idx * state_stride_slot + hv_idx * state_stride_h +
          value_idx * state_stride_v + k_idx * state_stride_k;
      const float new_state =
          static_cast<float>(state[state_offset]) * decay + v_residual * k_val;
      state[state_offset] = static_cast<state_t>(new_state);
      out_val += new_state * q_shared[group_idx * width + k_idx];
    }
  }

  out[token_idx * out_stride_t + 0 * out_stride_one + hv_idx * out_stride_h +
      value_idx * out_stride_v] = static_cast<scalar_t>(out_val);
}

template <typename scalar_t, typename state_t>
__device__ float causal_conv1d_gfx906_compute_update_dim(
    const scalar_t* __restrict__ x, state_t* __restrict__ conv_state,
    const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
    int64_t token_idx, int64_t state_idx, int64_t dim_idx, int64_t width,
    int64_t state_len, int64_t x_stride_b, int64_t x_stride_d,
    int64_t state_stride_b, int64_t state_stride_d, int64_t state_stride_s,
    int64_t weight_stride_d, int64_t weight_stride_w, bool has_bias,
    bool silu_activation) {
  float acc = has_bias ? static_cast<float>(bias[dim_idx]) : 0.0f;
  const float x_val =
      static_cast<float>(x[token_idx * x_stride_b + dim_idx * x_stride_d]);
  for (int64_t w_idx = 0; w_idx < width; ++w_idx) {
    const int64_t source_idx = state_len - (width - 1) + w_idx;
    const float source =
        (w_idx == width - 1)
            ? x_val
            : static_cast<float>(
                  conv_state[state_idx * state_stride_b +
                             dim_idx * state_stride_d +
                             source_idx * state_stride_s]);
    acc += source *
           static_cast<float>(
               weight[dim_idx * weight_stride_d + w_idx * weight_stride_w]);
  }
  if (silu_activation) {
    acc = acc / (1.0f + expf(-acc));
  }

  if (state_len > 0) {
    const int64_t keep_old = min(width - 1, state_len - 1);
    const int64_t target_start = state_len - 1 - keep_old;
    for (int64_t s_idx = 0; s_idx < target_start; ++s_idx) {
      conv_state[state_idx * state_stride_b + dim_idx * state_stride_d +
                 s_idx * state_stride_s] = static_cast<state_t>(0.0f);
    }
    for (int64_t s_idx = 0; s_idx < keep_old; ++s_idx) {
      const int64_t source_idx = state_len - keep_old + s_idx;
      const int64_t target_idx = target_start + s_idx;
      conv_state[state_idx * state_stride_b + dim_idx * state_stride_d +
                 target_idx * state_stride_s] =
          conv_state[state_idx * state_stride_b + dim_idx * state_stride_d +
                     source_idx * state_stride_s];
    }
    conv_state[state_idx * state_stride_b + dim_idx * state_stride_d +
               (state_len - 1) * state_stride_s] =
        static_cast<state_t>(x_val);
  }
  return acc;
}

template <typename scalar_t, typename state_t>
__global__ void
causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode_kernel(
    const scalar_t* __restrict__ mixed_qkv,
    state_t* __restrict__ conv_state, const scalar_t* __restrict__ conv_weight,
    const scalar_t* __restrict__ conv_bias, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, const scalar_t* __restrict__ A_log,
    const scalar_t* __restrict__ dt_bias, state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices, scalar_t* __restrict__ out,
    int64_t tokens, int64_t heads, int64_t kv_heads, int64_t key_dim,
    int64_t value_dim, int64_t mixed_stride_t, int64_t mixed_stride_d,
    int64_t conv_state_stride_b, int64_t conv_state_stride_d,
    int64_t conv_state_stride_s, int64_t conv_weight_stride_d,
    int64_t conv_weight_stride_w, int64_t conv_width, int64_t conv_state_len,
    int64_t a_stride_t, int64_t a_stride_h, int64_t b_stride_t,
    int64_t b_stride_h, int64_t state_stride_slot, int64_t state_stride_h,
    int64_t state_stride_v, int64_t state_stride_k, int64_t indices_stride,
    int64_t out_stride_t, int64_t out_stride_one, int64_t out_stride_h,
    int64_t out_stride_v, double scale, int64_t pad_slot_id, bool has_bias,
    bool silu_activation, bool use_tiled_qk_head_mapping) {
  constexpr int shared_width = 128;
  const int64_t token_idx = blockIdx.x / heads;
  const int64_t q_head_idx = blockIdx.x - token_idx * heads;
  const int64_t state_idx = state_indices[token_idx * indices_stride];
  const int64_t group_idx = threadIdx.x / shared_width;
  const int64_t lane = threadIdx.x - group_idx * shared_width;
  const int64_t hv_idx = use_tiled_qk_head_mapping
                              ? q_head_idx + group_idx * heads
                              : q_head_idx * 2 + group_idx;
  const int64_t value_idx = lane;

  if (token_idx >= tokens || q_head_idx >= heads || hv_idx >= kv_heads) {
    return;
  }
  if (state_idx == pad_slot_id || state_idx < 0) {
    out[token_idx * out_stride_t + 0 * out_stride_one +
        hv_idx * out_stride_h + value_idx * out_stride_v] =
        static_cast<scalar_t>(0.0f);
    return;
  }

  const int64_t q_offset = q_head_idx * key_dim;
  const int64_t k_offset = heads * key_dim + q_head_idx * key_dim;
  const int64_t v_offset = 2 * heads * key_dim + hv_idx * value_dim;

  __shared__ float q_shared[shared_width];
  __shared__ float k_shared[shared_width];
  __shared__ float q_norm_sums[shared_width];
  __shared__ float k_norm_sums[shared_width];
  __shared__ float q_norm_scale;
  __shared__ float k_norm_scale;
  __shared__ float decay_scales[2];
  __shared__ float beta_scales[2];

  if (group_idx == 0) {
    const float q_val = causal_conv1d_gfx906_compute_update_dim(
        mixed_qkv, conv_state, conv_weight, conv_bias, token_idx, state_idx,
        q_offset + lane, conv_width, conv_state_len, mixed_stride_t,
        mixed_stride_d, conv_state_stride_b, conv_state_stride_d,
        conv_state_stride_s, conv_weight_stride_d, conv_weight_stride_w,
        has_bias, silu_activation);
    const float k_val = causal_conv1d_gfx906_compute_update_dim(
        mixed_qkv, conv_state, conv_weight, conv_bias, token_idx, state_idx,
        k_offset + lane, conv_width, conv_state_len, mixed_stride_t,
        mixed_stride_d, conv_state_stride_b, conv_state_stride_d,
        conv_state_stride_s, conv_weight_stride_d, conv_weight_stride_w,
        has_bias, silu_activation);
    q_shared[lane] = q_val;
    k_shared[lane] = k_val;
    q_norm_sums[lane] = q_val * q_val;
    k_norm_sums[lane] = k_val * k_val;
  }

  const float v_val = causal_conv1d_gfx906_compute_update_dim(
      mixed_qkv, conv_state, conv_weight, conv_bias, token_idx, state_idx,
      v_offset + value_idx, conv_width, conv_state_len, mixed_stride_t,
      mixed_stride_d, conv_state_stride_b, conv_state_stride_d,
      conv_state_stride_s, conv_weight_stride_d, conv_weight_stride_w, has_bias,
      silu_activation);
  __syncthreads();

  for (int stride = shared_width / 2; stride > 0; stride >>= 1) {
    if (group_idx == 0 && lane < stride) {
      q_norm_sums[lane] += q_norm_sums[lane + stride];
      k_norm_sums[lane] += k_norm_sums[lane + stride];
    }
    __syncthreads();
  }
  if (group_idx == 0 && lane == 0) {
    q_norm_scale = rsqrtf(q_norm_sums[0] + 1e-6f);
    k_norm_scale = rsqrtf(k_norm_sums[0] + 1e-6f);
  }
  __syncthreads();

  if (group_idx == 0) {
    q_shared[lane] = q_shared[lane] * q_norm_scale * static_cast<float>(scale);
    k_shared[lane] = k_shared[lane] * k_norm_scale;
  }
  if (lane == 0) {
    const float x = static_cast<float>(
                        a[token_idx * a_stride_t + hv_idx * a_stride_h]) +
                    static_cast<float>(dt_bias[hv_idx]);
    const float softplus_x = x <= 20.0f ? log1pf(expf(x)) : x;
    const float g = -expf(static_cast<float>(A_log[hv_idx])) * softplus_x;
    beta_scales[group_idx] =
        1.0f /
        (1.0f + expf(-static_cast<float>(
                    b[token_idx * b_stride_t + hv_idx * b_stride_h])));
    decay_scales[group_idx] = expf(g);
  }
  __syncthreads();

  const float beta_t = beta_scales[group_idx];
  const float decay = decay_scales[group_idx];
  float state_dot_k = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val = k_shared[k_idx];
    const int64_t state_offset =
        state_idx * state_stride_slot + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const float decayed_state = static_cast<float>(state[state_offset]) * decay;
    state_dot_k += decayed_state * k_val;
  }

  const float v_residual = (v_val - state_dot_k) * beta_t;
  float out_val = 0.0f;
  for (int64_t k_idx = 0; k_idx < key_dim; ++k_idx) {
    const float k_val = k_shared[k_idx];
    const int64_t state_offset =
        state_idx * state_stride_slot + hv_idx * state_stride_h +
        value_idx * state_stride_v + k_idx * state_stride_k;
    const float new_state =
        static_cast<float>(state[state_offset]) * decay + v_residual * k_val;
    state[state_offset] = static_cast<state_t>(new_state);
    out_val += new_state * q_shared[k_idx];
  }

  out[token_idx * out_stride_t + 0 * out_stride_one + hv_idx * out_stride_h +
      value_idx * out_stride_v] = static_cast<scalar_t>(out_val);
}

}  // namespace

torch::Tensor causal_conv1d_gfx906_decode_update(
    torch::Tensor x, torch::Tensor conv_state, torch::Tensor weight,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> conv_state_indices, int64_t pad_slot_id,
    bool silu_activation) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(conv_state.is_cuda(), "conv_state must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must have shape [batch, dim]");
  TORCH_CHECK(conv_state.dim() == 3,
              "conv_state must have shape [cache, dim, state_len]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [dim, width]");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
              "x and weight must have the same dtype");
  TORCH_CHECK(conv_state.scalar_type() == x.scalar_type() ||
                  conv_state.scalar_type() == at::ScalarType::Float,
              "conv_state must have the same dtype as x or be float32");
  TORCH_CHECK(x.size(1) == conv_state.size(1),
              "x dim must match conv_state dim");
  TORCH_CHECK(x.size(1) == weight.size(0), "x dim must match weight dim");
  TORCH_CHECK(conv_state.size(2) >= weight.size(1) - 1,
              "conv_state is shorter than convolution width - 1");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->scalar_type() == x.scalar_type(),
                "bias must have the same dtype as x");
    TORCH_CHECK(bias->numel() == x.size(1), "bias shape must be [dim]");
  }
  if (conv_state_indices.has_value()) {
    TORCH_CHECK(conv_state_indices->is_cuda(),
                "conv_state_indices must be a CUDA tensor");
    TORCH_CHECK(conv_state_indices->scalar_type() == at::ScalarType::Int,
                "conv_state_indices must be int32");
    TORCH_CHECK(conv_state_indices->numel() == x.size(0),
                "conv_state_indices shape must be [batch]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto out = torch::empty_like(x);
  const int64_t batch = x.size(0);
  const int64_t dim = x.size(1);
  const int64_t total = batch * dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (conv_state.scalar_type() == at::ScalarType::Float) {
    VLLM_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "causal_conv1d_gfx906_decode_update", [&] {
          causal_conv1d_gfx906_decode_update_kernel<scalar_t, float>
              <<<blocks, threads, 0, stream>>>(
                  x.data_ptr<scalar_t>(), conv_state.data_ptr<float>(),
                  weight.data_ptr<scalar_t>(),
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  conv_state_indices.has_value()
                      ? conv_state_indices->data_ptr<int32_t>()
                      : nullptr,
                  out.data_ptr<scalar_t>(), batch, dim, weight.size(1),
                  conv_state.size(2), x.stride(0), x.stride(1),
                  conv_state.stride(0), conv_state.stride(1),
                  conv_state.stride(2), weight.stride(0), weight.stride(1),
                  out.stride(0), out.stride(1), pad_slot_id, bias.has_value(),
                  silu_activation);
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "causal_conv1d_gfx906_decode_update", [&] {
          causal_conv1d_gfx906_decode_update_kernel<scalar_t, scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  x.data_ptr<scalar_t>(), conv_state.data_ptr<scalar_t>(),
                  weight.data_ptr<scalar_t>(),
                  bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                  conv_state_indices.has_value()
                      ? conv_state_indices->data_ptr<int32_t>()
                      : nullptr,
                  out.data_ptr<scalar_t>(), batch, dim, weight.size(1),
                  conv_state.size(2), x.stride(0), x.stride(1),
                  conv_state.stride(0), conv_state.stride(1),
                  conv_state.stride(2), weight.stride(0), weight.stride(1),
                  out.stride(0), out.stride(1), pad_slot_id, bias.has_value(),
                  silu_activation);
        });
  }
  return out;
}

torch::Tensor causal_conv1d_gfx906_mtp_update(
    torch::Tensor x, torch::Tensor conv_state, torch::Tensor weight,
    std::optional<torch::Tensor> bias, torch::Tensor state_indices,
    torch::Tensor cu_seqlens, torch::Tensor num_accepted_tokens,
    int64_t pad_slot_id, bool silu_activation) {
  TORCH_CHECK(x.is_cuda() && conv_state.is_cuda() && weight.is_cuda(),
              "x, conv_state, and weight must be CUDA tensors");
  TORCH_CHECK(state_indices.is_cuda() && cu_seqlens.is_cuda() &&
                  num_accepted_tokens.is_cuda(),
              "MTP metadata must be CUDA tensors");
  TORCH_CHECK(x.device() == conv_state.device() &&
                  x.device() == weight.device() &&
                  x.device() == state_indices.device() &&
                  x.device() == cu_seqlens.device() &&
                  x.device() == num_accepted_tokens.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(x.dim() == 2, "x must have shape [total_tokens, dim]");
  TORCH_CHECK(conv_state.dim() == 3,
              "conv_state must have shape [slots, dim, state_len]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [dim, width]");
  TORCH_CHECK(state_indices.dim() == 1,
              "state_indices must have shape [requests]");
  TORCH_CHECK(cu_seqlens.dim() == 1,
              "cu_seqlens must have shape [requests + 1]");
  TORCH_CHECK(num_accepted_tokens.dim() == 1,
              "num_accepted_tokens must have shape [requests]");
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int &&
                  cu_seqlens.scalar_type() == at::ScalarType::Int,
              "state_indices and cu_seqlens must be int32");
  TORCH_CHECK(num_accepted_tokens.scalar_type() == at::ScalarType::Int ||
                  num_accepted_tokens.scalar_type() == at::ScalarType::Long,
              "num_accepted_tokens must be int32 or int64");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||
                  x.scalar_type() == at::ScalarType::BFloat16,
              "x and weight must be float16 or bfloat16");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(),
              "x and weight must have the same dtype");
  TORCH_CHECK(conv_state.scalar_type() == at::ScalarType::Half ||
                  conv_state.scalar_type() == at::ScalarType::BFloat16 ||
                  conv_state.scalar_type() == at::ScalarType::Float,
              "conv_state must be float16, bfloat16, or float32");
  TORCH_CHECK(x.size(1) == conv_state.size(1),
              "x dim must match conv_state dim");
  TORCH_CHECK(x.size(1) == weight.size(0), "x dim must match weight dim");
  TORCH_CHECK(weight.size(1) >= 2,
              "convolution width must be at least 2");
  TORCH_CHECK(weight.size(1) <= 6,
              "convolution width greater than 6 is not supported");
  TORCH_CHECK(conv_state.size(2) <= 64,
              "conv_state length greater than 64 is not supported");
  TORCH_CHECK(conv_state.size(2) >= weight.size(1) - 1,
              "conv_state is shorter than convolution width - 1");
  const int64_t requests = state_indices.numel();
  TORCH_CHECK(cu_seqlens.numel() == requests + 1,
              "cu_seqlens shape must be [requests + 1]");
  TORCH_CHECK(num_accepted_tokens.numel() == requests,
              "num_accepted_tokens shape must be [requests]");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->device() == x.device(),
                "bias must be on the same device as x");
    TORCH_CHECK(bias->scalar_type() == x.scalar_type(),
                "bias must have the same dtype as x");
    TORCH_CHECK(bias->numel() == x.size(1), "bias shape must be [dim]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto out = torch::empty_like(x);
  if (x.numel() == 0 || requests == 0) {
    return out;
  }
  const int64_t dim = x.size(1);
  const int64_t width = weight.size(1);
  const int64_t state_len = conv_state.size(2);
  const int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(requests),
                    static_cast<unsigned int>((dim + threads - 1) / threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

#define VLLM_GFX906_CAUSAL_CONV_MTP_LAUNCH(STATE_TYPE, ACCEPTED_TYPE)   \
  VLLM_DISPATCH_HALF_TYPES(                                             \
      x.scalar_type(), "causal_conv1d_gfx906_mtp_update", [&] {         \
        if (width == 4) {                                               \
          causal_conv1d_gfx906_mtp_update_width4_kernel<                \
              scalar_t, STATE_TYPE, ACCEPTED_TYPE>                      \
              <<<blocks, threads, 0, stream>>>(                         \
              x.data_ptr<scalar_t>(), conv_state.data_ptr<STATE_TYPE>(),\
              weight.data_ptr<scalar_t>(),                              \
              bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,  \
              state_indices.data_ptr<int32_t>(),                        \
              cu_seqlens.data_ptr<int32_t>(),                           \
              num_accepted_tokens.data_ptr<ACCEPTED_TYPE>(),            \
              out.data_ptr<scalar_t>(), dim, state_len, x.stride(0),    \
              x.stride(1), conv_state.stride(0),                        \
              conv_state.stride(1), conv_state.stride(2),               \
              weight.stride(0), weight.stride(1),                       \
              state_indices.stride(0), out.stride(0), out.stride(1),    \
              pad_slot_id,                                              \
              bias.has_value(), silu_activation);                       \
        } else {                                                        \
          causal_conv1d_gfx906_mtp_update_kernel<                       \
              scalar_t, STATE_TYPE, ACCEPTED_TYPE>                      \
              <<<blocks, threads, 0, stream>>>(                         \
              x.data_ptr<scalar_t>(), conv_state.data_ptr<STATE_TYPE>(),\
              weight.data_ptr<scalar_t>(),                              \
              bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,  \
              state_indices.data_ptr<int32_t>(),                        \
              cu_seqlens.data_ptr<int32_t>(),                           \
              num_accepted_tokens.data_ptr<ACCEPTED_TYPE>(),            \
              out.data_ptr<scalar_t>(), dim, width, state_len,          \
              x.stride(0), x.stride(1), conv_state.stride(0),           \
              conv_state.stride(1),                                     \
              conv_state.stride(2), weight.stride(0), weight.stride(1), \
              state_indices.stride(0), out.stride(0), out.stride(1),    \
              pad_slot_id,                                              \
              bias.has_value(), silu_activation);                       \
        }                                                               \
      })

  if (num_accepted_tokens.scalar_type() == at::ScalarType::Long) {
    if (conv_state.scalar_type() == at::ScalarType::Float) {
      VLLM_GFX906_CAUSAL_CONV_MTP_LAUNCH(float, int64_t);
    } else if (conv_state.scalar_type() == at::ScalarType::Half) {
      VLLM_GFX906_CAUSAL_CONV_MTP_LAUNCH(at::Half, int64_t);
    } else {
      VLLM_GFX906_CAUSAL_CONV_MTP_LAUNCH(at::BFloat16, int64_t);
    }
  } else if (conv_state.scalar_type() == at::ScalarType::Float) {
    VLLM_GFX906_CAUSAL_CONV_MTP_LAUNCH(float, int32_t);
  } else if (conv_state.scalar_type() == at::ScalarType::Half) {
    VLLM_GFX906_CAUSAL_CONV_MTP_LAUNCH(at::Half, int32_t);
  } else {
    VLLM_GFX906_CAUSAL_CONV_MTP_LAUNCH(at::BFloat16, int32_t);
  }
#undef VLLM_GFX906_CAUSAL_CONV_MTP_LAUNCH
  return out;
}

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_decode(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1,
              "q must have shape [1, tokens, heads, key_dim]");
  TORCH_CHECK(k.dim() == 4 && k.size(0) == 1,
              "k must have shape [1, tokens, heads, key_dim]");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == 1,
              "v must have shape [1, tokens, kv_heads, value_dim]");
  TORCH_CHECK(state.dim() == 4,
              "state must have shape [tokens, kv_heads, value_dim, key_dim]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, and v must have the same dtype");
  TORCH_CHECK(state.scalar_type() == q.scalar_type() ||
                  state.scalar_type() == at::ScalarType::Float,
              "state must have the same dtype as q or be float32");
  TORCH_CHECK(A_log.scalar_type() == q.scalar_type() &&
                  a.scalar_type() == q.scalar_type() &&
                  b.scalar_type() == q.scalar_type() &&
                  dt_bias.scalar_type() == q.scalar_type(),
              "A_log, a, b, dt_bias, q, k, and v must have the same dtype");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "q, k, and v token counts must match");
  TORCH_CHECK(q.size(2) == k.size(2), "q and k head counts must match");
  TORCH_CHECK(q.size(3) == k.size(3), "q and k key dims must match");
  TORCH_CHECK(v.size(2) % q.size(2) == 0,
              "kv_heads must be divisible by heads");
  TORCH_CHECK(state.size(0) >= q.size(1), "state token dim is too small");
  TORCH_CHECK(state.size(1) == v.size(2), "state kv_heads mismatch");
  TORCH_CHECK(state.size(2) == v.size(3), "state value_dim mismatch");
  TORCH_CHECK(state.size(3) == q.size(3), "state key_dim mismatch");
  TORCH_CHECK(A_log.numel() == v.size(2), "A_log shape must be [kv_heads]");
  TORCH_CHECK(dt_bias.numel() == v.size(2), "dt_bias shape must be [kv_heads]");
  if (a.dim() == 2) {
    TORCH_CHECK(a.size(0) == q.size(1) && a.size(1) == v.size(2),
                "a shape must be [tokens, kv_heads]");
  } else {
    TORCH_CHECK(a.dim() == 3 && a.size(0) == 1 && a.size(1) == q.size(1) &&
                    a.size(2) == v.size(2),
                "a shape must be [tokens, kv_heads] or [1, tokens, kv_heads]");
  }
  if (b.dim() == 2) {
    TORCH_CHECK(b.size(0) == q.size(1) && b.size(1) == v.size(2),
                "b shape must be [tokens, kv_heads]");
  } else {
    TORCH_CHECK(b.dim() == 3 && b.size(0) == 1 && b.size(1) == q.size(1) &&
                    b.size(2) == v.size(2),
                "b shape must be [tokens, kv_heads] or [1, tokens, kv_heads]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const int64_t tokens = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t kv_heads = v.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  auto out = torch::empty({1, tokens, kv_heads, value_dim}, q.options());
  const int64_t total = tokens * kv_heads * value_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const torch::Tensor a_view = a.dim() == 2 ? a : a.squeeze(0);
  const torch::Tensor b_view = b.dim() == 2 ? b : b.squeeze(0);

  if (state.scalar_type() == at::ScalarType::Float) {
    VLLM_DISPATCH_FLOATING_TYPES(
        q.scalar_type(), "fused_sigmoid_gating_delta_rule_gfx906_decode", [&] {
          fused_sigmoid_gating_delta_rule_gfx906_decode_kernel<scalar_t, float>
              <<<blocks, threads, 0, stream>>>(
                  A_log.data_ptr<scalar_t>(), a_view.data_ptr<scalar_t>(),
                  b_view.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), state.data_ptr<float>(),
                  out.data_ptr<scalar_t>(), tokens, heads, kv_heads, key_dim,
                  value_dim, q.stride(1), q.stride(2), q.stride(3),
                  k.stride(1), k.stride(2), k.stride(3), v.stride(1),
                  v.stride(2), v.stride(3), a_view.stride(0),
                  a_view.stride(1), b_view.stride(0), b_view.stride(1),
                  state.stride(0), state.stride(1), state.stride(2),
                  state.stride(3), out.stride(1), out.stride(2),
                  out.stride(3), beta, threshold, scale,
                  use_qk_l2norm_in_kernel);
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        q.scalar_type(), "fused_sigmoid_gating_delta_rule_gfx906_decode", [&] {
          fused_sigmoid_gating_delta_rule_gfx906_decode_kernel<scalar_t, scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  A_log.data_ptr<scalar_t>(), a_view.data_ptr<scalar_t>(),
                  b_view.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                  out.data_ptr<scalar_t>(), tokens, heads, kv_heads, key_dim,
                  value_dim, q.stride(1), q.stride(2), q.stride(3),
                  k.stride(1), k.stride(2), k.stride(3), v.stride(1),
                  v.stride(2), v.stride(3), a_view.stride(0),
                  a_view.stride(1), b_view.stride(0), b_view.stride(1),
                  state.stride(0), state.stride(1), state.stride(2),
                  state.stride(3), out.stride(1), out.stride(2),
                  out.stride(3), beta, threshold, scale,
                  use_qk_l2norm_in_kernel);
        });
  }
  return out;
}

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_prefill(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    torch::Tensor cu_seqlens, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
  TORCH_CHECK(cu_seqlens.is_cuda(), "cu_seqlens must be a CUDA tensor");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1,
              "q must have shape [1, tokens, heads, key_dim]");
  TORCH_CHECK(k.dim() == 4 && k.size(0) == 1,
              "k must have shape [1, tokens, heads, key_dim]");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == 1,
              "v must have shape [1, tokens, kv_heads, value_dim]");
  TORCH_CHECK(state.dim() == 4,
              "state must have shape [seqs, kv_heads, value_dim, key_dim]");
  TORCH_CHECK(cu_seqlens.dim() == 1,
              "cu_seqlens must have shape [num_seqs + 1]");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int ||
                  cu_seqlens.scalar_type() == at::ScalarType::Long,
              "cu_seqlens must be int32 or int64");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, and v must have the same dtype");
  TORCH_CHECK(state.scalar_type() == q.scalar_type() ||
                  state.scalar_type() == at::ScalarType::Float,
              "state must have the same dtype as q or be float32");
  TORCH_CHECK(A_log.scalar_type() == q.scalar_type() &&
                  a.scalar_type() == q.scalar_type() &&
                  b.scalar_type() == q.scalar_type() &&
                  dt_bias.scalar_type() == q.scalar_type(),
              "A_log, a, b, dt_bias, q, k, and v must have the same dtype");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "q, k, and v token counts must match");
  TORCH_CHECK(q.size(2) == k.size(2), "q and k head counts must match");
  TORCH_CHECK(q.size(3) == k.size(3), "q and k key dims must match");
  TORCH_CHECK(v.size(2) % q.size(2) == 0,
              "kv_heads must be divisible by heads");
  TORCH_CHECK(cu_seqlens.size(0) >= 2, "cu_seqlens must contain at least one seq");
  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  TORCH_CHECK(state.size(0) >= num_seqs, "state seq dim is too small");
  TORCH_CHECK(state.size(1) == v.size(2), "state kv_heads mismatch");
  TORCH_CHECK(state.size(2) == v.size(3), "state value_dim mismatch");
  TORCH_CHECK(state.size(3) == q.size(3), "state key_dim mismatch");
  TORCH_CHECK(A_log.numel() == v.size(2), "A_log shape must be [kv_heads]");
  TORCH_CHECK(dt_bias.numel() == v.size(2), "dt_bias shape must be [kv_heads]");
  if (a.dim() == 2) {
    TORCH_CHECK(a.size(0) == q.size(1) && a.size(1) == v.size(2),
                "a shape must be [tokens, kv_heads]");
  } else {
    TORCH_CHECK(a.dim() == 3 && a.size(0) == 1 && a.size(1) == q.size(1) &&
                    a.size(2) == v.size(2),
                "a shape must be [tokens, kv_heads] or [1, tokens, kv_heads]");
  }
  if (b.dim() == 2) {
    TORCH_CHECK(b.size(0) == q.size(1) && b.size(1) == v.size(2),
                "b shape must be [tokens, kv_heads]");
  } else {
    TORCH_CHECK(b.dim() == 3 && b.size(0) == 1 && b.size(1) == q.size(1) &&
                    b.size(2) == v.size(2),
                "b shape must be [tokens, kv_heads] or [1, tokens, kv_heads]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const int64_t tokens = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t kv_heads = v.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  auto out = torch::empty({1, tokens, kv_heads, value_dim}, q.options());
  const int64_t total = num_seqs * kv_heads * value_dim;
  const int threads = 256;
  const int blocks = static_cast<int>(total);
  const int shared_bytes = threads * sizeof(float);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const torch::Tensor a_view = a.dim() == 2 ? a : a.squeeze(0);
  const torch::Tensor b_view = b.dim() == 2 ? b : b.squeeze(0);

#define VLLM_GFX906_GDN_PREFILL_DISPATCH(INDEX_T)                              \
  if (state.scalar_type() == at::ScalarType::Float) {                          \
    VLLM_DISPATCH_FLOATING_TYPES(                                              \
        q.scalar_type(), "fused_sigmoid_gating_delta_rule_gfx906_prefill",     \
        [&] {                                                                  \
          fused_sigmoid_gating_delta_rule_gfx906_prefill_parallel_kernel<      \
              scalar_t, float, INDEX_T>                                        \
              <<<blocks, threads, shared_bytes, stream>>>(                     \
                  A_log.data_ptr<scalar_t>(), a_view.data_ptr<scalar_t>(),     \
                  b_view.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),   \
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),              \
                  v.data_ptr<scalar_t>(), state.data_ptr<float>(),             \
                  cu_seqlens.data_ptr<INDEX_T>(), out.data_ptr<scalar_t>(),    \
                  num_seqs, heads, kv_heads, key_dim, value_dim, q.stride(1),  \
                  q.stride(2), q.stride(3), k.stride(1), k.stride(2),          \
                  k.stride(3), v.stride(1), v.stride(2), v.stride(3),          \
                  a_view.stride(0), a_view.stride(1), b_view.stride(0),        \
                  b_view.stride(1), state.stride(0), state.stride(1),          \
                  state.stride(2), state.stride(3), out.stride(1),             \
                  out.stride(2), out.stride(3), beta, threshold, scale,        \
                  use_qk_l2norm_in_kernel);                                    \
        });                                                                    \
  } else {                                                                     \
    VLLM_DISPATCH_FLOATING_TYPES(                                              \
        q.scalar_type(), "fused_sigmoid_gating_delta_rule_gfx906_prefill",     \
        [&] {                                                                  \
          fused_sigmoid_gating_delta_rule_gfx906_prefill_parallel_kernel<      \
              scalar_t, scalar_t, INDEX_T>                                     \
              <<<blocks, threads, shared_bytes, stream>>>(                     \
                  A_log.data_ptr<scalar_t>(), a_view.data_ptr<scalar_t>(),     \
                  b_view.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),   \
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),              \
                  v.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),          \
                  cu_seqlens.data_ptr<INDEX_T>(), out.data_ptr<scalar_t>(),    \
                  num_seqs, heads, kv_heads, key_dim, value_dim, q.stride(1),  \
                  q.stride(2), q.stride(3), k.stride(1), k.stride(2),          \
                  k.stride(3), v.stride(1), v.stride(2), v.stride(3),          \
                  a_view.stride(0), a_view.stride(1), b_view.stride(0),        \
                  b_view.stride(1), state.stride(0), state.stride(1),          \
                  state.stride(2), state.stride(3), out.stride(1),             \
                  out.stride(2), out.stride(3), beta, threshold, scale,        \
                  use_qk_l2norm_in_kernel);                                    \
        });                                                                    \
  }

  if (cu_seqlens.scalar_type() == at::ScalarType::Int) {
    VLLM_GFX906_GDN_PREFILL_DISPATCH(int32_t);
  } else {
    VLLM_GFX906_GDN_PREFILL_DISPATCH(int64_t);
  }
#undef VLLM_GFX906_GDN_PREFILL_DISPATCH
  return out;
}

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_indexed_decode(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    torch::Tensor state_indices, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
  TORCH_CHECK(state_indices.is_cuda(), "state_indices must be a CUDA tensor");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1,
              "q must have shape [1, tokens, heads, key_dim]");
  TORCH_CHECK(k.dim() == 4 && k.size(0) == 1,
              "k must have shape [1, tokens, heads, key_dim]");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == 1,
              "v must have shape [1, tokens, kv_heads, value_dim]");
  TORCH_CHECK(state.dim() == 4,
              "state must have shape [slots, kv_heads, value_dim, key_dim]");
  TORCH_CHECK(state_indices.dim() == 1, "state_indices must have shape [tokens]");
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int,
              "state_indices must be int32");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, and v must have the same dtype");
  TORCH_CHECK(state.scalar_type() == q.scalar_type() ||
                  state.scalar_type() == at::ScalarType::Float,
              "state must have the same dtype as q or be float32");
  TORCH_CHECK(A_log.scalar_type() == q.scalar_type() &&
                  a.scalar_type() == q.scalar_type() &&
                  b.scalar_type() == q.scalar_type() &&
                  dt_bias.scalar_type() == q.scalar_type(),
              "A_log, a, b, dt_bias, q, k, and v must have the same dtype");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "q, k, and v token counts must match");
  TORCH_CHECK(state_indices.numel() == q.size(1),
              "state_indices shape must be [tokens]");
  TORCH_CHECK(q.size(2) == k.size(2), "q and k head counts must match");
  TORCH_CHECK(q.size(3) == k.size(3), "q and k key dims must match");
  TORCH_CHECK(v.size(2) % q.size(2) == 0,
              "kv_heads must be divisible by heads");
  TORCH_CHECK(state.size(1) == v.size(2), "state kv_heads mismatch");
  TORCH_CHECK(state.size(2) == v.size(3), "state value_dim mismatch");
  TORCH_CHECK(state.size(3) == q.size(3), "state key_dim mismatch");
  TORCH_CHECK(A_log.numel() == v.size(2), "A_log shape must be [kv_heads]");
  TORCH_CHECK(dt_bias.numel() == v.size(2), "dt_bias shape must be [kv_heads]");
  if (a.dim() == 2) {
    TORCH_CHECK(a.size(0) == q.size(1) && a.size(1) == v.size(2),
                "a shape must be [tokens, kv_heads]");
  } else {
    TORCH_CHECK(a.dim() == 3 && a.size(0) == 1 && a.size(1) == q.size(1) &&
                    a.size(2) == v.size(2),
                "a shape must be [tokens, kv_heads] or [1, tokens, kv_heads]");
  }
  if (b.dim() == 2) {
    TORCH_CHECK(b.size(0) == q.size(1) && b.size(1) == v.size(2),
                "b shape must be [tokens, kv_heads]");
  } else {
    TORCH_CHECK(b.dim() == 3 && b.size(0) == 1 && b.size(1) == q.size(1) &&
                    b.size(2) == v.size(2),
                "b shape must be [tokens, kv_heads] or [1, tokens, kv_heads]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const int64_t tokens = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t kv_heads = v.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  auto out = torch::empty({1, tokens, kv_heads, value_dim}, q.options());
  const int64_t total = tokens * kv_heads * value_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const torch::Tensor a_view = a.dim() == 2 ? a : a.squeeze(0);
  const torch::Tensor b_view = b.dim() == 2 ? b : b.squeeze(0);

  if (state.scalar_type() == at::ScalarType::Float) {
    VLLM_DISPATCH_FLOATING_TYPES(
        q.scalar_type(), "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode", [&] {
          fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kernel<scalar_t, float>
              <<<blocks, threads, 0, stream>>>(
                  A_log.data_ptr<scalar_t>(), a_view.data_ptr<scalar_t>(),
                  b_view.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), state.data_ptr<float>(),
                  state_indices.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
                  tokens, heads, kv_heads, key_dim, value_dim, q.stride(1),
                  q.stride(2), q.stride(3), k.stride(1), k.stride(2),
                  k.stride(3), v.stride(1), v.stride(2), v.stride(3),
                  a_view.stride(0), a_view.stride(1), b_view.stride(0),
                  b_view.stride(1), state.stride(0), state.stride(1),
                  state.stride(2), state.stride(3), state_indices.stride(0),
                  out.stride(1), out.stride(2), out.stride(3), beta,
                  threshold, scale, use_qk_l2norm_in_kernel);
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        q.scalar_type(), "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode", [&] {
          fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kernel<scalar_t,
                                                                       scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  A_log.data_ptr<scalar_t>(), a_view.data_ptr<scalar_t>(),
                  b_view.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                  state_indices.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
                  tokens, heads, kv_heads, key_dim, value_dim, q.stride(1),
                  q.stride(2), q.stride(3), k.stride(1), k.stride(2),
                  k.stride(3), v.stride(1), v.stride(2), v.stride(3),
                  a_view.stride(0), a_view.stride(1), b_view.stride(0),
                  b_view.stride(1), state.stride(0), state.stride(1),
                  state.stride(2), state.stride(3), state_indices.stride(0),
                  out.stride(1), out.stride(2), out.stride(3), beta,
                  threshold, scale, use_qk_l2norm_in_kernel);
        });
  }
  return out;
}

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    torch::Tensor state_indices, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
  TORCH_CHECK(state_indices.is_cuda(), "state_indices must be a CUDA tensor");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1,
              "q must have shape [1, tokens, heads, key_dim]");
  TORCH_CHECK(k.dim() == 4 && k.size(0) == 1,
              "k must have shape [1, tokens, heads, key_dim]");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == 1,
              "v must have shape [1, tokens, kv_heads, value_dim]");
  TORCH_CHECK(state.dim() == 4,
              "state must have shape [slots, kv_heads, key_dim, value_dim]");
  TORCH_CHECK(state_indices.dim() == 1, "state_indices must have shape [tokens]");
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int,
              "state_indices must be int32");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, and v must have the same dtype");
  TORCH_CHECK(state.scalar_type() == q.scalar_type() ||
                  state.scalar_type() == at::ScalarType::Float,
              "state must have the same dtype as q or be float32");
  TORCH_CHECK(A_log.scalar_type() == q.scalar_type() &&
                  a.scalar_type() == q.scalar_type() &&
                  b.scalar_type() == q.scalar_type() &&
                  dt_bias.scalar_type() == q.scalar_type(),
              "A_log, a, b, dt_bias, q, k, and v must have the same dtype");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "q, k, and v token counts must match");
  TORCH_CHECK(state_indices.numel() == q.size(1),
              "state_indices shape must be [tokens]");
  TORCH_CHECK(q.size(2) == k.size(2), "q and k head counts must match");
  TORCH_CHECK(q.size(3) == k.size(3), "q and k key dims must match");
  TORCH_CHECK(v.size(2) % q.size(2) == 0,
              "kv_heads must be divisible by heads");
  TORCH_CHECK(state.size(1) == v.size(2), "state kv_heads mismatch");
  TORCH_CHECK(state.size(2) == q.size(3), "state key_dim mismatch");
  TORCH_CHECK(state.size(3) == v.size(3), "state value_dim mismatch");
  TORCH_CHECK(A_log.numel() == v.size(2), "A_log shape must be [kv_heads]");
  TORCH_CHECK(dt_bias.numel() == v.size(2), "dt_bias shape must be [kv_heads]");
  if (a.dim() == 2) {
    TORCH_CHECK(a.size(0) == q.size(1) && a.size(1) == v.size(2),
                "a shape must be [tokens, kv_heads]");
  } else {
    TORCH_CHECK(a.dim() == 3 && a.size(0) == 1 && a.size(1) == q.size(1) &&
                    a.size(2) == v.size(2),
                "a shape must be [tokens, kv_heads] or [1, tokens, kv_heads]");
  }
  if (b.dim() == 2) {
    TORCH_CHECK(b.size(0) == q.size(1) && b.size(1) == v.size(2),
                "b shape must be [tokens, kv_heads]");
  } else {
    TORCH_CHECK(b.dim() == 3 && b.size(0) == 1 && b.size(1) == q.size(1) &&
                    b.size(2) == v.size(2),
                "b shape must be [tokens, kv_heads] or [1, tokens, kv_heads]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const int64_t tokens = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t kv_heads = v.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  auto out = torch::empty({1, tokens, kv_heads, value_dim}, q.options());
  const int64_t total = tokens * kv_heads * value_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const torch::Tensor a_view = a.dim() == 2 ? a : a.squeeze(0);
  const torch::Tensor b_view = b.dim() == 2 ? b : b.squeeze(0);

  if (state.scalar_type() == at::ScalarType::Float) {
    VLLM_DISPATCH_FLOATING_TYPES(
        q.scalar_type(), "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state", [&] {
          fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kernel<scalar_t, float>
              <<<blocks, threads, 0, stream>>>(
                  A_log.data_ptr<scalar_t>(), a_view.data_ptr<scalar_t>(),
                  b_view.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), state.data_ptr<float>(),
                  state_indices.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
                  tokens, heads, kv_heads, key_dim, value_dim, q.stride(1),
                  q.stride(2), q.stride(3), k.stride(1), k.stride(2),
                  k.stride(3), v.stride(1), v.stride(2), v.stride(3),
                  a_view.stride(0), a_view.stride(1), b_view.stride(0),
                  b_view.stride(1), state.stride(0), state.stride(1),
                  state.stride(3), state.stride(2), state_indices.stride(0),
                  out.stride(1), out.stride(2), out.stride(3), beta,
                  threshold, scale, use_qk_l2norm_in_kernel);
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        q.scalar_type(), "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state", [&] {
          fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kernel<scalar_t,
                                                                       scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  A_log.data_ptr<scalar_t>(), a_view.data_ptr<scalar_t>(),
                  b_view.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                  state_indices.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
                  tokens, heads, kv_heads, key_dim, value_dim, q.stride(1),
                  q.stride(2), q.stride(3), k.stride(1), k.stride(2),
                  k.stride(3), v.stride(1), v.stride(2), v.stride(3),
                  a_view.stride(0), a_view.stride(1), b_view.stride(0),
                  b_view.stride(1), state.stride(0), state.stride(1),
                  state.stride(3), state.stride(2), state_indices.stride(0),
                  out.stride(1), out.stride(2), out.stride(3), beta,
                  threshold, scale, use_qk_l2norm_in_kernel);
        });
  }
  return out;
}

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_mtp_update(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b,
    torch::Tensor dt_bias, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor state, torch::Tensor state_indices,
    torch::Tensor cu_seqlens, torch::Tensor num_accepted_tokens, double beta,
    double threshold, double scale, bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "q, k, and v must be CUDA tensors");
  TORCH_CHECK(A_log.is_cuda() && a.is_cuda() && b.is_cuda() &&
                  dt_bias.is_cuda(),
              "A_log, a, b, and dt_bias must be CUDA tensors");
  TORCH_CHECK(state.is_cuda() && state_indices.is_cuda() &&
                  cu_seqlens.is_cuda() && num_accepted_tokens.is_cuda(),
              "state and MTP metadata must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == A_log.device() && q.device() == a.device() &&
                  q.device() == b.device() && q.device() == dt_bias.device() &&
                  q.device() == state.device() &&
                  q.device() == state_indices.device() &&
                  q.device() == cu_seqlens.device() &&
                  q.device() == num_accepted_tokens.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1,
              "q must have shape [1, total_tokens, heads, key_dim]");
  TORCH_CHECK(k.dim() == 4 && k.size(0) == 1,
              "k must have shape [1, total_tokens, heads, key_dim]");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == 1,
              "v must have shape [1, total_tokens, kv_heads, value_dim]");
  TORCH_CHECK(state.dim() == 4,
              "state must have shape [slots, kv_heads, value_dim, key_dim]");
  TORCH_CHECK(state_indices.dim() == 2,
              "state_indices must have shape [requests, max_query_len]");
  TORCH_CHECK(cu_seqlens.dim() == 1,
              "cu_seqlens must have shape [requests + 1]");
  TORCH_CHECK(num_accepted_tokens.dim() == 1,
              "num_accepted_tokens must have shape [requests]");
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int &&
                  cu_seqlens.scalar_type() == at::ScalarType::Int,
              "state_indices and cu_seqlens must be int32");
  TORCH_CHECK(num_accepted_tokens.scalar_type() == at::ScalarType::Int ||
                  num_accepted_tokens.scalar_type() == at::ScalarType::Long,
              "num_accepted_tokens must be int32 or int64");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Half ||
                  q.scalar_type() == at::ScalarType::BFloat16,
              "q, k, v, A_log, a, b, and dt_bias must be float16 or bfloat16");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                  q.scalar_type() == v.scalar_type() &&
                  q.scalar_type() == A_log.scalar_type() &&
                  q.scalar_type() == a.scalar_type() &&
                  q.scalar_type() == b.scalar_type() &&
                  q.scalar_type() == dt_bias.scalar_type(),
              "q, k, v, A_log, a, b, and dt_bias must have the same dtype");
  TORCH_CHECK(state.scalar_type() == at::ScalarType::Half ||
                  state.scalar_type() == at::ScalarType::BFloat16 ||
                  state.scalar_type() == at::ScalarType::Float,
              "state must be float16, bfloat16, or float32");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "q, k, and v token counts must match");
  TORCH_CHECK(q.size(2) == k.size(2), "q and k head counts must match");
  TORCH_CHECK(q.size(3) == k.size(3), "q and k key dims must match");
  TORCH_CHECK(q.size(2) > 0 && v.size(2) % q.size(2) == 0,
              "kv_heads must be divisible by heads");
  TORCH_CHECK(state.size(1) == v.size(2), "state kv_heads mismatch");
  TORCH_CHECK(state.size(2) == v.size(3), "state value_dim mismatch");
  TORCH_CHECK(state.size(3) == q.size(3), "state key_dim mismatch");
  TORCH_CHECK(A_log.numel() == v.size(2),
              "A_log shape must be [kv_heads]");
  TORCH_CHECK(dt_bias.numel() == v.size(2),
              "dt_bias shape must be [kv_heads]");
  const int64_t requests = state_indices.size(0);
  const int64_t max_query_len = state_indices.size(1);
  TORCH_CHECK(cu_seqlens.numel() == requests + 1,
              "cu_seqlens shape must be [requests + 1]");
  TORCH_CHECK(num_accepted_tokens.numel() == requests,
              "num_accepted_tokens shape must be [requests]");
  TORCH_CHECK(max_query_len > 0 || q.size(1) == 0,
              "state_indices must have at least one token column");
  TORCH_CHECK(beta > 0.0, "beta must be positive");
  TORCH_CHECK(scale > 0.0, "scale must be positive");
  if (a.dim() == 2) {
    TORCH_CHECK(a.size(0) == q.size(1) && a.size(1) == v.size(2),
                "a shape must be [total_tokens, kv_heads]");
  } else {
    TORCH_CHECK(a.dim() == 3 && a.size(0) == 1 &&
                    a.size(1) == q.size(1) && a.size(2) == v.size(2),
                "a shape must be [total_tokens, kv_heads] or "
                "[1, total_tokens, kv_heads]");
  }
  if (b.dim() == 2) {
    TORCH_CHECK(b.size(0) == q.size(1) && b.size(1) == v.size(2),
                "b shape must be [total_tokens, kv_heads]");
  } else {
    TORCH_CHECK(b.dim() == 3 && b.size(0) == 1 &&
                    b.size(1) == q.size(1) && b.size(2) == v.size(2),
                "b shape must be [total_tokens, kv_heads] or "
                "[1, total_tokens, kv_heads]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const int64_t tokens = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t kv_heads = v.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  auto out = torch::empty({tokens, kv_heads, value_dim}, q.options());
  if (tokens == 0 || requests == 0) {
    return out;
  }

  const torch::Tensor a_view = a.dim() == 2 ? a : a.squeeze(0);
  const torch::Tensor b_view = b.dim() == 2 ? b : b.squeeze(0);
  const int threads = 256;
  const int64_t total = requests * kv_heads * value_dim;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // Warp-cooperative path (see the v2 kernel above). Falls back to the serial
  // per-column kernel for shapes it does not cover or when explicitly
  // disabled via VLLM_GFX906_GDN_MTP_V2=0.
  static const bool mtp_v2_enabled = [] {
    const char* env = std::getenv("VLLM_GFX906_GDN_MTP_V2");
    return env == nullptr || std::atoi(env) != 0;
  }();
  // v2 targets the transposed state pool ([slot][head][key][value] with v
  // contiguous); other layouts fall back to the generic serial kernel.
  const bool mtp_v2_shape_ok =
      key_dim == 128 && value_dim == 128 && max_query_len <= 4 &&
      kv_heads % heads == 0 && q.stride(3) == 1 && k.stride(3) == 1 &&
      v.stride(3) == 1 && state.stride(2) == 1 &&
      state.stride(3) == value_dim && a_view.stride(1) == 1 &&
      b_view.stride(1) == 1;
  static const bool mtp_v2_debug = [] {
    const char* env = std::getenv("VLLM_GFX906_GDN_MTP_V2_DEBUG");
    return env != nullptr && std::atoi(env) != 0;
  }();
  if (mtp_v2_debug) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      std::fprintf(
          stderr,
          "GFX906_MTP_V2 %s: key_dim=%ld value_dim=%ld maxq=%ld kv_heads=%ld "
          "heads=%ld q_s3=%ld k_s3=%ld v_s3=%ld state_s3=%ld state_s2=%ld "
          "a_s1=%ld b_s1=%ld\n",
          (mtp_v2_enabled && mtp_v2_shape_ok) ? "HIT" : "MISS",
          static_cast<long>(key_dim), static_cast<long>(value_dim),
          static_cast<long>(max_query_len), static_cast<long>(kv_heads),
          static_cast<long>(heads), static_cast<long>(q.stride(3)),
          static_cast<long>(k.stride(3)), static_cast<long>(v.stride(3)),
          static_cast<long>(state.stride(3)),
          static_cast<long>(state.stride(2)),
          static_cast<long>(a_view.stride(1)),
          static_cast<long>(b_view.stride(1)));
    }
  }
  if (mtp_v2_enabled && mtp_v2_shape_ok) {
    // Optional per-call CUDA-event timing (VLLM_GFX906_GDN_MTP_V2_TIME=1):
    // drains a 64-call-old event ring so the sync never stalls the GPU.
    static const bool mtp_v2_time = [] {
      const char* env = std::getenv("VLLM_GFX906_GDN_MTP_V2_TIME");
      return env != nullptr && std::atoi(env) != 0;
    }();
    constexpr int kEventRing = 128;
    static cudaEvent_t time_start[kEventRing];
    static cudaEvent_t time_stop[kEventRing];
    static bool time_events_ready = false;
    static int time_ring_idx = 0;
    static int time_calls = 0;
    static double time_total_ms = 0.0;
    static int time_samples = 0;
    if (mtp_v2_time && !time_events_ready) {
      for (int i = 0; i < kEventRing; ++i) {
        cudaEventCreate(&time_start[i]);
        cudaEventCreate(&time_stop[i]);
      }
      time_events_ready = true;
    }
    if (mtp_v2_time) {
      cudaEventRecord(time_start[time_ring_idx], stream);
    }
    GdnMtpV2Strides strides;
    strides.q_t = q.stride(1);
    strides.q_h = q.stride(2);
    strides.k_t = k.stride(1);
    strides.k_h = k.stride(2);
    strides.v_t = v.stride(1);
    strides.v_h = v.stride(2);
    strides.a_t = a_view.stride(0);
    strides.a_h = a_view.stride(1);
    strides.b_t = b_view.stride(0);
    strides.b_h = b_view.stride(1);
    strides.state_slot = state.stride(0);
    strides.state_h = state.stride(1);
    strides.out_t = out.stride(0);
    strides.out_h = out.stride(1);
    strides.si_req = state_indices.stride(0);
    strides.si_tok = state_indices.stride(1);
    const dim3 v2_grid(static_cast<unsigned>(kv_heads),
                       static_cast<unsigned>(requests));
#define VLLM_GFX906_MTP_V2_LAUNCH(STATE_TYPE, ACCEPTED_TYPE)              \
  VLLM_DISPATCH_HALF_TYPES(                                              \
      q.scalar_type(),                                                   \
      "fused_sigmoid_gating_delta_rule_gfx906_mtp_v2", [&] {             \
        fused_sigmoid_gating_delta_rule_gfx906_mtp_v2_kernel<            \
            scalar_t, STATE_TYPE, ACCEPTED_TYPE>                         \
            <<<v2_grid, 256, 0, stream>>>(                               \
                A_log.data_ptr<scalar_t>(),                              \
                a_view.data_ptr<scalar_t>(),                             \
                b_view.data_ptr<scalar_t>(),                             \
                dt_bias.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(),    \
                k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),          \
                state.data_ptr<STATE_TYPE>(),                            \
                state_indices.data_ptr<int32_t>(),                       \
                cu_seqlens.data_ptr<int32_t>(),                          \
                num_accepted_tokens.data_ptr<ACCEPTED_TYPE>(),           \
                out.data_ptr<scalar_t>(), max_query_len, heads, beta,    \
                threshold, scale, use_qk_l2norm_in_kernel,               \
                /*use_tiled_qk_head_mapping=*/false, strides);          \
      })
    if (num_accepted_tokens.scalar_type() == at::ScalarType::Long) {
      if (state.scalar_type() == at::ScalarType::Float) {
        VLLM_GFX906_MTP_V2_LAUNCH(float, int64_t);
      } else if (state.scalar_type() == at::ScalarType::Half) {
        VLLM_GFX906_MTP_V2_LAUNCH(at::Half, int64_t);
      } else {
        VLLM_GFX906_MTP_V2_LAUNCH(at::BFloat16, int64_t);
      }
    } else if (state.scalar_type() == at::ScalarType::Float) {
      VLLM_GFX906_MTP_V2_LAUNCH(float, int32_t);
    } else if (state.scalar_type() == at::ScalarType::Half) {
      VLLM_GFX906_MTP_V2_LAUNCH(at::Half, int32_t);
    } else {
      VLLM_GFX906_MTP_V2_LAUNCH(at::BFloat16, int32_t);
    }
#undef VLLM_GFX906_MTP_V2_LAUNCH
    if (mtp_v2_time) {
      cudaEventRecord(time_stop[time_ring_idx], stream);
      time_ring_idx = (time_ring_idx + 1) % kEventRing;
      ++time_calls;
      if (time_calls % 64 == 0) {
        for (int j = 0; j < 64; ++j) {
          const int drained =
              (time_ring_idx + kEventRing - 64 + j) % kEventRing;
          cudaEventSynchronize(time_stop[drained]);
          float ms = 0.0f;
          cudaEventElapsedTime(&ms, time_start[drained], time_stop[drained]);
          time_total_ms += ms;
          ++time_samples;
        }
        std::fprintf(stderr, "GFX906_MTP_V2_TIME avg %.1f us over %d calls\n",
                     time_total_ms * 1000.0 / time_samples, time_samples);
      }
    }
    return out;
  }

#define VLLM_GFX906_MTP_LAUNCH(STATE_TYPE, ACCEPTED_TYPE)               \
  VLLM_DISPATCH_HALF_TYPES(                                             \
      q.scalar_type(),                                                  \
      "fused_sigmoid_gating_delta_rule_gfx906_mtp_update", [&] {       \
        fused_sigmoid_gating_delta_rule_gfx906_mtp_update_kernel<      \
            scalar_t, STATE_TYPE, ACCEPTED_TYPE>                        \
            <<<blocks, threads, 0, stream>>>(                           \
                A_log.data_ptr<scalar_t>(), a_view.data_ptr<scalar_t>(),\
                b_view.data_ptr<scalar_t>(),                            \
                dt_bias.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(),   \
                k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),         \
                state.data_ptr<STATE_TYPE>(),                           \
                state_indices.data_ptr<int32_t>(),                      \
                cu_seqlens.data_ptr<int32_t>(),                         \
                num_accepted_tokens.data_ptr<ACCEPTED_TYPE>(),          \
                out.data_ptr<scalar_t>(), requests, max_query_len,      \
                heads, kv_heads, key_dim, value_dim, q.stride(1),       \
                q.stride(2), q.stride(3), k.stride(1), k.stride(2),     \
                k.stride(3), v.stride(1), v.stride(2), v.stride(3),     \
                a_view.stride(0), a_view.stride(1), b_view.stride(0),   \
                b_view.stride(1), state.stride(0), state.stride(1),     \
                state.stride(2), state.stride(3),                       \
                state_indices.stride(0), state_indices.stride(1),       \
                out.stride(0), out.stride(1), out.stride(2), beta,      \
                threshold, scale, use_qk_l2norm_in_kernel);             \
      })

  if (num_accepted_tokens.scalar_type() == at::ScalarType::Long) {
    if (state.scalar_type() == at::ScalarType::Float) {
      VLLM_GFX906_MTP_LAUNCH(float, int64_t);
    } else if (state.scalar_type() == at::ScalarType::Half) {
      VLLM_GFX906_MTP_LAUNCH(at::Half, int64_t);
    } else {
      VLLM_GFX906_MTP_LAUNCH(at::BFloat16, int64_t);
    }
  } else if (state.scalar_type() == at::ScalarType::Float) {
    VLLM_GFX906_MTP_LAUNCH(float, int32_t);
  } else if (state.scalar_type() == at::ScalarType::Half) {
    VLLM_GFX906_MTP_LAUNCH(at::Half, int32_t);
  } else {
    VLLM_GFX906_MTP_LAUNCH(at::BFloat16, int32_t);
  }
#undef VLLM_GFX906_MTP_LAUNCH
  return out;
}

torch::Tensor fused_recurrent_gated_delta_rule_gfx906_packed_decode(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state,
    torch::Tensor out, torch::Tensor state_indices, double scale,
    bool use_qk_l2norm_in_kernel, bool use_tiled_qk_head_mapping,
    bool use_transposed_state) {
  TORCH_CHECK(mixed_qkv.is_cuda(), "mixed_qkv must be a CUDA tensor");
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(A_log.is_cuda(), "A_log must be a CUDA tensor");
  TORCH_CHECK(dt_bias.is_cuda(), "dt_bias must be a CUDA tensor");
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(state_indices.is_cuda(), "state_indices must be a CUDA tensor");
  TORCH_CHECK(mixed_qkv.dim() == 2, "mixed_qkv must have shape [tokens, dim]");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must have shape [tokens, heads]");
  TORCH_CHECK(state.dim() == 4,
              "state must have shape [slots, kv_heads, value_dim, key_dim]");
  TORCH_CHECK(out.dim() == 4 && out.size(1) == 1,
              "out must have shape [tokens, 1, kv_heads, value_dim]");
  TORCH_CHECK(state_indices.dim() == 1, "state_indices must have shape [tokens]");
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int,
              "state_indices must be int32");
  TORCH_CHECK(mixed_qkv.scalar_type() == a.scalar_type() &&
                  mixed_qkv.scalar_type() == b.scalar_type() &&
                  mixed_qkv.scalar_type() == A_log.scalar_type() &&
                  mixed_qkv.scalar_type() == dt_bias.scalar_type() &&
                  mixed_qkv.scalar_type() == out.scalar_type(),
              "mixed_qkv, a, b, A_log, dt_bias, and out must have the same dtype");
  TORCH_CHECK(state.scalar_type() == mixed_qkv.scalar_type() ||
                  state.scalar_type() == at::ScalarType::Float,
              "state must have the same dtype as mixed_qkv or be float32");

  const int64_t tokens = mixed_qkv.size(0);
  const int64_t kv_heads = state.size(1);
  const int64_t value_dim = state.size(2);
  const int64_t key_dim = state.size(3);
  TORCH_CHECK(!use_transposed_state || key_dim == value_dim,
              "transposed packed GDN state requires key_dim == value_dim");
  TORCH_CHECK(a.size(0) == tokens && b.size(0) == tokens,
              "a and b token counts must match mixed_qkv");
  TORCH_CHECK(a.size(1) == kv_heads && b.size(1) == kv_heads,
              "a and b head counts must match state kv_heads");
  TORCH_CHECK(A_log.numel() == kv_heads && dt_bias.numel() == kv_heads,
              "A_log and dt_bias must have kv_heads elements");
  TORCH_CHECK(out.size(0) == tokens && out.size(2) == kv_heads &&
                  out.size(3) == value_dim,
              "out shape mismatch");
  TORCH_CHECK(state_indices.numel() == tokens,
              "state_indices shape must be [tokens]");

  const int64_t qkv_dim = mixed_qkv.size(1);
  const int64_t qk_dim = qkv_dim - kv_heads * value_dim;
  TORCH_CHECK(qk_dim > 0 && qk_dim % 2 == 0,
              "Invalid packed mixed_qkv dimension");
  const int64_t q_dim = qk_dim / 2;
  TORCH_CHECK(q_dim % key_dim == 0,
              "Packed q dimension must be divisible by key_dim");
  const int64_t heads = q_dim / key_dim;
  TORCH_CHECK(heads > 0 && kv_heads % heads == 0,
              "Invalid packed GDN head configuration");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(mixed_qkv));
  const int64_t total = tokens * kv_heads * value_dim;
  const int threads = gfx906_gdn_packed_decode_threads();
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t state_stride_v =
      use_transposed_state ? state.stride(3) : state.stride(2);
  const int64_t state_stride_k =
      use_transposed_state ? state.stride(2) : state.stride(3);
  // Warp-cooperative single-token decode path (same decomposition as the
  // MTP v2 kernel above): grid = (kv_heads, 1), 256 threads. q/k/v pointers
  // are carved out of the packed mixed_qkv rows; the token's own slot serves
  // as both source and destination (in-place update). Falls back to the
  // serial per-column kernels otherwise or via VLLM_GFX906_GDN_DEC_V2=0.
  static const bool dec_v2_enabled = [] {
    const char* env = std::getenv("VLLM_GFX906_GDN_DEC_V2");
    return env == nullptr || std::atoi(env) != 0;
  }();
  const bool dec_v2_shape_ok =
      tokens == 1 && key_dim == 128 && value_dim == 128 &&
      kv_heads % heads == 0 && mixed_qkv.stride(1) == 1 &&
      a.stride(1) == 1 && b.stride(1) == 1 && out.stride(3) == 1 &&
      out.stride(1) == out.stride(2) * kv_heads &&
      state_stride_v == 1 && state_stride_k == value_dim &&
      state_indices.stride(0) == 1;
  static const bool dec_v2_debug = [] {
    const char* env = std::getenv("VLLM_GFX906_GDN_DEC_V2_DEBUG");
    return env != nullptr && std::atoi(env) != 0;
  }();
  if (dec_v2_debug) {
    static bool dec_reported = false;
    if (!dec_reported) {
      dec_reported = true;
      std::fprintf(
          stderr,
          "GFX906_GDN_DEC_V2 %s: tokens=%ld heads=%ld kv_heads=%ld sv=%ld "
          "sk=%ld tiled_map=%d transposed=%d\n",
          (dec_v2_enabled && dec_v2_shape_ok) ? "HIT" : "MISS",
          static_cast<long>(tokens), static_cast<long>(heads),
          static_cast<long>(kv_heads), static_cast<long>(state_stride_v),
          static_cast<long>(state_stride_k), use_tiled_qk_head_mapping ? 1 : 0,
          use_transposed_state ? 1 : 0);
    }
  }
  if (dec_v2_enabled && dec_v2_shape_ok) {
    static torch::Tensor dec_cu_seqlens;
    static torch::Tensor dec_num_accepted;
    if (!dec_cu_seqlens.defined()) {
      auto opts = state_indices.options();
      dec_cu_seqlens = torch::tensor({0, 1}, opts);
      dec_num_accepted = torch::tensor({1}, opts);
    }
    // Optional per-call CUDA-event timing (VLLM_GFX906_GDN_DEC_V2_TIME=1).
    static const bool dec_v2_time = [] {
      const char* env = std::getenv("VLLM_GFX906_GDN_DEC_V2_TIME");
      return env != nullptr && std::atoi(env) != 0;
    }();
    constexpr int kDecEventRing = 128;
    static cudaEvent_t dec_time_start[kDecEventRing];
    static cudaEvent_t dec_time_stop[kDecEventRing];
    static bool dec_time_ready = false;
    static int dec_time_idx = 0;
    static int dec_time_calls = 0;
    static double dec_time_total_ms = 0.0;
    static int dec_time_samples = 0;
    if (dec_v2_time && !dec_time_ready) {
      for (int i = 0; i < kDecEventRing; ++i) {
        cudaEventCreate(&dec_time_start[i]);
        cudaEventCreate(&dec_time_stop[i]);
      }
      dec_time_ready = true;
    }
    if (dec_v2_time) {
      cudaEventRecord(dec_time_start[dec_time_idx], stream);
    }
    GdnMtpV2Strides st;
    st.q_t = mixed_qkv.stride(0);
    st.q_h = key_dim;
    st.k_t = mixed_qkv.stride(0);
    st.k_h = key_dim;
    st.v_t = mixed_qkv.stride(0);
    st.v_h = value_dim;
    st.a_t = a.stride(0);
    st.a_h = a.stride(1);
    st.b_t = b.stride(0);
    st.b_h = b.stride(1);
    st.state_slot = state.stride(0);
    st.state_h = state.stride(1);
    st.out_t = out.stride(0);
    st.out_h = out.stride(2);
    st.si_req = 1;
    st.si_tok = 0;
    const dim3 dec_grid(static_cast<unsigned>(kv_heads), 1);
#define VLLM_GFX906_DEC_V2_LAUNCH(STATE_TYPE)                            \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      mixed_qkv.scalar_type(),                                           \
      "fused_recurrent_gated_delta_rule_gfx906_dec_v2", [&] {            \
        const scalar_t* q_ptr = mixed_qkv.data_ptr<scalar_t>();          \
        const scalar_t* k_ptr = q_ptr + heads * key_dim;                 \
        const scalar_t* v_ptr = q_ptr + 2 * heads * key_dim;             \
        fused_sigmoid_gating_delta_rule_gfx906_mtp_v2_kernel<            \
            scalar_t, STATE_TYPE, int32_t>                               \
            <<<dec_grid, 256, 0, stream>>>(                              \
                A_log.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),      \
                b.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),    \
                q_ptr, k_ptr, v_ptr, state.data_ptr<STATE_TYPE>(),       \
                state_indices.data_ptr<int32_t>(),                       \
                dec_cu_seqlens.data_ptr<int32_t>(),                      \
                dec_num_accepted.data_ptr<int32_t>(),                    \
                out.data_ptr<scalar_t>(), /*max_query_len=*/1, heads,    \
                /*beta=*/1.0, /*threshold=*/20.0, scale,                 \
                use_qk_l2norm_in_kernel, use_tiled_qk_head_mapping, st); \
      })
    if (state.scalar_type() == at::ScalarType::Float) {
      VLLM_GFX906_DEC_V2_LAUNCH(float);
    } else if (state.scalar_type() == at::ScalarType::Half) {
      VLLM_GFX906_DEC_V2_LAUNCH(at::Half);
    } else {
      VLLM_GFX906_DEC_V2_LAUNCH(at::BFloat16);
    }
#undef VLLM_GFX906_DEC_V2_LAUNCH
    if (dec_v2_time) {
      cudaEventRecord(dec_time_stop[dec_time_idx], stream);
      dec_time_idx = (dec_time_idx + 1) % kDecEventRing;
      ++dec_time_calls;
      if (dec_time_calls % 64 == 0) {
        for (int j = 0; j < 64; ++j) {
          const int drained =
              (dec_time_idx + kDecEventRing - 64 + j) % kDecEventRing;
          cudaEventSynchronize(dec_time_stop[drained]);
          float ms = 0.0f;
          cudaEventElapsedTime(&ms, dec_time_start[drained],
                               dec_time_stop[drained]);
          dec_time_total_ms += ms;
          ++dec_time_samples;
        }
        std::fprintf(stderr,
                     "GFX906_GDN_DEC_V2_TIME avg %.1f us over %d calls\n",
                     dec_time_total_ms * 1000.0 / dec_time_samples,
                     dec_time_samples);
      }
    }
    return out;
  }

  const bool use_specialized_tiled128 =
      gfx906_gdn_specialized_packed_decode_enabled() &&
      use_qk_l2norm_in_kernel && use_tiled_qk_head_mapping &&
      key_dim == 128 && value_dim == 128 && threads == 256 &&
      total % threads == 0;

  if (state.scalar_type() == at::ScalarType::Float) {
    VLLM_DISPATCH_FLOATING_TYPES(
        mixed_qkv.scalar_type(),
        "fused_recurrent_gated_delta_rule_gfx906_packed_decode", [&] {
          if (use_specialized_tiled128) {
            fused_recurrent_gated_delta_rule_gfx906_packed_decode_tiled128_kernel<
                scalar_t, float><<<blocks, threads, 0, stream>>>(
                mixed_qkv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(), A_log.data_ptr<scalar_t>(),
                dt_bias.data_ptr<scalar_t>(), state.data_ptr<float>(),
                state_indices.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
                tokens, heads, kv_heads, mixed_qkv.stride(0),
                mixed_qkv.stride(1), a.stride(0), a.stride(1), b.stride(0),
                b.stride(1), state.stride(0), state.stride(1), state_stride_v,
                state_stride_k, state_indices.stride(0), out.stride(0),
                out.stride(1), out.stride(2), out.stride(3), scale);
          } else {
            fused_recurrent_gated_delta_rule_gfx906_packed_decode_kernel<
                scalar_t, float><<<blocks, threads, 0, stream>>>(
                mixed_qkv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(), A_log.data_ptr<scalar_t>(),
                dt_bias.data_ptr<scalar_t>(), state.data_ptr<float>(),
                state_indices.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
                tokens, heads, kv_heads, key_dim, value_dim,
                mixed_qkv.stride(0), mixed_qkv.stride(1), a.stride(0),
                a.stride(1), b.stride(0), b.stride(1), state.stride(0),
                state.stride(1), state_stride_v, state_stride_k,
                state_indices.stride(0), out.stride(0), out.stride(1),
                out.stride(2), out.stride(3), scale,
                use_qk_l2norm_in_kernel, use_tiled_qk_head_mapping);
          }
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        mixed_qkv.scalar_type(),
        "fused_recurrent_gated_delta_rule_gfx906_packed_decode", [&] {
          if (use_specialized_tiled128) {
            fused_recurrent_gated_delta_rule_gfx906_packed_decode_tiled128_kernel<
                scalar_t, scalar_t><<<blocks, threads, 0, stream>>>(
                mixed_qkv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(), A_log.data_ptr<scalar_t>(),
                dt_bias.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                state_indices.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
                tokens, heads, kv_heads, mixed_qkv.stride(0),
                mixed_qkv.stride(1), a.stride(0), a.stride(1), b.stride(0),
                b.stride(1), state.stride(0), state.stride(1), state_stride_v,
                state_stride_k, state_indices.stride(0), out.stride(0),
                out.stride(1), out.stride(2), out.stride(3), scale);
          } else {
            fused_recurrent_gated_delta_rule_gfx906_packed_decode_kernel<
                scalar_t, scalar_t><<<blocks, threads, 0, stream>>>(
                mixed_qkv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(), A_log.data_ptr<scalar_t>(),
                dt_bias.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                state_indices.data_ptr<int32_t>(), out.data_ptr<scalar_t>(),
                tokens, heads, kv_heads, key_dim, value_dim,
                mixed_qkv.stride(0), mixed_qkv.stride(1), a.stride(0),
                a.stride(1), b.stride(0), b.stride(1), state.stride(0),
                state.stride(1), state_stride_v, state_stride_k,
                state_indices.stride(0), out.stride(0), out.stride(1),
                out.stride(2), out.stride(3), scale,
                use_qk_l2norm_in_kernel, use_tiled_qk_head_mapping);
          }
        });
  }
  return out;
}

torch::Tensor causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode(
    torch::Tensor mixed_qkv, torch::Tensor conv_state,
    torch::Tensor conv_weight, std::optional<torch::Tensor> conv_bias,
    torch::Tensor a, torch::Tensor b, torch::Tensor A_log,
    torch::Tensor dt_bias, torch::Tensor state, torch::Tensor out,
    torch::Tensor state_indices, int64_t pad_slot_id, double scale,
    bool silu_activation, bool use_qk_l2norm_in_kernel,
    bool use_tiled_qk_head_mapping, bool use_transposed_state) {
  auto conv_out = causal_conv1d_gfx906_decode_update(
      mixed_qkv, conv_state, conv_weight, conv_bias, state_indices, pad_slot_id,
      silu_activation);
  return fused_recurrent_gated_delta_rule_gfx906_packed_decode(
      conv_out, a, b, A_log, dt_bias, state, out, state_indices, scale,
      use_qk_l2norm_in_kernel, use_tiled_qk_head_mapping,
      use_transposed_state);
}

torch::Tensor
causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode(
    torch::Tensor mixed_qkv, torch::Tensor conv_state,
    torch::Tensor conv_weight, std::optional<torch::Tensor> conv_bias,
    torch::Tensor a, torch::Tensor b, torch::Tensor A_log,
    torch::Tensor dt_bias, torch::Tensor state, torch::Tensor out,
    torch::Tensor state_indices, int64_t pad_slot_id, double scale,
    bool silu_activation, bool use_qk_l2norm_in_kernel,
    bool use_tiled_qk_head_mapping, bool use_transposed_state) {
  TORCH_CHECK(mixed_qkv.is_cuda(), "mixed_qkv must be a CUDA tensor");
  TORCH_CHECK(conv_state.is_cuda(), "conv_state must be a CUDA tensor");
  TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(A_log.is_cuda(), "A_log must be a CUDA tensor");
  TORCH_CHECK(dt_bias.is_cuda(), "dt_bias must be a CUDA tensor");
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(state_indices.is_cuda(), "state_indices must be a CUDA tensor");
  TORCH_CHECK(mixed_qkv.dim() == 2, "mixed_qkv must have shape [tokens, dim]");
  TORCH_CHECK(conv_state.dim() == 3,
              "conv_state must have shape [cache, dim, state_len]");
  TORCH_CHECK(conv_weight.dim() == 2,
              "conv_weight must have shape [dim, width]");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
              "a and b must have shape [tokens, heads]");
  TORCH_CHECK(state.dim() == 4,
              "state must have shape [slots, kv_heads, value_dim, key_dim]");
  TORCH_CHECK(out.dim() == 4 && out.size(1) == 1,
              "out must have shape [tokens, 1, kv_heads, value_dim]");
  TORCH_CHECK(state_indices.dim() == 1, "state_indices must have shape [tokens]");
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int,
              "state_indices must be int32");
  TORCH_CHECK(!conv_bias.has_value() || conv_bias->is_cuda(),
              "conv_bias must be a CUDA tensor");
  TORCH_CHECK(mixed_qkv.scalar_type() == conv_weight.scalar_type() &&
                  mixed_qkv.scalar_type() == a.scalar_type() &&
                  mixed_qkv.scalar_type() == b.scalar_type() &&
                  mixed_qkv.scalar_type() == A_log.scalar_type() &&
                  mixed_qkv.scalar_type() == dt_bias.scalar_type() &&
                  mixed_qkv.scalar_type() == out.scalar_type(),
              "mixed_qkv, conv_weight, a, b, A_log, dt_bias, and out must "
              "have the same dtype");
  TORCH_CHECK(!conv_bias.has_value() ||
                  conv_bias->scalar_type() == mixed_qkv.scalar_type(),
              "conv_bias must have the same dtype as mixed_qkv");
  TORCH_CHECK(conv_state.scalar_type() == state.scalar_type(),
              "ratio2 fused GDN path requires conv_state and state to have "
              "the same dtype");
  TORCH_CHECK(state.scalar_type() == mixed_qkv.scalar_type() ||
                  state.scalar_type() == at::ScalarType::Float,
              "state must have the same dtype as mixed_qkv or be float32");
  TORCH_CHECK(use_qk_l2norm_in_kernel,
              "ratio2 fused GDN path requires in-kernel q/k l2norm");

  const int64_t tokens = mixed_qkv.size(0);
  const int64_t kv_heads = state.size(1);
  const int64_t value_dim = state.size(2);
  const int64_t key_dim = state.size(3);
  TORCH_CHECK(key_dim == 128 && value_dim == 128,
              "ratio2 fused GDN path requires key_dim == value_dim == 128");
  TORCH_CHECK(!use_transposed_state || key_dim == value_dim,
              "transposed packed GDN state requires key_dim == value_dim");
  TORCH_CHECK(a.size(0) == tokens && b.size(0) == tokens,
              "a and b token counts must match mixed_qkv");
  TORCH_CHECK(a.size(1) == kv_heads && b.size(1) == kv_heads,
              "a and b head counts must match state kv_heads");
  TORCH_CHECK(A_log.numel() == kv_heads && dt_bias.numel() == kv_heads,
              "A_log and dt_bias must have kv_heads elements");
  TORCH_CHECK(out.size(0) == tokens && out.size(2) == kv_heads &&
                  out.size(3) == value_dim,
              "out shape mismatch");
  TORCH_CHECK(state_indices.numel() == tokens,
              "state_indices shape must be [tokens]");

  const int64_t qkv_dim = mixed_qkv.size(1);
  const int64_t qk_dim = qkv_dim - kv_heads * value_dim;
  TORCH_CHECK(qk_dim > 0 && qk_dim % 2 == 0,
              "Invalid packed mixed_qkv dimension");
  const int64_t q_dim = qk_dim / 2;
  TORCH_CHECK(q_dim % key_dim == 0,
              "Packed q dimension must be divisible by key_dim");
  const int64_t heads = q_dim / key_dim;
  TORCH_CHECK(heads > 0 && kv_heads == 2 * heads,
              "ratio2 fused GDN path requires kv_heads == 2 * heads");
  TORCH_CHECK(conv_state.size(1) == mixed_qkv.size(1),
              "conv_state dim must match mixed_qkv dim");
  TORCH_CHECK(conv_weight.size(0) == mixed_qkv.size(1),
              "conv_weight dim must match mixed_qkv dim");
  TORCH_CHECK(!conv_bias.has_value() ||
                  conv_bias->numel() == mixed_qkv.size(1),
              "conv_bias size must match mixed_qkv dim");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(mixed_qkv));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t state_stride_v =
      use_transposed_state ? state.stride(3) : state.stride(2);
  const int64_t state_stride_k =
      use_transposed_state ? state.stride(2) : state.stride(3);
  const int blocks = static_cast<int>(tokens * heads);
  constexpr int threads = 256;

  const bool has_bias = conv_bias.has_value();

  if (state.scalar_type() == at::ScalarType::Float) {
    VLLM_DISPATCH_FLOATING_TYPES(
        mixed_qkv.scalar_type(),
        "causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode",
        [&] {
          const scalar_t* typed_bias_ptr =
              has_bias ? conv_bias->data_ptr<scalar_t>() : nullptr;
          causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode_kernel<
              scalar_t, float><<<blocks, threads, 0, stream>>>(
              mixed_qkv.data_ptr<scalar_t>(), conv_state.data_ptr<float>(),
              conv_weight.data_ptr<scalar_t>(), typed_bias_ptr,
              a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
              A_log.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),
              state.data_ptr<float>(), state_indices.data_ptr<int32_t>(),
              out.data_ptr<scalar_t>(), tokens, heads, kv_heads, key_dim,
              value_dim, mixed_qkv.stride(0), mixed_qkv.stride(1),
              conv_state.stride(0), conv_state.stride(1),
              conv_state.stride(2), conv_weight.stride(0),
              conv_weight.stride(1), conv_weight.size(1), conv_state.size(2),
              a.stride(0), a.stride(1), b.stride(0), b.stride(1),
              state.stride(0), state.stride(1), state_stride_v,
              state_stride_k, state_indices.stride(0), out.stride(0),
              out.stride(1), out.stride(2), out.stride(3), scale, pad_slot_id,
              has_bias, silu_activation, use_tiled_qk_head_mapping);
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        mixed_qkv.scalar_type(),
        "causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode",
        [&] {
          const scalar_t* typed_bias_ptr =
              has_bias ? conv_bias->data_ptr<scalar_t>() : nullptr;
          causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode_kernel<
              scalar_t, scalar_t><<<blocks, threads, 0, stream>>>(
              mixed_qkv.data_ptr<scalar_t>(), conv_state.data_ptr<scalar_t>(),
              conv_weight.data_ptr<scalar_t>(), typed_bias_ptr,
              a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
              A_log.data_ptr<scalar_t>(), dt_bias.data_ptr<scalar_t>(),
              state.data_ptr<scalar_t>(), state_indices.data_ptr<int32_t>(),
              out.data_ptr<scalar_t>(), tokens, heads, kv_heads, key_dim,
              value_dim, mixed_qkv.stride(0), mixed_qkv.stride(1),
              conv_state.stride(0), conv_state.stride(1),
              conv_state.stride(2), conv_weight.stride(0),
              conv_weight.stride(1), conv_weight.size(1), conv_state.size(2),
              a.stride(0), a.stride(1), b.stride(0), b.stride(1),
              state.stride(0), state.stride(1), state_stride_v,
              state_stride_k, state_indices.stride(0), out.stride(0),
              out.stride(1), out.stride(2), out.stride(3), scale, pad_slot_id,
              has_bias, silu_activation, use_tiled_qk_head_mapping);
        });
  }
  return out;
}
