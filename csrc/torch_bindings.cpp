#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/version.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops
  //

  ops.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // PagedAttention V2.
  ops.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

  // Merge attn states
  // Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
  // can be used to combine partial attention results (in the split-KV case)
#ifndef USE_ROCM
  ops.def(
      "convert_vertical_slash_indexes("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  ops.impl("convert_vertical_slash_indexes", torch::kCUDA,
           &convert_vertical_slash_indexes);

  ops.def(
      "convert_vertical_slash_indexes_mergehead("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   Tensor vertical_indices_count, Tensor slash_indices_count, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  ops.impl("convert_vertical_slash_indexes_mergehead", torch::kCUDA,
           &convert_vertical_slash_indexes_mergehead);
#endif

  // Activation ops
  // Activation function used in SwiGLU.
#ifndef USE_ROCM
#endif

  ops.def("shared_expert_gate_mul(Tensor! out, Tensor input, Tensor weight) -> ()");
  ops.impl("shared_expert_gate_mul", torch::kCUDA, &shared_expert_gate_mul);

  ops.def(
      "shared_expert_gate_add(Tensor! routed_out, Tensor shared_out, Tensor input, "
      "Tensor weight) -> ()");
  ops.impl("shared_expert_gate_add", torch::kCUDA, &shared_expert_gate_add);

  ops.def(
      "rms_norm_gated_gfx906(Tensor input, Tensor weight, Tensor gate, "
      "float epsilon, bool norm_before_gate) -> Tensor");
  ops.impl("rms_norm_gated_gfx906", torch::kCUDA, &rms_norm_gated_gfx906);

  ops.def(
      "gemma_rms_norm_gfx906(Tensor input, Tensor weight, float epsilon) -> "
      "Tensor");
  ops.impl("gemma_rms_norm_gfx906", torch::kCUDA, &gemma_rms_norm_gfx906);

  ops.def(
      "gemma_fused_add_rms_norm_gfx906(Tensor input, Tensor residual, Tensor "
      "weight, float epsilon) -> Tensor[]");
  ops.impl("gemma_fused_add_rms_norm_gfx906", torch::kCUDA,
           &gemma_fused_add_rms_norm_gfx906);

  // Apply repetition penalties to logits in-place
  ops.def(
      "top_k_per_row(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, "
      "int stride1) -> ()");
  ops.impl("top_k_per_row", torch::kCUDA, &top_k_per_row);

  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");
  ops.impl("fused_add_rms_norm_static_fp8_quant", torch::kCUDA,
           &fused_add_rms_norm_static_fp8_quant);

  // Fused Layernorm + Quant kernels
#ifndef USE_ROCM
  // Quantized GEMM for AWQ.
  ops.def(
      "gptq_marlin_24_gemm(Tensor a, Tensor b_q_weight, Tensor b_meta, "
      "Tensor b_scales, Tensor workspace, "
      "int b_q_type, "
      "SymInt size_m, SymInt size_n, SymInt size_k) -> Tensor");
  //  conditionally compiled so impl in source file

  // Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
  ops.impl("permute_cols", torch::kCUDA, &permute_cols);

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // gptq_marlin repack from GPTQ.
#endif

  // Dequantization for GGML.
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");
  ops.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);
  ops.def(
      "ggml_repack_iq4_xs_to_q8_0(Tensor W, SymInt row, SymInt col) -> Tensor");
  ops.impl("ggml_repack_iq4_xs_to_q8_0", torch::kCUDA,
           &ggml_repack_iq4_xs_to_q8_0);

  // mmvq kernel for GGML.
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");
  ops.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8);

  ops.def("ggml_quantize_row_q8_1(Tensor X) -> Tensor");
  ops.impl("ggml_quantize_row_q8_1", torch::kCUDA, &ggml_quantize_row_q8_1);
  ops.def("ggml_quantize_row_q8_1_out(Tensor X, Tensor! quant_X) -> ()");
  ops.impl("ggml_quantize_row_q8_1_out", torch::kCUDA,
           &ggml_quantize_row_q8_1_out);

  ops.def("ggml_silu_and_mul_quantize_row_q8_1(Tensor X) -> Tensor");
  ops.impl("ggml_silu_and_mul_quantize_row_q8_1", torch::kCUDA,
           &ggml_silu_and_mul_quantize_row_q8_1);
  ops.def(
      "ggml_silu_and_mul_quantize_row_q8_1_out(Tensor X, Tensor! quant_X) -> ()");
  ops.impl("ggml_silu_and_mul_quantize_row_q8_1_out", torch::kCUDA,
           &ggml_silu_and_mul_quantize_row_q8_1_out);

  ops.def(
      "ggml_sigmoid_and_mul_quantize_row_q8_1(Tensor X, Tensor gate) -> Tensor");
  ops.impl("ggml_sigmoid_and_mul_quantize_row_q8_1", torch::kCUDA,
           &ggml_sigmoid_and_mul_quantize_row_q8_1);
  ops.def(
      "ggml_sigmoid_and_mul_quantize_row_q8_1_out(Tensor X, Tensor gate, "
      "Tensor! quant_X) -> ()");
  ops.impl("ggml_sigmoid_and_mul_quantize_row_q8_1_out", torch::kCUDA,
           &ggml_sigmoid_and_mul_quantize_row_q8_1_out);

  ops.def(
      "ggml_rms_norm_gated_quantize_row_q8_1(Tensor X, Tensor weight, "
      "Tensor gate, float epsilon, bool norm_before_gate) -> Tensor");
  ops.impl("ggml_rms_norm_gated_quantize_row_q8_1", torch::kCUDA,
           &ggml_rms_norm_gated_quantize_row_q8_1);
  ops.def(
      "ggml_rms_norm_gated_quantize_row_q8_1_out(Tensor X, Tensor weight, "
      "Tensor gate, Tensor! quant_X, float epsilon, bool norm_before_gate) -> ()");
  ops.impl("ggml_rms_norm_gated_quantize_row_q8_1_out", torch::kCUDA,
           &ggml_rms_norm_gated_quantize_row_q8_1_out);

  ops.def(
      "ggml_mul_mat_vec_q8(Tensor W, Tensor quant_X, int type, SymInt row, "
      "SymInt col, ScalarType dtype) -> Tensor");
  ops.impl("ggml_mul_mat_vec_q8", torch::kCUDA, &ggml_mul_mat_vec_q8);
  ops.def(
      "ggml_mul_mat_vec_q8_out(Tensor W, Tensor quant_X, Tensor! Y, int type, "
      "SymInt row, SymInt col) -> ()");
  ops.impl("ggml_mul_mat_vec_q8_out", torch::kCUDA, &ggml_mul_mat_vec_q8_out);

  ops.def(
      "ggml_mul_mat_vec_q8_0_fast(Tensor W, Tensor quant_X, SymInt row, "
      "SymInt col, ScalarType dtype) -> Tensor");
  ops.impl("ggml_mul_mat_vec_q8_0_fast", torch::kCUDA,
           &ggml_mul_mat_vec_q8_0_fast);

  ops.def(
      "ggml_mul_mat_vec_a8_sharded(Tensor[] W, Tensor X, int[] types) "
      "-> Tensor");
  ops.impl("ggml_mul_mat_vec_a8_sharded", torch::kCUDA,
           &ggml_mul_mat_vec_a8_sharded);

  ops.def(
      "ggml_mul_mat_vec_q8_sharded(Tensor[] W, Tensor quant_X, int[] types, "
      "SymInt col, ScalarType dtype) -> Tensor");
  ops.impl("ggml_mul_mat_vec_q8_sharded", torch::kCUDA,
           &ggml_mul_mat_vec_q8_sharded);
  ops.def(
      "ggml_mul_mat_vec_q8_sharded_out(Tensor[] W, Tensor quant_X, Tensor! Y, "
      "int[] types, SymInt col) -> ()");
  ops.impl("ggml_mul_mat_vec_q8_sharded_out", torch::kCUDA,
           &ggml_mul_mat_vec_q8_sharded_out);

  ops.def(
      "ggml_mul_mat_vec_q8_grouped_same_type(Tensor[] W, Tensor quant_X, "
      "int type, SymInt col, ScalarType dtype) -> Tensor");
  ops.impl("ggml_mul_mat_vec_q8_grouped_same_type", torch::kCUDA,
           &ggml_mul_mat_vec_q8_grouped_same_type);
  ops.def(
      "ggml_mul_mat_vec_q8_grouped_same_type_out(Tensor[] W, Tensor quant_X, "
      "Tensor! Y, int type, SymInt col) -> ()");
  ops.impl("ggml_mul_mat_vec_q8_grouped_same_type_out", torch::kCUDA,
           &ggml_mul_mat_vec_q8_grouped_same_type_out);

  ops.def(
      "ggml_mul_mat_vec_q8_qkv3(Tensor W0, Tensor W1, Tensor W2, "
      "Tensor quant_X, int type01, SymInt col, ScalarType dtype) -> Tensor");
  ops.impl("ggml_mul_mat_vec_q8_qkv3", torch::kCUDA,
           &ggml_mul_mat_vec_q8_qkv3);

  // mmq kernel for GGML.
  ops.def(
      "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.impl("ggml_mul_mat_a8", torch::kCUDA, &ggml_mul_mat_a8);

  // moe kernel for GGML.
  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.impl("ggml_moe_a8", torch::kCUDA, &ggml_moe_a8);

  ops.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens) -> Tensor");
  ops.impl("ggml_moe_a8_vec", torch::kCUDA, &ggml_moe_a8_vec);
  ops.def(
      "ggml_moe_a8_vec_out(Tensor X, Tensor W, Tensor topk_ids, "
      "Tensor! output, Tensor! quant_X, int top_k, int type, SymInt row, "
      "SymInt tokens) -> ()");
  ops.impl("ggml_moe_a8_vec_out", torch::kCUDA, &ggml_moe_a8_vec_out);

  ops.def(
      "ggml_moe_q8_vec(Tensor quant_X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens, SymInt col, ScalarType dtype) "
      "-> Tensor");
  ops.impl("ggml_moe_q8_vec", torch::kCUDA, &ggml_moe_q8_vec);

  ops.def(
      "ggml_moe_q8_vec_weighted_sum(Tensor quant_X, Tensor W, "
      "Tensor topk_ids, Tensor topk_weights, int top_k, "
      "int type, SymInt row, SymInt tokens, SymInt col, ScalarType dtype) "
      "-> Tensor");
  ops.impl("ggml_moe_q8_vec_weighted_sum", torch::kCUDA,
           &ggml_moe_q8_vec_weighted_sum);
  ops.def(
      "ggml_moe_q8_vec_weighted_sum_out(Tensor quant_X, Tensor W, "
      "Tensor topk_ids, Tensor topk_weights, Tensor! output, int top_k, "
      "int type, SymInt row, SymInt tokens, SymInt col) -> ()");
  ops.impl("ggml_moe_q8_vec_weighted_sum_out", torch::kCUDA,
           &ggml_moe_q8_vec_weighted_sum_out);

  ops.def(
      "ggml_moe_a8_vec_silu_q8_weighted_sum_out("
      "Tensor X, Tensor W1, Tensor W2, Tensor topk_ids, Tensor topk_weights, "
      "Tensor! w1_output, Tensor! quant_X, Tensor! quant_w1_output, "
      "Tensor! output, int top_k, int w1_type, int w2_type, SymInt w1_row, "
      "SymInt w2_row, SymInt tokens) -> ()");
  ops.impl("ggml_moe_a8_vec_silu_q8_weighted_sum_out", torch::kCUDA,
           &ggml_moe_a8_vec_silu_q8_weighted_sum_out);

  ops.def("ggml_moe_get_block_size", &ggml_moe_get_block_size);

  ops.def(
      "causal_conv1d_gfx906_decode_update("
      "Tensor x, Tensor conv_state, Tensor weight, Tensor? bias, "
      "Tensor? conv_state_indices, int pad_slot_id, bool silu_activation) "
      "-> Tensor");
  ops.impl("causal_conv1d_gfx906_decode_update", torch::kCUDA,
           &causal_conv1d_gfx906_decode_update);

  ops.def(
      "causal_conv1d_gfx906_mtp_update("
      "Tensor x, Tensor conv_state, Tensor weight, Tensor? bias, "
      "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
      "int pad_slot_id, bool silu_activation) -> Tensor");
  ops.impl("causal_conv1d_gfx906_mtp_update", torch::kCUDA,
           &causal_conv1d_gfx906_mtp_update);

  ops.def(
      "fused_sigmoid_gating_delta_rule_gfx906_decode("
      "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
      "Tensor v, Tensor state, float beta, float threshold, float scale, "
      "bool use_qk_l2norm_in_kernel) -> Tensor");
  ops.impl("fused_sigmoid_gating_delta_rule_gfx906_decode", torch::kCUDA,
           &fused_sigmoid_gating_delta_rule_gfx906_decode);

  ops.def(
      "fused_sigmoid_gating_delta_rule_gfx906_prefill("
      "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
      "Tensor v, Tensor state, Tensor cu_seqlens, float beta, "
      "float threshold, float scale, bool use_qk_l2norm_in_kernel) -> Tensor");
  ops.impl("fused_sigmoid_gating_delta_rule_gfx906_prefill", torch::kCUDA,
           &fused_sigmoid_gating_delta_rule_gfx906_prefill);

  ops.def(
      "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode("
      "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
      "Tensor v, Tensor state, Tensor state_indices, float beta, "
      "float threshold, float scale, bool use_qk_l2norm_in_kernel) -> Tensor");
  ops.impl("fused_sigmoid_gating_delta_rule_gfx906_indexed_decode", torch::kCUDA,
           &fused_sigmoid_gating_delta_rule_gfx906_indexed_decode);

  ops.def(
      "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state("
      "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
      "Tensor v, Tensor state, Tensor state_indices, float beta, "
      "float threshold, float scale, bool use_qk_l2norm_in_kernel) -> Tensor");
  ops.impl("fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state",
           torch::kCUDA,
           &fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state);

  ops.def(
      "fused_sigmoid_gating_delta_rule_gfx906_mtp_update("
      "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
      "Tensor v, Tensor state, Tensor state_indices, Tensor cu_seqlens, "
      "Tensor num_accepted_tokens, float beta, float threshold, float scale, "
      "bool use_qk_l2norm_in_kernel) -> Tensor");
  ops.impl("fused_sigmoid_gating_delta_rule_gfx906_mtp_update", torch::kCUDA,
           &fused_sigmoid_gating_delta_rule_gfx906_mtp_update);

  ops.def(
      "fused_recurrent_gated_delta_rule_gfx906_packed_decode("
      "Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor state, Tensor out, Tensor state_indices, float scale, "
      "bool use_qk_l2norm_in_kernel, bool use_tiled_qk_head_mapping, "
      "bool use_transposed_state) -> Tensor");
  ops.impl("fused_recurrent_gated_delta_rule_gfx906_packed_decode", torch::kCUDA,
           &fused_recurrent_gated_delta_rule_gfx906_packed_decode);

  ops.def(
      "causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode("
      "Tensor mixed_qkv, Tensor conv_state, Tensor conv_weight, "
      "Tensor? conv_bias, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor state, Tensor out, Tensor state_indices, int pad_slot_id, "
      "float scale, bool silu_activation, bool use_qk_l2norm_in_kernel, "
      "bool use_tiled_qk_head_mapping, bool use_transposed_state) -> Tensor");
  ops.impl(
      "causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode",
      torch::kCUDA,
      &causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode);

  ops.def(
      "causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode("
      "Tensor mixed_qkv, Tensor conv_state, Tensor conv_weight, "
      "Tensor? conv_bias, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor state, Tensor out, Tensor state_indices, int pad_slot_id, "
      "float scale, bool silu_activation, bool use_qk_l2norm_in_kernel, "
      "bool use_tiled_qk_head_mapping, bool use_transposed_state) -> Tensor");
  ops.impl(
      "causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode",
      torch::kCUDA,
      &causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode);

#ifndef USE_ROCM
  // CUTLASS nvfp4 block scaled GEMM
  ops.def(
      "cutlass_blockwise_scaled_grouped_mm(Tensor! output, Tensor a, Tensor b, "
      "Tensor scales_a, Tensor scales_b, "
      "Tensor problem_sizes, Tensor expert_offsets) -> ()");
  // conditionally compiled so impl registration is in source file

  // cutlass nvfp4 block scaled group GEMM
  ops.def(
      "get_cutlass_moe_mm_problem_sizes(Tensor topk_ids, "
      "                                 Tensor! problem_sizes1, "
      "                                 Tensor! problem_sizes2, "
      "                                 int num_experts, int n, int k, "
      "                                 Tensor? blockscale_offsets) -> ()");
  ops.impl("get_cutlass_moe_mm_problem_sizes", torch::kCUDA,
           &get_cutlass_moe_mm_problem_sizes);

  // A function that computes data required to run fused MoE with w8a8 grouped
  // GEMM and PPLX. It takes expert_num_tokens and non_zero_expert_idxs
  // as an input, and computes expert_offsets (token start indices of each
  // expert). In addition to this, it computes problem sizes for each expert's
  // multiplication used by the two mms called from fused MoE operation.
  ops.def(
      "get_cutlass_pplx_moe_mm_data(Tensor! expert_offsets, "
      "                             Tensor! problem_sizes1, "
      "                             Tensor! problem_sizes2, "
      "                             Tensor expert_num_tokens, "
      "                             int num_local_experts, int padded_m, "
      "                             int n, int k) -> ()");
  ops.impl("get_cutlass_pplx_moe_mm_data", torch::kCUDA,
           &get_cutlass_pplx_moe_mm_data);

  // Check if cutlass scaled_mm supports block quantization (used by DeepSeekV3)
  ops.def(
      "cutlass_sparse_scaled_mm_supported(int cuda_device_capability) -> bool");
  ops.impl("cutlass_sparse_scaled_mm_supported",
           &cutlass_sparse_scaled_mm_supported);

  // CUTLASS sparse GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  ops.def(
      "cutlass_scaled_sparse_mm(Tensor! out, Tensor a,"
      "                         Tensor bt_nzs,"
      "                         Tensor bt_meta, Tensor a_scales,"
      "                         Tensor b_scales, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_sparse_mm", torch::kCUDA, &cutlass_scaled_sparse_mm);

  // CUTLASS sparse matrix compressor
  ops.def("cutlass_sparse_compress(Tensor a) -> Tensor[]");
  ops.impl("cutlass_sparse_compress", &cutlass_sparse_compress);

  // SM100 CUTLASS MLA decode
#endif

  // Quantized GEMM for GPTQ.
  // Note: even though the C++ inferred schema is correct for this op, it seems
  // to prevent the meta function registry.
  ops.def("gptq_shuffle_awq_qweight(Tensor! q_weight, int bit) -> ()");
  ops.impl("gptq_shuffle_awq_qweight", torch::kCUDA, &gptq_shuffle_awq_qweight);

  // Compute FP8 quantized tensor for given scaling factor.
#ifndef USE_ROCM
  // Compute per-token-group FP8 quantized tensor and scaling factor.
  ops.def(
      "rearrange_kn_weight_as_n32k16_order(Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_zeros, "
      "bool has_zp, Tensor! b_qweight_reorder, Tensor! b_scales_reorder, "
      "Tensor!? b_zeros_reorder, "
      "int K, int N, int N_32align) -> ()");
  //  conditionally compiled so impl in source file

  // AllSpark quantization ops
  ops.def(
      "allspark_w8a16_gemm(Tensor a, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, "
      "SymInt n, SymInt group_size, SymInt sm_count, SymInt sm_version, SymInt "
      "CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder) -> Tensor");
  //  conditionally compiled so impl in source file
#endif
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
      "Tensor block_mapping) -> ()");
  cache_ops.impl("copy_blocks", torch::kCUDA, &copy_blocks);

  cache_ops.def(
      "copy_blocks_mla(Tensor(a!)[] kv_caches, Tensor block_mapping) -> ()");
  cache_ops.impl("copy_blocks_mla", torch::kCUDA, &copy_blocks_mla);

  // Reshape the key and value tensors and cache them.
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils

  // Gets the specified device attribute.
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.impl("get_device_attribute", &get_device_attribute);

  // Gets the maximum shared memory per block device attribute.
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  &get_max_shared_memory_per_block_device_attribute);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Custom all-reduce kernels
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("all_reduce", torch::kCUDA, &all_reduce);

  custom_ar.def("dispose", &dispose);
  custom_ar.def("meta_size", &meta_size);

  custom_ar.def("register_buffer", &register_buffer);
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  custom_ar.def("register_graph_buffers", &register_graph_buffers);

  custom_ar.def("allocate_shared_buffer_and_handle",
                &allocate_shared_buffer_and_handle);
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int", &open_mem_handle);
  custom_ar.impl("open_mem_handle", torch::kCPU, &open_mem_handle);

  custom_ar.def("free_shared_buffer", &free_shared_buffer);
#ifdef USE_ROCM
  // Quick Reduce all-reduce kernels
  custom_ar.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  custom_ar.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);

  custom_ar.def("init_custom_qr", &init_custom_qr);
  custom_ar.def("qr_destroy", &qr_destroy);

  custom_ar.def("qr_get_handle", &qr_get_handle);

  custom_ar.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  custom_ar.impl("qr_open_handles", torch::kCPU, &qr_open_handles);

  // Max input size in bytes
  custom_ar.def("qr_max_size", &qr_max_size);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
