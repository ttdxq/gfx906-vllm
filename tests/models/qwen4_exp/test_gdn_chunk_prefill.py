# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3NextConfig

from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="GDN gfx906 path requires ROCm"
)


@pytest.mark.parametrize(
    "k,v,heads,kv_state",
    [(64, 32, 3, False), (128, 128, 8, False), (128, 128, 8, True)],
)
@pytest.mark.parametrize("decode_tokens", [1, 2])
@pytest.mark.parametrize("normalize_decode", [False, True])
@torch.inference_mode()
def test_gdn_chunk_varlen_state_and_continuation(
    default_vllm_config,
    dist_init,
    k,
    v,
    heads,
    kv_state,
    decode_tokens,
    normalize_decode,
):
    """Preserve adjacent GQA, both cache layouts, reset masks and continuation."""
    torch.manual_seed(7319)
    hv = heads * 3
    lengths = [129, 257]
    total = sum(lengths)
    config = Qwen3NextConfig(
        hidden_size=64,
        linear_num_key_heads=heads,
        linear_num_value_heads=hv,
        linear_key_head_dim=k,
        linear_value_head_dim=v,
    )
    with torch.device("cuda"), set_default_torch_dtype(torch.float16):
        layer = GatedDeltaNetAttention(
            config,
            default_vllm_config,
            prefix="model.layers.0.linear_attn",
            use_chunk_prefill_on_gfx906=True,
        )
    layer.use_transposed_state_for_packed_decode = kv_state
    q = F.normalize(torch.randn(1, total, heads, k, device="cuda"), dim=-1).half()
    key = F.normalize(torch.randn_like(q).float(), dim=-1).half()
    value = torch.randn(1, total, hv, v, device="cuda", dtype=torch.float16)
    a = torch.randn(total, hv, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    a_log = torch.zeros(hv, device="cuda", dtype=torch.float16)
    dt_bias = torch.ones_like(a_log)
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    beta = b.float().sigmoid()
    cached = torch.randn(2, hv, v, k, device="cuda") * 0.1
    mask = torch.tensor([True, False], device="cuda")
    cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    reference_state = cached.clone()
    reference_state[1] = 0
    if kv_state:
        cached = cached.transpose(-1, -2).contiguous()
    expected, expected_state = fused_sigmoid_gating_delta_rule_update(
        a_log,
        a,
        b,
        dt_bias,
        q,
        key,
        value,
        initial_state=reference_state,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=False,
    )
    actual, actual_state = layer._forward_chunk_prefill_gfx906(
        q, key, value, g[None], beta[None], cached, mask, cu
    )
    canonical_state = actual_state.transpose(-1, -2) if kv_state else actual_state
    for got, ref in [(actual, expected), (canonical_state, expected_state)]:
        relative_l2 = (got.float() - ref.float()).norm() / ref.float().norm()
        assert relative_l2 < 1e-3
        torch.testing.assert_close(got.float(), ref.float(), atol=5e-4, rtol=2e-3)

    # Split each request at a different boundary, preserving per-request state.
    split = [65, 129]
    tensors = [q, key, value, g[None], beta[None]]
    first = [
        torch.cat([t[:, : split[0]], t[:, lengths[0] : lengths[0] + split[1]]], 1)
        for t in tensors
    ]
    rest = [
        torch.cat([t[:, split[0] : lengths[0]], t[:, lengths[0] + split[1] :]], 1)
        for t in tensors
    ]
    first_cu = cu.new_tensor([0, split[0], sum(split)])
    rest_cu = cu.new_tensor([0, lengths[0] - split[0], total - sum(split)])
    first_out, state = layer._forward_chunk_prefill_gfx906(
        *first, cached.clone(), mask, first_cu
    )
    rest_out, state = layer._forward_chunk_prefill_gfx906(
        *rest, state, torch.ones_like(mask), rest_cu
    )
    reconstructed = torch.cat(
        [
            first_out[:, : split[0]],
            rest_out[:, : lengths[0] - split[0]],
            first_out[:, split[0] :],
            rest_out[:, lengths[0] - split[0] :],
        ],
        1,
    )
    canonical_state = state.transpose(-1, -2) if kv_state else state
    for got, ref in [(reconstructed, expected), (canonical_state, expected_state)]:
        assert (got.float() - ref.float()).norm() / ref.float().norm() < 1e-3
        torch.testing.assert_close(got.float(), ref.float(), atol=5e-4, rtol=2e-3)

    # Hand the continued cache to packed decode, including reordered slots.
    mixed = torch.cat(
        [t[0, :decode_tokens].flatten(1) for t in [q, key, value]], dim=-1
    )
    decode_cu = torch.arange(decode_tokens + 1, device=cu.device, dtype=cu.dtype)
    decode_ref, decode_state_ref = fused_sigmoid_gating_delta_rule_update(
        a_log,
        a[:decode_tokens],
        b[:decode_tokens],
        dt_bias,
        q[:, :decode_tokens],
        key[:, :decode_tokens],
        value[:, :decode_tokens],
        initial_state=expected_state.clone(),
        cu_seqlens=decode_cu,
        ssm_state_indices=cu.new_tensor([0, 1])[:decode_tokens],
        use_qk_l2norm_in_kernel=normalize_decode,
    )
    packed_state = state.flip(0).contiguous()
    packed_out = value.new_empty(decode_tokens, 1, hv, v)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed,
        a[:decode_tokens],
        b[:decode_tokens],
        a_log,
        dt_bias,
        k**-0.5,
        packed_state,
        packed_out,
        cu.new_tensor([1, 0])[:decode_tokens],
        use_qk_l2norm_in_kernel=normalize_decode,
        use_transposed_state=kv_state,
    )
    torch.testing.assert_close(packed_out[:, 0], decode_ref[0], atol=5e-4, rtol=2e-3)
    canonical_state = packed_state.transpose(-1, -2) if kv_state else packed_state
    torch.testing.assert_close(
        canonical_state.flip(0), decode_state_ref, atol=5e-4, rtol=2e-3
    )


@pytest.mark.parametrize("mixed_batch", [False, True])
@torch.inference_mode()
def test_kv_cache_recurrent_decode_and_mixed_batch(
    default_vllm_config, dist_init, mixed_batch
):
    """Both generic decode callers must preserve the KxV persistent cache."""
    torch.manual_seed(7321)
    heads, hv, width = 2, 6, 128
    tokens = 66 if mixed_batch else 2
    config = Qwen3NextConfig(
        hidden_size=64,
        linear_num_key_heads=heads,
        linear_num_value_heads=hv,
        linear_key_head_dim=width,
        linear_value_head_dim=width,
    )
    with torch.device("cuda"), set_default_torch_dtype(torch.float16):
        layer = GatedDeltaNetAttention(
            config,
            default_vllm_config,
            prefix="model.layers.1.linear_attn",
            use_chunk_prefill_on_gfx906=True,
        )
    layer.enable_packed_recurrent_decode = False
    layer.A_log.zero_()
    layer.dt_bias.fill_(1)
    layer.conv1d.weight.normal_(std=0.1)
    mixed = torch.randn(tokens, (2 * heads + hv) * width, device="cuda").half() * 0.1
    a = torch.randn(tokens, hv, device="cuda").half()
    b = torch.randn_like(a)
    state = torch.randn(2, hv, width, width, device="cuda") * 0.01
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    mask = torch.ones(2, device="cuda", dtype=torch.bool)
    metadata = GDNAttentionMetadata(
        num_prefills=int(mixed_batch),
        num_prefill_tokens=tokens - 1 if mixed_batch else 0,
        num_decodes=1 if mixed_batch else 2,
        num_decode_tokens=1 if mixed_batch else 2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=tokens,
        has_initial_state=mask,
        non_spec_query_start_loc=indices.new_tensor([0, 1, tokens]),
        non_spec_state_indices_tensor=indices,
        prefill_query_start_loc=indices.new_tensor([0, tokens - 1]),
        prefill_state_indices=indices[1:],
        prefill_has_initial_state=mask[1:],
    )
    if mixed_batch:
        (
            metadata.nums_dict,
            metadata.batch_ptr,
            metadata.token_chunk_offset_ptr,
        ) = compute_causal_conv1d_metadata(
            torch.tensor([0, 1, tokens], dtype=torch.int32), device=mixed.device
        )
    results = []
    for kv_state in (False, True):
        layer.use_transposed_state_for_packed_decode = kv_state
        cached = state.transpose(-1, -2).contiguous() if kv_state else state.clone()
        conv_state = mixed.new_zeros(2, mixed.shape[1], 3)
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2).contiguous()
        layer.kv_cache = (conv_state, cached)
        output = mixed.new_empty(tokens, hv, width)
        with set_forward_context({layer.prefix: metadata}, default_vllm_config):
            layer._forward_core(mixed.clone(), b, a, output)
        results.append((output, cached.transpose(-1, -2) if kv_state else cached))
    for got, ref in zip(results[1], results[0]):
        torch.testing.assert_close(got, ref, atol=5e-4, rtol=2e-3)
