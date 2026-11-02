# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3NextConfig

from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="GDN gfx906 path requires ROCm"
)


@pytest.mark.parametrize("k,v,heads", [(64, 32, 3), (128, 128, 8)])
@torch.inference_mode()
def test_gdn_chunk_varlen_state_and_continuation(
    default_vllm_config, dist_init, k, v, heads
):
    """Preserve adjacent GQA, VxK cache layout, reset masks and continuation."""
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
    for got, ref in [(actual, expected), (actual_state, expected_state)]:
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
    for got, ref in [(reconstructed, expected), (state, expected_state)]:
        assert (got.float() - ref.float()).norm() / ref.float().norm() < 1e-3
        torch.testing.assert_close(got.float(), ref.float(), atol=5e-4, rtol=2e-3)

    # Hand the continued cache to packed decode, including reordered slots.
    mixed = torch.cat([t[0, :2].flatten(1) for t in [q, key, value]], dim=-1)
    decode_cu = cu.new_tensor([0, 1, 2])
    decode_ref, decode_state_ref = fused_sigmoid_gating_delta_rule_update(
        a_log,
        a[:2],
        b[:2],
        dt_bias,
        q[:, :2],
        key[:, :2],
        value[:, :2],
        initial_state=expected_state.clone(),
        cu_seqlens=decode_cu,
    )
    packed_state = state.flip(0).contiguous()
    packed_out = value.new_empty(2, 1, hv, v)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed,
        a[:2],
        b[:2],
        a_log,
        dt_bias,
        k**-0.5,
        packed_state,
        packed_out,
        cu.new_tensor([1, 0]),
        use_qk_l2norm_in_kernel=False,
    )
    torch.testing.assert_close(packed_out[:, 0], decode_ref[0], atol=5e-4, rtol=2e-3)
    torch.testing.assert_close(
        packed_state.flip(0), decode_state_ref, atol=5e-4, rtol=2e-3
    )
