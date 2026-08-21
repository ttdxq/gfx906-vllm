# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the fused GDN post-convolution preparation kernel."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)


def _reference_post_conv(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool,
    output_g_exp: bool,
):
    num_v_heads = A_log.shape[0]
    q_flat, k_flat, v_flat = torch.split(
        conv_output,
        [
            num_k_heads * head_k_dim,
            num_k_heads * head_k_dim,
            num_v_heads * head_v_dim,
        ],
        dim=-1,
    )
    q = q_flat.view(-1, num_k_heads, head_k_dim).contiguous()
    k = k_flat.view(-1, num_k_heads, head_k_dim).contiguous()
    v = v_flat.view(-1, num_v_heads, head_v_dim).contiguous()
    if apply_l2norm:
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(conv_output.dtype)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(conv_output.dtype)

    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(x, beta=1.0, threshold=20.0)
    if output_g_exp:
        g = torch.exp(g)
    return q, k, v, g, torch.sigmoid(b.float())


@pytest.mark.parametrize(
    "num_k_heads, num_v_heads, head_k_dim, head_v_dim, seq_len",
    [(2, 4, 8, 8, 17), (16, 32, 128, 128, 33)],
)
@pytest.mark.parametrize("apply_l2norm", [True, False])
@pytest.mark.parametrize("output_g_exp", [True, False])
def test_fused_post_conv_matches_reference(
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    seq_len,
    apply_l2norm,
    output_g_exp,
):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")
    torch.manual_seed(42)
    device = "cuda"
    qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    conv_output = torch.randn(seq_len, qkv_dim, device=device, dtype=torch.bfloat16)
    a = torch.randn(seq_len, num_v_heads, device=device, dtype=torch.bfloat16)
    b = torch.randn(seq_len, num_v_heads, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(num_v_heads, device=device, dtype=torch.float32) - 2.0
    dt_bias = torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1

    ref = _reference_post_conv(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        apply_l2norm,
        output_g_exp,
    )
    fused = fused_post_conv_prep(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        apply_l2norm,
        output_g_exp,
    )

    assert [tuple(x.shape) for x in fused] == [tuple(x.shape) for x in ref]
    assert fused[0].is_contiguous()
    assert fused[1].is_contiguous()
    assert fused[2].is_contiguous()
    assert fused[3].dtype == torch.float32
    assert fused[4].dtype == torch.float32
    qkv_tol = (1e-2, 1e-2) if apply_l2norm else (1e-3, 1e-3)
    for actual, expected in zip(fused[:2], ref[:2]):
        torch.testing.assert_close(actual, expected, atol=qkv_tol[0], rtol=qkv_tol[1])
    torch.testing.assert_close(fused[2], ref[2], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(fused[3], ref[3], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fused[4], ref[4], atol=1e-4, rtol=1e-4)


def test_fused_post_conv_empty_and_noncontiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")
    device = "cuda"
    H, HV, K, V, L = 4, 8, 16, 16, 7
    qkv_dim = 2 * H * K + HV * V
    base = torch.randn(qkv_dim, L, device=device, dtype=torch.bfloat16)
    conv_output = base.transpose(0, 1)
    assert not conv_output.is_contiguous()
    a = torch.randn(HV, L, device=device, dtype=torch.bfloat16).transpose(0, 1)
    b = torch.randn(HV, L, device=device, dtype=torch.bfloat16).transpose(0, 1)
    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32)

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output, a, b, A_log, dt_bias, H, K, V
    )
    assert (q.shape, k.shape, v.shape, g.shape, beta.shape) == (
        (L, H, K),
        (L, H, K),
        (L, HV, V),
        (L, HV),
        (L, HV),
    )

    empty = torch.empty(0, qkv_dim, device=device, dtype=torch.bfloat16)
    empty_a = torch.empty(0, HV, device=device, dtype=torch.bfloat16)
    empty_out = fused_post_conv_prep(empty, empty_a, empty_a, A_log, dt_bias, H, K, V)
    assert empty_out[0].shape == (0, H, K)
    assert empty_out[3].shape == (0, HV)
