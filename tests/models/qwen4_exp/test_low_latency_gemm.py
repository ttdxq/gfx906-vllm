# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.gguf import (
    WeightType,
    _fused_mul_mat_gguf,
)
from vllm.models.qwen4_exp.amd.low_latency_gemm import (
    _hc_gemv,
    enable_qwen4_exp_low_latency_gemm,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx906

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="gfx906 Triton path requires ROCm"
)


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("rows,k", [(320, 10240), (336, 10240), (19, 8193), (7, 16384)])
@torch.inference_mode()
def test_hc_gemv_long_k_and_strided_rows(tokens: int, rows: int, k: int):
    """Cover HC shapes, padded K, output tails and noncontiguous row strides."""
    torch.manual_seed(7319)
    x = torch.randn(tokens, k + 11, device="cuda", dtype=torch.float16)[:, :k]
    weight = torch.randn(rows, k + 7, device="cuda", dtype=torch.float16)[:, :k]
    actual = _hc_gemv(x, weight)
    expected = torch.nn.functional.linear(x.float(), weight.float()).half()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("tokens", [1, 16, 17])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_hc_linear_hook_preserves_projection(
    default_vllm_config, dist_init, tokens, bias
):
    """The model hook must preserve the public linear forward, including fallback."""
    torch.manual_seed(7319)
    model = nn.Module()
    with torch.device("cuda"):
        model.input_mix_weight_down = ReplicatedLinear(
            10240, 336, bias=bias, params_dtype=torch.float16
        )
    layer = model.input_mix_weight_down
    layer.weight.normal_(std=0.01)
    if layer.bias is not None:
        layer.bias.normal_(std=0.01)
    x = torch.randn(tokens, 10240, device="cuda", dtype=torch.float16)
    expected, _ = layer(x)
    enable_qwen4_exp_low_latency_gemm(model, torch.float16)
    actual, _ = layer(x)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
    if on_gfx906():
        from vllm.models.qwen4_exp.amd.low_latency_gemm import Qwen4ExpHCGemvMethod

        assert isinstance(layer.quant_method, Qwen4ExpHCGemvMethod)


@pytest.mark.parametrize("tokens", [1, 2])
@torch.inference_mode()
def test_gguf_f16_gdn_out_proj_gemv_matches_mm(tokens: int):
    """Use the long-K GEMV only for the single-token GDN out projection."""
    torch.manual_seed(7319)
    x = torch.randn(tokens, 3072, device="cuda", dtype=torch.float16)
    weight = torch.randn(2560, 3072, device="cuda", dtype=torch.float16) * 0.01
    actual = _fused_mul_mat_gguf(x, weight, WeightType.F16)
    expected = x @ weight.T
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
