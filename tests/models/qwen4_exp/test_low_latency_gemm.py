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
    Qwen4ExpHCGemvMethod,
    _hc_gemv,
    enable_qwen4_exp_low_latency_gemm,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx906

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="gfx906 Triton path requires ROCm"
)


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("per_branch_norm", [False, True])
@torch.inference_mode()
def test_native_hc_decode_mix_preserves_real_layer(
    default_vllm_config, dist_init, monkeypatch, pending, per_branch_norm
):
    from vllm.models.qwen4_exp.amd.hyperconnection import (
        GatedResidual,
        HyperConnectionConfig,
    )

    monkeypatch.setenv("VLLM_QWEN4EXP_GGUF_HC_UP_QUANT", "0")
    torch.manual_seed(7319)
    with torch.device("cuda"):
        layer = GatedResidual(
            HyperConnectionConfig(
                hidden_size=2560,
                hc_lowrank=320,
                hc_count=4,
                params_dtype=torch.float16,
                hc_per_branch_norm=per_branch_norm,
            )
        )
    for param in layer.parameters():
        param.normal_(std=0.01)
    enable_qwen4_exp_low_latency_gemm(layer, torch.float16)
    residual = torch.randn(1, 10240, dtype=torch.float16, device="cuda")
    block = torch.randn(1, 2560, dtype=torch.float16, device="cuda")
    injection = torch.randn(1, 4, dtype=torch.float16, device="cuda")

    def run():
        return (
            layer.combine_and_mix(residual, block, injection)
            if pending
            else layer.mix(residual)
        )

    monkeypatch.setenv("VLLM_QWEN4EXP_NATIVE_HC_MIX", "0")
    expected = run()
    monkeypatch.setenv("VLLM_QWEN4EXP_NATIVE_HC_MIX", "1")
    actual = run()
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, atol=2e-3, rtol=1e-3)
    if on_gfx906():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            run()
        assert any(e.key == "_rocm_C::qwen4exp_hc_mix" for e in profile.key_averages())


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


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5])
@torch.inference_mode()
def test_gguf_f16_gdn_out_proj_gemv_matches_mm(tokens: int):
    """The long-K GEMV covers the GDN out projection for verify-sized batches."""
    torch.manual_seed(7319)
    x = torch.randn(tokens, 3072, device="cuda", dtype=torch.float16)
    weight = torch.randn(2560, 3072, device="cuda", dtype=torch.float16) * 0.01
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        actual = _fused_mul_mat_gguf(x, weight, WeightType.F16)
    expected = x @ weight.T
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
    used_gemv = any(
        "qwen4_exp_gemv_gfx906" in e.key for e in profile.key_averages()
    )
    assert used_gemv == (tokens <= 4)


@pytest.mark.parametrize("tokens", [2, 3, 4, 5])
@pytest.mark.parametrize(
    "rows,k,gemv_expected",
    [
        (512, 2560, True),  # MoE router
        (16, 2560, True),  # GDN a/b projections
        (2560, 3072, True),  # GDN out projection shape via the unquantized path
        (10240, 320, False),  # HC up: small K stays on triton_matmul
        (19, 257, False),  # odd rows and K tail below the crossover
        (128, 8192, True),  # K gate boundary
    ],
)
@torch.inference_mode()
def test_small_m_gemv_matches_fp32(tokens: int, rows: int, k: int, gemv_expected: bool):
    """Verify-sized 2-4 row batches ride the shared-weight GEMV where the
    measured crossover says it wins; small-K shapes and 5 rows fall back."""
    from vllm.model_executor.layers import utils

    torch.manual_seed(7319)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.01
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        actual = utils.rocm_unquantized_gemm_impl(x, weight, None)
    expected = torch.nn.functional.linear(x.float(), weight.float()).half()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
    used_gemv = any(
        "_small_m_gemv_kernel" in e.key for e in profile.key_averages()
    )
    assert used_gemv == (gemv_expected and tokens <= 4)


@torch.inference_mode()
def test_small_m_gemv_accepts_row_strided_activation():
    """A row-strided activation (stride(0) > K, unit inner stride) rides the GEMV."""
    from vllm.model_executor.layers import utils

    torch.manual_seed(7319)
    x = torch.randn(2, 2, 2560, device="cuda", dtype=torch.float16)[:, 0, :]
    assert x.shape == (2, 2560) and x.stride() == (5120, 1)
    weight = torch.randn(512, 2560, device="cuda", dtype=torch.float16) * 0.01
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        actual = utils.rocm_unquantized_gemm_impl(x, weight, None)
    expected = torch.nn.functional.linear(x.float(), weight.float()).half()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
    assert any(
        "_small_m_gemv_kernel" in e.key for e in profile.key_averages()
    )


@torch.inference_mode()
def test_small_m_gemv_opt_out(monkeypatch):
    """VLLM_QWEN4EXP_SMALL_M_GEMV=0 restores the triton_matmul dispatch."""
    from vllm import envs
    from vllm.model_executor.layers import utils

    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_SMALL_M_GEMV", False)
    torch.manual_seed(7319)
    x = torch.randn(2, 2560, device="cuda", dtype=torch.float16)
    weight = torch.randn(512, 2560, device="cuda", dtype=torch.float16) * 0.01
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        actual = utils.rocm_unquantized_gemm_impl(x, weight, None)
    expected = torch.nn.functional.linear(x.float(), weight.float()).half()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
    assert not any(
        "_small_m_gemv_kernel" in e.key for e in profile.key_averages()
    )


@pytest.mark.parametrize("tokens", [1, 2, 16])
@pytest.mark.parametrize("k", [257, 2560, 8192])
@torch.inference_mode()
def test_scalar_expert_gate_gemv_matches_fp32(tokens: int, k: int):
    """Scalar gates also support odd K and strided token/weight rows."""
    torch.manual_seed(7319)
    x = torch.randn(tokens, k + 11, device="cuda", dtype=torch.float16)[:, :k]
    weight = torch.randn(1, k + 7, device="cuda", dtype=torch.float16)[:, :k]
    actual = _hc_gemv(x, weight)
    expected = torch.nn.functional.linear(x.float(), weight.float()).half()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("tokens", [1, 16, 17])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_scalar_expert_gate_hook_preserves_linear(
    default_vllm_config, dist_init, tokens, bias
):
    """Check the real linear hook and bias/large-batch fallback."""
    torch.manual_seed(7319)
    model = nn.Module()
    with torch.device("cuda"):
        model.shared_expert_gate = ReplicatedLinear(
            2560, 1, bias=bias, params_dtype=torch.float16
        )
    layer = model.shared_expert_gate
    layer.weight.normal_(std=0.01)
    if layer.bias is not None:
        layer.bias.normal_(std=0.01)
    x = torch.randn(tokens, 2560, device="cuda", dtype=torch.float16)
    expected, _ = layer(x)
    enable_qwen4_exp_low_latency_gemm(model, torch.float16)
    actual, _ = layer(x)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
    if on_gfx906():
        assert isinstance(layer.quant_method, Qwen4ExpHCGemvMethod)


@torch.inference_mode()
def test_scalar_expert_gate_hook_opt_out(default_vllm_config, dist_init, monkeypatch):
    from vllm import envs
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    monkeypatch.setattr(envs, "VLLM_QWEN4EXP_SHARED_EXPERT_GATE_GEMV", False)
    model = nn.Module()
    with torch.device("cuda"):
        model.shared_expert_gate = ReplicatedLinear(
            2560, 1, bias=False, params_dtype=torch.float16
        )
    enable_qwen4_exp_low_latency_gemm(model, torch.float16)
    assert type(model.shared_expert_gate.quant_method) is UnquantizedLinearMethod
