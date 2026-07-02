# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.quant_utils import FP8_DTYPE
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.platforms import current_platform

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    if residual is not None:
        opcheck(
            torch.ops._C.fused_add_rms_norm,
            (x, residual, layer.weight.data, layer.variance_epsilon),
        )
    else:
        opcheck(
            torch.ops._C.rms_norm, (out, x, layer.weight.data, layer.variance_epsilon)
        )


@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_rms_norm_gated_gfx906(dtype: torch.dtype, norm_before_gate: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = "cuda"
    torch.manual_seed(0)
    num_tokens = 5
    hidden_size = 1024
    eps = 1e-6
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * 0.1
    z = torch.randn_like(x) * 0.1
    weight = (torch.randn(hidden_size, device=device, dtype=dtype) * 0.1 + 1.0)

    actual = torch.ops._C.rms_norm_gated_gfx906(
        x, weight, z, eps, norm_before_gate
    )
    ref_x = x.float()
    if not norm_before_gate:
        ref_x = ref_x * torch.nn.functional.silu(z.float())
    ref = ref_x * torch.rsqrt(ref_x.pow(2).mean(dim=-1, keepdim=True) + eps)
    ref = ref * weight.float()
    if norm_before_gate:
        ref = ref * torch.nn.functional.silu(z.float())
    ref = ref.to(dtype)

    torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(3, 2560), (1, 16, 128)])
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_gemma_rms_norm_gfx906(
    shape: tuple[int, ...],
    add_residual: bool,
    dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not hasattr(torch.ops._C, "gemma_rms_norm_gfx906"):
        pytest.skip("gemma_rms_norm_gfx906 is not available")

    device = "cuda"
    torch.manual_seed(0)
    hidden_size = shape[-1]
    layer = GemmaRMSNorm(hidden_size).to(device=device, dtype=dtype)
    layer.weight.data.normal_(mean=0.0, std=0.1)
    x = torch.randn(*shape, device=device, dtype=dtype) * 0.1
    residual = torch.randn_like(x) * 0.1 if add_residual else None

    ref = layer.forward_native(x, residual)
    if residual is None:
        actual = torch.ops._C.gemma_rms_norm_gfx906(
            x, layer.weight.data, layer.variance_epsilon
        )
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)
    else:
        actual = torch.ops._C.gemma_fused_add_rms_norm_gfx906(
            x, residual, layer.weight.data, layer.variance_epsilon
        )
        torch.testing.assert_close(actual[0], ref[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual[1], ref[1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_scale", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
def test_fused_rms_norm_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_scale: float,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x_base = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x_base[..., :hidden_size]
    assert x.is_contiguous() != strided_input

    x *= scale
    if add_residual:
        residual = torch.randn_like(x) * scale
        residual_fused = residual.clone()
    else:
        residual = residual_fused = None

    out_norm = torch.empty_like(x)
    out_quant = torch.empty_like(x, dtype=FP8_DTYPE)
    out_quant_fused = torch.empty_like(out_quant)

    quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32)

    if add_residual:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6
        )

        # Unfused kernel is in-place so it goes second
        # Also use a separate clone of x to avoid modifying the input
        x_unfused_base = x_base.clone()
        x_unfused = x_unfused_base[..., :hidden_size]
        assert x_unfused.is_contiguous() != strided_input
        torch.ops._C.fused_add_rms_norm(x_unfused, residual, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(
            out_quant, x_unfused.contiguous(), quant_scale_t
        )

        torch.cuda.synchronize()
        torch.testing.assert_close(residual_fused, residual, atol=1e-2, rtol=1e-2)
        opcheck(
            torch.ops._C.fused_add_rms_norm_static_fp8_quant,
            (out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6),
        )
    else:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_quant_fused, x, weight, quant_scale_t, 1e-6
        )

        torch.ops._C.rms_norm(out_norm, x, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(out_quant, out_norm, quant_scale_t)

        opcheck(
            torch.ops._C.rms_norm_static_fp8_quant,
            (out_quant_fused, x, weight, quant_scale_t, 1e-6),
        )

    torch.testing.assert_close(
        out_quant.to(dtype=torch.float32),
        out_quant_fused.to(dtype=torch.float32),
        atol=1e-3,
        rtol=1e-3,
    )
