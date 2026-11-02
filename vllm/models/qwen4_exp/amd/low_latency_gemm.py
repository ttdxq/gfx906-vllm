# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp low-latency GEMM hook for AMD ROCm."""

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms.rocm import on_gfx906
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _hc_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_x_row,
    stride_w_row,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + token * stride_x_row + offsets, offsets < K, other=0)
    weight = tl.load(weight_ptr + row * stride_w_row + offsets, offsets < K, other=0)
    # A CTA per output row exposes all HC rows to the GPU. Accumulation is
    # FP32, including the multiplication, before the final FP16 rounding.
    value = tl.sum(x.to(tl.float32) * weight.to(tl.float32), axis=0)
    tl.store(out_ptr + token * N + row, value)


def _hc_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = x.new_empty((x.shape[0], weight.shape[0]))
    _hc_gemv_kernel[(weight.shape[0], x.shape[0])](
        x,
        weight,
        out,
        x.stride(0),
        weight.stride(0),
        N=weight.shape[0],
        K=weight.shape[1],
        BLOCK_K=triton.next_power_of_2(weight.shape[1]),
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


def _hc_gemv_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_gemv_gfx906",
    op_func=_hc_gemv,
    fake_impl=_hc_gemv_fake,
)


class Qwen4ExpHCGemvMethod(UnquantizedLinearMethod):
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        if (
            not envs.VLLM_BATCH_INVARIANT
            and bias is None
            and x.ndim == 2
            and 1 <= x.shape[0] <= 16
            and x.is_cuda
            and x.device == weight.device
            and x.dtype == weight.dtype == torch.float16
            and x.shape[1] == weight.shape[1]
            and 8192 < weight.shape[1] <= 16384
            and x.stride(1) == weight.stride(1) == 1
        ):
            return torch.ops.vllm.qwen4_exp_gemv_gfx906(x, weight)
        return super().apply(layer, x, bias)


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """Use a long-K GEMV for gfx906 FP16 HC down projections."""
    if not on_gfx906() or dtype != torch.float16 or not envs.VLLM_QWEN4EXP_HC_GEMV:
        return
    config = get_current_vllm_config_or_none()
    if config is not None and config.kernel_config.linear_backend != "auto":
        return
    for name, child in module.named_modules():
        if not name.endswith(
            ("input_mix_weight_down", "input_mix_weight_down_block_inject")
        ):
            continue
        if not (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        ):
            continue
        weight = child.weight
        if weight.ndim == 2 and 8192 < weight.shape[1] <= 16384:
            child.quant_method = Qwen4ExpHCGemvMethod()
