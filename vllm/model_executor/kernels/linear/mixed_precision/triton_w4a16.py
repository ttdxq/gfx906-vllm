# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton W4A16 linear kernel for GPTQ-format packed weights."""

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

TRITON_W4A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
TRITON_W4A16_SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8,
    scalar_types.uint4,
]


@triton.jit
def triton_w4a16_gemm_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    zeros_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    shifts_row = tl.arange(0, 8) * 4
    shifts_1d = tl.reshape(
        tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8)),
        (BLOCK_N,),
    )
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )

        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_packed = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_bn[None, :] < N // 8),
            other=0,
        )
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xF

        group_idx = (k_start * BLOCK_K) // group_size
        scale_offset = group_idx * N + offs_sn
        scales = tl.load(scales_ptr + scale_offset, mask=offs_sn < N, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        if HAS_ZP:
            zero_offset = group_idx * (N // 8) + offs_bn
            z_packed = tl.load(
                zeros_ptr + zero_offset,
                mask=offs_bn < N // 8,
                other=0,
            )
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xF
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

        b_fp = (b - z).to(a.dtype) * scales
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_w4a16_gemm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 8,
) -> torch.Tensor:
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[1] * 8
    assert b_q.shape == (K, N // 8)
    assert scales.shape == (K // group_size, N)
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // 8)

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    zeros_ptr = qzeros if qzeros is not None else b_q

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1x

        if on_gfx1x():
            if M <= 32:
                block_m, block_n, block_k = 32, 32, 64
            elif M <= 64:
                block_m, block_n, block_k = 64, 64, 32
            else:
                block_m, block_n, block_k = 128, 32, 64
        else:
            if M <= 32:
                block_m, block_n, block_k = 32, 64, 32
            elif M <= 64:
                block_m, block_n, block_k = 64, 64, 32
            else:
                block_m, block_n, block_k = 128, 128, 32
    else:
        if M <= 32:
            block_m, block_n, block_k = 32, 64, 32
        elif M <= 64:
            block_m, block_n, block_k = 64, 64, 32
        else:
            block_m, block_n, block_k = 128, 128, 32

    if group_size < block_k:
        block_k = group_size

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    triton_w4a16_gemm_kernel[grid](
        a,
        b_q,
        scales,
        zeros_ptr,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        c.stride(0),
        c.stride(1),
        group_size=group_size,
        HAS_ZP=qzeros is not None,
        ZP_BIAS=zp_bias,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


class TritonW4A16LinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = TRITON_W4A16_SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not (current_platform.is_rocm() or current_platform.is_cuda()):
            return False, "TritonW4A16LinearKernel requires CUDA or ROCm"
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx906

            if on_gfx906():
                return False, "Triton W4A16 does not compile for ROCm gfx906"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Unsupported quant type {c.weight_type}"
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"
        if c.partition_weight_shape[1] % 8 != 0:
            return False, "Output features must be divisible by 8"
        if c.has_g_idx:
            return False, "Activation reordering is not supported"

        group_size = c.group_size
        if (
            group_size not in TRITON_W4A16_SUPPORTED_GROUP_SIZES
            and group_size != c.full_weight_shape[0]
        ):
            return False, f"Unsupported group size {group_size}"

        effective_group_size = (
            group_size if group_size != -1 else c.partition_weight_shape[0]
        )
        if c.partition_weight_shape[0] % effective_group_size != 0:
            return False, "Input features must be divisible by group size"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data
            output_size, packed_input_size = w.shape
            input_size = packed_input_size * 8
            shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
            unpacked = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(
                output_size, input_size
            )
            transposed = unpacked.t().contiguous()
            packed_output_size = output_size // 8
            x.data = torch.sum(
                (transposed.view(input_size, packed_output_size, 8) & 0xF)
                << shifts,
                dim=2,
                dtype=torch.int32,
            ).contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            qzeros = getattr(layer, self.w_zp_name, None)
            if qzeros is not None:
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(
                        qzeros.data.t().contiguous(),
                        requires_grad=False,
                    ),
                )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (config.partition_weight_shape[1],)

        input_size = config.partition_weight_shape[0]
        group_size = config.group_size if config.group_size != -1 else input_size
        zp_bias = config.weight_type.bias if config.weight_type.has_bias() else 0
        output = triton_w4a16_gemm(
            x_2d,
            w_q,
            w_s,
            w_zp,
            group_size,
            zp_bias,
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
