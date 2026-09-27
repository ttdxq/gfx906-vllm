# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.rocm import on_gfx9
from vllm.utils.flashinfer import (
    flashinfer_bf16_mm,
    is_flashinfer_cutedsl_bf16_gemm_supported,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=3,
            num_warps=2,
        ),
    ]


def get_heuristics():
    return {"BLOCK_SIZE_M": lambda args: min(16, triton.next_power_of_2(args["M"]))}


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.heuristics(values=get_heuristics())
@triton.jit
def triton_matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], (
        "Incompatible dimensions"
    )  # NOTE(gfx906): b.shape inv
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = b.shape  # NOTE(gfx906): b.shape inv
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    triton_matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(1),
        b.stride(0),  # NOTE(gfx906): b.stride inv
        c.stride(0),
        c.stride(1),  #
    )
    return c


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    linear_input = x if x.dtype == weight.dtype else x.to(weight.dtype)
    return torch.nn.functional.linear(linear_input, weight, bias)


_FlashInferBf16RuntimeCheck = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None], bool
]


@dataclass(frozen=True)
class _FlashInferBf16Backend:
    flashinfer_backend: str
    is_supported: Callable[[], bool]
    can_implement: _FlashInferBf16RuntimeCheck


def _can_use_flashinfer_cutedsl_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return False
    if x.ndim < 1 or weight.ndim != 2:
        return False
    if (
        not x.is_cuda
        or not weight.is_cuda
        or x.device != weight.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        return False

    k = x.shape[-1]
    n = weight.shape[0]
    if (
        k <= 0
        or n <= 0
        or weight.shape[1] != k
        or k % 128 != 0
        or x.data_ptr() % 32 != 0
        or weight.data_ptr() % 32 != 0
    ):
        return False

    m = x.numel() // k
    if not 1 <= m <= 32:
        return False
    return bias is None or (
        bias.is_cuda
        and bias.device == x.device
        and bias.dtype == torch.bfloat16
        and bias.ndim == 1
        and bias.shape[0] == n
        and bias.is_contiguous()
    )


_FLASHINFER_BF16_BACKENDS = {
    "flashinfer_cutedsl": _FlashInferBf16Backend(
        flashinfer_backend="cute-dsl",
        is_supported=is_flashinfer_cutedsl_bf16_gemm_supported,
        can_implement=_can_use_flashinfer_cutedsl_bf16,
    ),
}


def _get_flashinfer_bf16_backend(vllm_backend: str) -> _FlashInferBf16Backend:
    backend_spec = _FLASHINFER_BF16_BACKENDS.get(vllm_backend)
    if backend_spec is None:
        supported = ", ".join(sorted(_FLASHINFER_BF16_BACKENDS))
        raise ValueError(
            f"Unsupported vLLM FlashInfer BF16 backend {vllm_backend!r}; "
            f"supported backends: {supported}"
        )
    return backend_spec


def cuda_flashinfer_bf16_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pdl: bool,
    vllm_backend: str,
) -> torch.Tensor:
    backend_spec = _get_flashinfer_bf16_backend(vllm_backend)
    if not backend_spec.can_implement(x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias)

    k = x.shape[-1]
    n = weight.shape[0]
    x_2d = x.view(-1, k)
    out_2d = flashinfer_bf16_mm(
        x_2d,
        weight.t(),
        bias,
        pdl,
        backend_spec.flashinfer_backend,
    )
    return out_2d.view(*x.shape[:-1], n)


def cuda_flashinfer_bf16_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pdl: bool,
    vllm_backend: str,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def cuda_flashinfer_bf16_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    vllm_backend: str,
    pdl: bool,
) -> torch.Tensor:
    return torch.ops.vllm.cuda_flashinfer_bf16_gemm(
        x,
        weight,
        bias,
        pdl,
        vllm_backend,
    )


direct_register_custom_op(
    op_name="cuda_flashinfer_bf16_gemm",
    op_func=cuda_flashinfer_bf16_gemm_impl,
    fake_impl=cuda_flashinfer_bf16_gemm_fake,
)


def use_aiter_triton_gemm(n, m, k, dtype):
    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
    )


def wvsplitkrc_dispatch(n: int, k: int, m: int, cu_count: int) -> tuple[int, bool]:
    """Pick the K-shard split for wvSplitKrc and say whether the shape fits.

    Mirrors wvSplitKrc() in csrc/rocm/skinny_gemms.cu, which is also where the
    shard cap is explained. Both must pick the same chunkk or the workspace
    check here bounds the wrong k_rnd.

    Returns:
        The CHUNKK the kernel will dispatch with, and whether the CU budget and
        split-K workspace admit the shape at all.

    """
    # Next ^2 of n
    N_p2 = 1 << (n - 1).bit_length()
    # How many of 4 waves in a group can work on same 16 Ms at same time?
    # This reduces the Ms each group works on, i.e. increasing the CUs needed.
    GrpsShrB = min(N_p2 // 16, 4)
    # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile), and each
    # working on a 512-shard of K, how many CUs would we need?
    CuNeeded = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512) * GrpsShrB

    CHUNKK2_MAX_SHARDS = 11
    shards_chunkk2 = (k + 256 - 1) // 256  # 256-wide shards
    chunkk = (
        2
        if (
            N_p2 != 16
            and CuNeeded * 2 <= cu_count
            and shards_chunkk2 <= CHUNKK2_MAX_SHARDS
        )
        else 1
    )

    # Deterministic reduction stores one fp32 partial per (M, N, k-shard); all
    # of them must fit the split-K workspace.
    k_rnd = (k + 512 // chunkk - 1) // (512 // chunkk)
    fits = N_p2 * m * k_rnd <= 128 * 1024 * 12 and CuNeeded <= cu_count
    return chunkk, fits


def rocm_unquantized_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    x_view = x.reshape(-1, x.size(-1))
    n = x_view.shape[0]
    m = weight.shape[0]
    k = weight.shape[1]

    # For FP16/BF16 without bias and k % 8 == 0, prefer skinny GEMV kernel
    if (
        x.dtype in [torch.float16, torch.bfloat16]
        and bias is None
        and m % 4 == 0
        and n == 1
        and k <= 8192
        and k % 8 == 0
    ):
        out = ops.LLMM1(weight, x_view, 4)
        return out.view(*x.shape[:-1], weight.shape[0])

    if n <= 16 and not on_gfx9():
        out = triton_matmul(x_view, weight).view(*x.shape[:-1], weight.shape[0])
        if bias is not None:
            out = out + bias
        return out

    linear_input = x if x.dtype == weight.dtype else x.to(weight.dtype)
    return torch.nn.functional.linear(linear_input, weight, bias)


def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm",
    op_func=rocm_unquantized_gemm_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)


@functools.cache
def warmup_rocm_skinny_gemm_workspaces(device: torch.device) -> None:
    """Eagerly allocate wvSplitKrc's per-device split-K workspace pool.

    wvSplitKrc partitions one per-device allocation into ``kWvSlots`` slots
    (csrc/rocm/skinny_gemms.cu) and hands each stream one on first use, so that
    two streams never share the split-K partials and counters.

    The pool is otherwise created lazily on the first qualifying GEMM
    (csrc/rocm/skinny_gemms.cu), which can be the first real request — after
    the KV cache backing buffer exists. If it landed in that segment's rounding
    tail, it would pin the entire segment at engine shutdown; it could also land
    inside a cudagraph capture, where it would be taken from the graph's private
    pool and its zero-fill would become a replayed graph node.
    """
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return
    try:
        x = torch.zeros(16, 1024, dtype=torch.bfloat16, device=device)
        weight = torch.zeros(32, 1024, dtype=torch.bfloat16, device=device)
        ops.wvSplitKrc(x, weight, num_compute_units())
    except Exception:
        logger.debug("wvSplitKrc workspace warmup failed", exc_info=True)


# Above this weight size, oneDNN's onednn_mm consistently matches or beats
# the SGL AMX kernel once M grows past decode-sized batches, and is within
# noise of it at decode-sized M -- so larger weights default to oneDNN
# rather than SGL. 1 MiB comfortably covers MoE router/gate weights (e.g.
# (2048, 128) .. (2880, 32) bf16/fp16, 180-720 KiB) while staying well below
# any dense qkv/o_proj/gate_up/down/lm_head projection in practice. This
# threshold is derived from bf16/fp16 unquantized dense-GEMM benchmarks only,
# so it does not apply to the int8 scaled_mm path below.
_CPU_SGL_GEMM_MAX_WEIGHT_BYTES = 1 * 1024 * 1024


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    if not torch.cpu._is_amx_tile_supported() or dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.int8,
    ):
        return False
    if dtype == torch.float16 and not torch.cpu._is_amx_fp16_supported():
        # AMX-BF16/INT8 (amx_tile) and AMX-FP16 are separate CPU ISA
        # extensions -- e.g. Sapphire/Emerald Rapids expose the former but
        # not the latter -- and can_use_brgemm<at::Half> (gemm.h) always
        # attempts brgemm for fp16 regardless of M, so this needs its own
        # capability check rather than piggybacking on amx_tile.
        return False
    if dtype == torch.int8:
        # int8_scaled_mm_with_quant requires the packed weight to stay int8
        # (gemm_int8.cpp); convert_weight_packed's N < TILE_N fallback
        # returns a float32 tensor instead (gemm.cpp), which would trip
        # that check, so N must be a full TILE_N tile here.
        return k % 32 == 0 and n % 16 == 0
    if n * k * dtype.itemsize > _CPU_SGL_GEMM_MAX_WEIGHT_BYTES:
        return False
    if n < 16:
        # convert_weight_packed transposes to fp32 instead of VNNI-packing
        # when N < TILE_N (gemm.cpp), and weight_packed_linear detects that
        # (via the packed weight's dtype) and routes to its fp32/brgemm
        # fallback kernel -- no N/K alignment required in that regime.
        return True
    return k % 32 == 0 and n % 16 == 0


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    # skip for missing layers
    if layer.weight.is_meta:
        layer.cpu_linear = torch.nn.functional.linear
        return

    # Skip CPU GEMM dispatch for non-2D weights (e.g. MoE 3D expert weights).
    # These layers are handled by their own specialized methods.
    if layer.weight.ndim != 2:
        # this is not a linear layer
        # For now it should be a causal_conv1d op or MoE 3D expert weights
        # The C++ causal_conv1d kernels use VDPBF16PS (no AMX tiles), so the
        # VNNI weight prepack applies to any AVX-512BF16 CPU, not just AMX
        # (e.g. AMD Zen5/Turin).
        if torch.cpu._is_avx512_bf16_supported() and hasattr(
            ops, "causal_conv1d_weight_pack"
        ):
            # prepack conv weight
            unpacked = (
                layer.weight.view(
                    layer.weight.size(0),
                    layer.weight.size(2),
                )
                .contiguous()
                .clone()
            )
            # Stash the un-packed (dim, width) weight so the speculative-decode
            # GDN path (which uses torch conv, not the C++ kernel) can use it.
            layer._cpu_unpacked_conv_weight = unpacked
            layer.weight.data = ops.causal_conv1d_weight_pack(unpacked)
        return

    N, K = layer.weight.size()
    dtype = layer.weight.dtype

    # Zen CPU path: zentorch_linear_unary with optional eager weight prepacking.
    if current_platform.is_zen_cpu() and hasattr(
        torch.ops.zentorch, "zentorch_linear_unary"
    ):
        zen_weight = layer.weight.detach()
        is_prepacked = False

        if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
            torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
        ):
            zen_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                zen_weight
            )
            is_prepacked = True

        layer.cpu_linear = lambda x, weight, bias, _p=is_prepacked: (
            torch.ops.zentorch.zentorch_linear_unary(
                x, zen_weight, bias, is_weight_prepacked=_p
            )
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(
                torch.empty(0, dtype=dtype, device=layer.weight.device),
                requires_grad=False,
            )
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
            is_prepacked,
        )
        return

    # Small weights (e.g. MoE router/gate projections, where N is the expert
    # count rather than a hidden-size-scaled dimension) never reach oneDNN's
    # compute-bound regime, no matter how large the batch gets: SGL's lower
    # per-call dispatch overhead wins consistently across the full measured
    # M range. Larger dense projections (qkv/o_proj/gate_up/down/lm_head)
    # cross over to favoring oneDNN once batch size grows past decode-sized
    # M, so they keep using oneDNN below.
    if check_cpu_sgl_kernel(N, K, dtype):
        # For small size GEMM, packed_weight might be float32
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            layer.bias.data = layer.bias.to(torch.float32)
        layer.cpu_linear = lambda x, weight, bias: ops.weight_packed_linear_cpu(
            x,
            packed_weight,
            N,
            bias,
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(
                torch.empty(0, dtype=dtype, device=layer.weight.device),
                requires_grad=False,
            )
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using sgl-kernel weight_packed_linear"
        )
        return

    if (
        ops._supports_onednn
        and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
    ):
        try:
            origin_weight = layer.weight
            handler = ops.create_onednn_mm(origin_weight.t(), 32)
            layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias)
            if remove_weight:
                layer.weight = torch.nn.Parameter(
                    torch.empty(0, dtype=dtype, device=layer.weight.device),
                    requires_grad=False,
                )
            logger.debug_once("CPU unquantized GEMM dispatch: using oneDNN onednn_mm")
            return
        except RuntimeError as e:
            logger.warning_once(
                "Failed to create oneDNN linear, fallback to torch linear."
                f" Exception: {e}"
            )

    # fallback case
    layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
        x, weight, bias
    )
    logger.debug_once(
        "CPU unquantized GEMM dispatch: using torch.nn.functional.linear (fallback)"
    )


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)


def dispatch_unquantized_gemm(
    linear_backend: str = "auto",
) -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    elif not current_platform.is_cuda():
        return default_unquantized_gemm

    backend_spec = _FLASHINFER_BF16_BACKENDS.get(linear_backend)
    if backend_spec is None:
        return default_unquantized_gemm

    if not backend_spec.is_supported():
        logger.warning_once(
            "--linear-backend=%s requested FlashInfer mm_bf16 backend %r, "
            "but it is unavailable on the current hardware or environment; "
            "using automatic selection for unquantized linear layers.",
            linear_backend,
            backend_spec.flashinfer_backend,
        )
        return default_unquantized_gemm

    logger.info_once(
        "Using FlashInfer %s for eligible unquantized BF16 GEMMs.",
        backend_spec.flashinfer_backend,
    )
    return functools.partial(
        cuda_flashinfer_bf16_gemm,
        vllm_backend=linear_backend,
        pdl=current_platform.is_arch_support_pdl(),
    )
