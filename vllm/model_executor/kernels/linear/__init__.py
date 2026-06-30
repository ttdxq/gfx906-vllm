# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
This module re-exports linear kernel implementations to provide a
stable import interface during an ongoing reorganization. Upcoming
PRs will remove the scaled_mm and mixed_precision subdirectories
and reorganize kernels by provider (aiter, cutlass, flashinfer, etc.)
rather than by precision type. By centralizing exports here, we
minimize the need to update imports across other modules when the
internal structure changes. If you are adding a new kernel selector
or kernel implementation, add it to this __init__.py to maintain
import stability.
"""

import os
from typing import TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mixed_precision import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.conch import (
    ConchLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.cpu import (
    CPUWNA16LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
    Dynamic4bitLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
    ExllamaLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
    ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)

try:
    from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
        CPUInt8ScaledMMLinearKernel,
    )
except ImportError:
    CPUInt8ScaledMMLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
    )
except ImportError:
    ChannelWiseTorchFP8ScaledMMLinearKernel = None
    PerTensorTorchFP8ScaledMMLinearKernel = None
    RowWiseTorchFP8ScaledMMLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
        ROCmFP8ScaledMMLinearKernel,
    )
except ImportError:
    ROCmFP8ScaledMMLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )
except ImportError:
    TritonInt8ScaledMMLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
        AiterInt8ScaledMMLinearKernel,
    )
except ImportError:
    AiterInt8ScaledMMLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassFP8ScaledMMLinearKernel,
        CutlassInt8ScaledMMLinearKernel,
    )
except ImportError:
    CutlassFP8ScaledMMLinearKernel = None
    CutlassInt8ScaledMMLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        FlashInferFP8ScaledMMLinearKernel,
    )
except ImportError:
    FlashInferFP8ScaledMMLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
        XPUFP8ScaledMMLinearKernel,
    )
except ImportError:
    XPUFP8ScaledMMLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.mixed_precision.allspark import (
        AllSparkLinearKernel,
    )
except ImportError:
    AllSparkLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.mixed_precision.cutlass import (
        CutlassW4A8LinearKernel,
    )
except ImportError:
    CutlassW4A8LinearKernel = None

try:
    from vllm.model_executor.kernels.linear.mixed_precision.machete import (
        MacheteLinearKernel,
    )
except ImportError:
    MacheteLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )
except ImportError:
    MarlinLinearKernel = None

try:
    from vllm.model_executor.kernels.linear.mixed_precision.xpu import (
        XPUwNa16LinearKernel,
    )
except ImportError:
    XPUwNa16LinearKernel = None

try:
    from vllm.model_executor.kernels.linear.mixed_precision.torch_wna16 import (
        TorchWNA16LinearKernel,
    )
except ImportError:
    TorchWNA16LinearKernel = None

# in priority/performance order (when available)
_POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[type[Int8ScaledMMLinearKernel]]] = {
    PlatformEnum.CPU: [],
    PlatformEnum.CUDA: [],
    PlatformEnum.ROCM: [],
}

if CPUInt8ScaledMMLinearKernel is not None:
    _POSSIBLE_INT8_KERNELS[PlatformEnum.CPU].append(CPUInt8ScaledMMLinearKernel)
if TritonInt8ScaledMMLinearKernel is not None:
    _POSSIBLE_INT8_KERNELS[PlatformEnum.CUDA].append(TritonInt8ScaledMMLinearKernel)
    _POSSIBLE_INT8_KERNELS[PlatformEnum.ROCM].append(TritonInt8ScaledMMLinearKernel)

if CutlassInt8ScaledMMLinearKernel is not None:
    _POSSIBLE_INT8_KERNELS[PlatformEnum.CUDA].insert(0, CutlassInt8ScaledMMLinearKernel)
if AiterInt8ScaledMMLinearKernel is not None:
    _POSSIBLE_INT8_KERNELS[PlatformEnum.ROCM].insert(0, AiterInt8ScaledMMLinearKernel)

# in priority/performance order (when available)
_POSSIBLE_FP8_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [],
    PlatformEnum.ROCM: [],
    PlatformEnum.CPU: [],
    PlatformEnum.XPU: [],
}

if PerTensorTorchFP8ScaledMMLinearKernel is not None:
    _POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA].append(
        PerTensorTorchFP8ScaledMMLinearKernel
    )
    _POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM].append(
        PerTensorTorchFP8ScaledMMLinearKernel
    )
    _POSSIBLE_FP8_KERNELS[PlatformEnum.CPU].append(
        PerTensorTorchFP8ScaledMMLinearKernel
    )
if ChannelWiseTorchFP8ScaledMMLinearKernel is not None:
    _POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA].append(
        ChannelWiseTorchFP8ScaledMMLinearKernel
    )
    _POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM].append(
        ChannelWiseTorchFP8ScaledMMLinearKernel
    )
    _POSSIBLE_FP8_KERNELS[PlatformEnum.CPU].append(
        ChannelWiseTorchFP8ScaledMMLinearKernel
    )
if ROCmFP8ScaledMMLinearKernel is not None:
    _POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM].insert(0, ROCmFP8ScaledMMLinearKernel)
if RowWiseTorchFP8ScaledMMLinearKernel is not None:
    insert_at = (
        1
        if ROCmFP8ScaledMMLinearKernel is not None
        else len(_POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM])
    )
    _POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM].insert(
        insert_at, RowWiseTorchFP8ScaledMMLinearKernel
    )

if FlashInferFP8ScaledMMLinearKernel is not None:
    _POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA].insert(
        0, FlashInferFP8ScaledMMLinearKernel
    )
if CutlassFP8ScaledMMLinearKernel is not None:
    cuda_fp8_index = 1 if FlashInferFP8ScaledMMLinearKernel is not None else 0
    _POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA].insert(
        cuda_fp8_index, CutlassFP8ScaledMMLinearKernel
    )
if XPUFP8ScaledMMLinearKernel is not None:
    _POSSIBLE_FP8_KERNELS[PlatformEnum.XPU].append(XPUFP8ScaledMMLinearKernel)

# in priority/performance order (when available)
_POSSIBLE_KERNELS: dict[PlatformEnum, list[type[MPLinearKernel]]] = {
    PlatformEnum.CUDA: [
        ConchLinearKernel,
        ExllamaLinearKernel,
    ],
    PlatformEnum.ROCM: [
        ConchLinearKernel,
        ExllamaLinearKernel,
    ],
    PlatformEnum.XPU: [],
    PlatformEnum.CPU: [
        Dynamic4bitLinearKernel,
        CPUWNA16LinearKernel,
    ],
}

if CutlassW4A8LinearKernel is not None:
    _POSSIBLE_KERNELS[PlatformEnum.CUDA].insert(0, CutlassW4A8LinearKernel)
if MacheteLinearKernel is not None:
    _POSSIBLE_KERNELS[PlatformEnum.CUDA].insert(
        1 if CutlassW4A8LinearKernel is not None else 0, MacheteLinearKernel
    )
if AllSparkLinearKernel is not None:
    cuda_index = (
        2
        if MacheteLinearKernel is not None and CutlassW4A8LinearKernel is not None
        else (
            1
            if (MacheteLinearKernel is not None or CutlassW4A8LinearKernel is not None)
            else 0
        )
    )
    _POSSIBLE_KERNELS[PlatformEnum.CUDA].insert(cuda_index, AllSparkLinearKernel)
if MarlinLinearKernel is not None:
    _POSSIBLE_KERNELS[PlatformEnum.CUDA].insert(
        len(_POSSIBLE_KERNELS[PlatformEnum.CUDA]) - 2, MarlinLinearKernel
    )
if XPUwNa16LinearKernel is not None:
    _POSSIBLE_KERNELS[PlatformEnum.XPU].append(XPUwNa16LinearKernel)
if TorchWNA16LinearKernel is not None:
    _POSSIBLE_KERNELS[PlatformEnum.ROCM].append(TorchWNA16LinearKernel)

_KernelT = TypeVar("_KernelT", bound=ScaledMMLinearKernel)
_KernelConfigT = TypeVar("_KernelConfigT", bound=ScaledMMLinearLayerConfig)


def is_supported_and_can_implement_kernel(
    kernel: type[_KernelT], config: _KernelConfigT, compute_capability: int | None
) -> tuple[bool, str]:
    # TODO: Fetch `VLLM_DISABLED_KERNELS` from vllm.envs instead.
    if kernel.__name__ in os.environ.get("VLLM_DISABLED_KERNELS", "").split(","):
        return False, f" {kernel.__name__} is disabled by environment variable"

    if compute_capability is None:
        _cc = current_platform.get_device_capability()
        if _cc is not None:
            compute_capability = _cc[0] * 10 + _cc[1]

    is_supported, failure_reason = kernel.is_supported(compute_capability)
    if not is_supported:
        return False, f"{kernel.__name__} {failure_reason}."

    can_implement, failure_reason = kernel.can_implement(config)
    if not can_implement:
        return (
            False,
            f"{kernel.__name__} {failure_reason}.",
        )

    return True, ""


def choose_scaled_mm_linear_kernel(
    config: _KernelConfigT,
    possible_kernels: dict[PlatformEnum, list[type[_KernelT]]],
    compute_capability: int | None = None,
    force_kernel: type[_KernelT] | None = None,
) -> type[_KernelT]:
    """
    Choose a _KernelT that can implement the given config for the
    given compute capability. Attempts to choose the best kernel in terms of
    performance.

    Args:
        config (_KernelConfigT): Description of the linear layer
            to be implemented.
        possible_kernels (dict[PlatformEnum, list[_KernelT]]): A
            dictionary of platforms and their list of possible kernels.
        compute_capability (Optional[int], optional): The compute capability of
            the target device, if None uses `current_platform` to get the
            compute capability. Defaults to None.
        force_kernel (Optional[type[_KernelT]]): An Optional forced kernel to override
            the possible_kernels if it can be implemented. If None, it will only try the
            possible kernels.

    Raises:
        ValueError: If no kernel can implement the given config.

    Returns:
        _KernelT: Chosen kernel.
    """

    failure_reason_list = []

    if force_kernel is not None:
        can_implement, failure_reason = is_supported_and_can_implement_kernel(
            force_kernel, config, compute_capability
        )
        if can_implement:
            return force_kernel

        logger.info_once(
            "Tried to force %s, but the kernel couldn't be implemented",
            force_kernel.__name__,
            scope="global",
        )

    for kernel in possible_kernels[current_platform._enum]:
        is_supported_and_can_implement, failure_reason = (
            is_supported_and_can_implement_kernel(kernel, config, compute_capability)
        )
        if is_supported_and_can_implement:
            return kernel
        failure_reason_list.append(failure_reason)

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "ScaledMM linear layer. Reasons: \n" + "\n".join(failure_reason_list)
    )


def init_fp8_linear_kernel(
    activation_quant_key: QuantKey,
    weight_quant_key: QuantKey,
    out_dtype: torch.dtype,
    force_kernel: type[FP8ScaledMMLinearKernel] | None = None,
    module_name: str | None = None,
) -> FP8ScaledMMLinearKernel:
    scaled_mm_linear_kernel_config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        out_dtype=out_dtype,
    )

    kernel_type = choose_scaled_mm_linear_kernel(
        scaled_mm_linear_kernel_config, _POSSIBLE_FP8_KERNELS, force_kernel=force_kernel
    )

    if module_name:
        logger.info_once(
            "Selected %s for %s",
            kernel_type.__name__,
            module_name,
            scope="global",
        )

    return kernel_type(
        scaled_mm_linear_kernel_config,
        layer_param_names=["weight", "weight_scale", "input_scale", "input_scale_ub"],
    )


def init_int8_linear_kernel(
    is_channelwise: bool,
    is_static_input_scheme: bool,
    input_symmetric: bool,
    module_name: str,
) -> Int8ScaledMMLinearKernel:
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=is_channelwise,
        is_static_input_scheme=is_static_input_scheme,
        input_symmetric=input_symmetric,
    )

    kernel_type = choose_scaled_mm_linear_kernel(
        config,
        _POSSIBLE_INT8_KERNELS,
    )

    logger.info_once(
        "Selected %s for %s",
        kernel_type.__name__,
        module_name,
        scope="global",
    )

    return kernel_type(
        config,
        layer_param_names=[
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        ],
    )


def choose_mp_linear_kernel(
    config: MPLinearLayerConfig, compute_capability: int | None = None
) -> type[MPLinearKernel]:
    """
    Choose an MPLinearKernel that can implement the given config for the given
     compute capability. Attempts to choose the best kernel in terms of
     performance.

    Args:
        config (MPLinearLayerConfig): Description of the linear layer to be
            implemented.
        compute_capability (Optional[int], optional): The compute capability of
            the target device, if None uses `current_platform` to get
            the compute capability. Defaults to None.

    Raises:
        ValueError: If no kernel can implement the given config.

    Returns:
        type[MPLinearKernel]: Chosen kernel.
    """
    if compute_capability is None:
        if current_platform is None:
            raise ValueError("Cannot determine compute capability")
        _cc = current_platform.get_device_capability()
        if _cc is not None:
            compute_capability = _cc[0] * 10 + _cc[1]

    failure_reasons = []
    for kernel in _POSSIBLE_KERNELS[current_platform._enum]:
        if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
            failure_reasons.append(
                f" {kernel.__name__} disabled by environment variable"
            )
            continue
        if (
            compute_capability is not None
            and kernel.get_min_capability() > compute_capability
        ):
            failure_reasons.append(
                f"{kernel.__name__} requires capability "
                f"{kernel.get_min_capability()}, current compute "
                f" capability is {compute_capability}"
            )
            continue

        can_implement, failure_reason = kernel.can_implement(config)
        if can_implement:
            return kernel
        else:
            failure_reasons.append(
                f" {kernel.__name__} cannot implement due to: {failure_reason}"
            )

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "WNA16 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )


__all__ = [
    "init_fp8_linear_kernel",
    "init_int8_linear_kernel",
    "choose_mp_linear_kernel",
    "FP8ScaledMMLinearKernel",
    "Int8ScaledMMLinearKernel",
    "ScaledMMLinearKernel",
    "FP8ScaledMMLinearLayerConfig",
    "Int8ScaledMMLinearLayerConfig",
    "ScaledMMLinearLayerConfig",
    "CPUInt8ScaledMMLinearKernel",
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
    "PerTensorTorchFP8ScaledMMLinearKernel",
    "RowWiseTorchFP8ScaledMMLinearKernel",
    "ROCmFP8ScaledMMLinearKernel",
    "TritonInt8ScaledMMLinearKernel",
    "MPLinearKernel",
    "MPLinearLayerConfig",
    "ConchLinearKernel",
    "CPUWNA16LinearKernel",
    "Dynamic4bitLinearKernel",
    "ExllamaLinearKernel",
]

if AllSparkLinearKernel is not None:
    __all__.append("AllSparkLinearKernel")
if AiterInt8ScaledMMLinearKernel is not None:
    __all__.append("AiterInt8ScaledMMLinearKernel")
if CutlassW4A8LinearKernel is not None:
    __all__.append("CutlassW4A8LinearKernel")
if CutlassFP8ScaledMMLinearKernel is not None:
    __all__.append("CutlassFP8ScaledMMLinearKernel")
if CutlassInt8ScaledMMLinearKernel is not None:
    __all__.append("CutlassInt8ScaledMMLinearKernel")
if FlashInferFP8ScaledMMLinearKernel is not None:
    __all__.append("FlashInferFP8ScaledMMLinearKernel")
if MacheteLinearKernel is not None:
    __all__.append("MacheteLinearKernel")
if MarlinLinearKernel is not None:
    __all__.append("MarlinLinearKernel")
if XPUFP8ScaledMMLinearKernel is not None:
    __all__.append("XPUFP8ScaledMMLinearKernel")
if XPUwNa16LinearKernel is not None:
    __all__.append("XPUwNa16LinearKernel")
if TorchWNA16LinearKernel is not None:
    __all__.append("TorchWNA16LinearKernel")
