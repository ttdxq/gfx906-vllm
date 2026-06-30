# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
    CPUInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
    ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
    ScaledMMLinearLayerConfig,
)

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

__all__ = [
    "FP8ScaledMMLinearKernel",
    "FP8ScaledMMLinearLayerConfig",
    "Int8ScaledMMLinearKernel",
    "Int8ScaledMMLinearLayerConfig",
    "ScaledMMLinearKernel",
    "ScaledMMLinearLayerConfig",
    "CPUInt8ScaledMMLinearKernel",
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
    "PerTensorTorchFP8ScaledMMLinearKernel",
    "RowWiseTorchFP8ScaledMMLinearKernel",
    "ROCmFP8ScaledMMLinearKernel",
]

if AiterInt8ScaledMMLinearKernel is not None:
    __all__.append("AiterInt8ScaledMMLinearKernel")
if CutlassFP8ScaledMMLinearKernel is not None:
    __all__.append("CutlassFP8ScaledMMLinearKernel")
if CutlassInt8ScaledMMLinearKernel is not None:
    __all__.append("CutlassInt8ScaledMMLinearKernel")
if FlashInferFP8ScaledMMLinearKernel is not None:
    __all__.append("FlashInferFP8ScaledMMLinearKernel")
if TritonInt8ScaledMMLinearKernel is not None:
    __all__.append("TritonInt8ScaledMMLinearKernel")
