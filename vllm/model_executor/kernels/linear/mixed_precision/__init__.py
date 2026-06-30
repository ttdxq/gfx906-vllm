# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)

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

__all__ = [
    "MPLinearKernel",
    "MPLinearLayerConfig",
    "ConchLinearKernel",
    "CPUWNA16LinearKernel",
    "Dynamic4bitLinearKernel",
    "ExllamaLinearKernel",
]

if AllSparkLinearKernel is not None:
    __all__.append("AllSparkLinearKernel")
if CutlassW4A8LinearKernel is not None:
    __all__.append("CutlassW4A8LinearKernel")
if MacheteLinearKernel is not None:
    __all__.append("MacheteLinearKernel")
if MarlinLinearKernel is not None:
    __all__.append("MarlinLinearKernel")
if XPUwNa16LinearKernel is not None:
    __all__.append("XPUwNa16LinearKernel")
