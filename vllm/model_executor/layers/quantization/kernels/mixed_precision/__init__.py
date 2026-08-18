# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.kernels.linear import (
    choose_mp_linear_kernel as _choose_mainline_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.allspark import (  # noqa: E501
    AllSparkLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.bitblas import (  # noqa: E501
    BitBLASLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.conch import (  # noqa: E501
    ConchLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.cutlass import (  # noqa: E501
    CutlassW4A8LinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.dynamic_4bit import (  # noqa: E501
    Dynamic4bitLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.exllama import (  # noqa: E501
    ExllamaLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.machete import (  # noqa: E501
    MacheteLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.marlin import (  # noqa: E501
    MarlinLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearKernel,
    MPLinearLayerConfig,
)

# in priority/performance order (when available)
_POSSIBLE_KERNELS: list[type[MPLinearKernel]] = [
    #CutlassW4A8LinearKernel,
    #MacheteLinearKernel,
    #AllSparkLinearKernel,
    #MarlinLinearKernel,
    #Dynamic4bitLinearKernel,
    #BitBLASLinearKernel,
    #ConchLinearKernel,
    ExllamaLinearKernel,
]


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
    return _choose_mainline_mp_linear_kernel(config, compute_capability)
