# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal, get_args

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "bitblas",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "gptq_bitblas",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "hqq",
    "experts_int8",
    "ipex",
    "quark",
    "moe_wna16",
    "torchao",
    "auto-round",
    "rtn",
    "inc",
    "mxfp4",
    "petit_nvfp4",
    "cpu_gptq",
    "cpu_awq",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


class _LazyQuarkConfig(QuantizationConfig):
    """Quark config proxy that avoids importing optional aiter dependencies.

    ModelConfig probes every quantization method for override support. Importing
    Quark during that probe can initialize ROCm aiter even when the model uses a
    different quantization format, so defer the real import until Quark is
    actually selected and instantiated from config.
    """

    def get_name(self) -> QuantizationMethods:
        return "quark"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QuantizationConfig:
        from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

        return QuarkConfig.from_config(config)

    def get_quant_method(self, layer, prefix):
        raise RuntimeError("Lazy Quark config should not be used at runtime")


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.warning(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]

    if quantization == "quark":
        return _LazyQuarkConfig

    from vllm.platforms import current_platform

    from .auto_round import AutoRoundConfig
    from .awq import AWQConfig
    from .bitblas import BitBLASConfig
    from .bitsandbytes import BitsAndBytesConfig
    from .compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from .cpu_wna16 import CPUAWQConfig, CPUGPTQConfig
    from .deepspeedfp import DeepSpeedFPConfig
    from .experts_int8 import ExpertsInt8Config
    from .fbgemm_fp8 import FBGEMMFp8Config
    from .fp8 import Fp8Config
    from .fp_quant import FPQuantConfig
    from .gguf import GGUFConfig
    from .gptq import GPTQConfig
    from .gptq_bitblas import GPTQBitBLASConfig
    from .gptq_marlin import GPTQMarlinConfig
    from .gptq_marlin_24 import GPTQMarlin24Config
    from .hqq_marlin import HQQMarlinConfig
    from .inc import INCConfig
    from .ipex_quant import IPEXConfig
    from .modelopt import ModelOptFp8Config, ModelOptNvFp4Config
    from .moe_wna16 import MoeWNA16Config
    from .mxfp4 import Mxfp4Config
    from .petit import PetitNvFp4Config
    from .ptpc_fp8 import PTPCFp8Config
    from .rtn import RTNConfig
    from .torchao import TorchAOConfig
    from .tpu_int8 import Int8TpuConfig

    if current_platform.is_cuda():
        from .awq_marlin import AWQMarlinConfig
    else:

        class AWQMarlinConfig(QuantizationConfig):
            @classmethod
            def get_name(cls) -> QuantizationMethods:
                return "awq_marlin"

            @classmethod
            def get_supported_act_dtypes(cls):
                return []

            @classmethod
            def get_min_capability(cls) -> int:
                return 0

            @staticmethod
            def get_config_filenames() -> list[str]:
                return []

            @classmethod
            def from_config(cls, config: dict):
                raise RuntimeError("awq_marlin is unavailable on non-CUDA platforms")

            def get_quant_method(self, layer, prefix):
                return None

    method_to_config: dict[str, type[QuantizationConfig]] = {
        "awq": AWQConfig,
        "deepspeedfp": DeepSpeedFPConfig,
        "tpu_int8": Int8TpuConfig,
        "fp8": Fp8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "fp_quant": FPQuantConfig,
        "modelopt": ModelOptFp8Config,
        "modelopt_fp4": ModelOptNvFp4Config,
        "bitblas": BitBLASConfig,
        "gguf": GGUFConfig,
        "gptq_marlin_24": GPTQMarlin24Config,
        "gptq_marlin": GPTQMarlinConfig,
        "gptq_bitblas": GPTQBitBLASConfig,
        "awq_marlin": AWQMarlinConfig,
        "gptq": GPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "ptpc_fp8": PTPCFp8Config,
        "hqq": HQQMarlinConfig,
        "experts_int8": ExpertsInt8Config,
        "ipex": IPEXConfig,
        "moe_wna16": MoeWNA16Config,
        "torchao": TorchAOConfig,
        "auto-round": AutoRoundConfig,
        "rtn": RTNConfig,
        "inc": INCConfig,
        "mxfp4": Mxfp4Config,
        "petit_nvfp4": PetitNvFp4Config,
        "cpu_gptq": CPUGPTQConfig,
        "cpu_awq": CPUAWQConfig,
    }

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
