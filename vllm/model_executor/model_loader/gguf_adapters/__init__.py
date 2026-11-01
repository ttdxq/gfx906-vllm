# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF weights-adapter registry.

Order matters: adapters are consulted first-to-last and the first match
wins, so family-specific adapters (MTP drafts, multimodal variants) must
be registered before their broader siblings.  Architectures without a
dedicated adapter fall back to the generic gguf-py-backed one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    BaseGGUFWeightsAdapter,
    GGUFWeight,
    build_name_map_via_gguf_py,
    get_mmproj_file,
)
from .generic import GenericGGUFWeightsAdapter
from .gemma4 import Gemma4GGUFAdapter, Gemma4MtpGGUFAdapter
from .qwen3_5 import Qwen35GGUFAdapter, Qwen35MtpGGUFAdapter

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_ADAPTER_REGISTRY: list[type[BaseGGUFWeightsAdapter]] = [
    Qwen35MtpGGUFAdapter,
    Qwen35GGUFAdapter,
    Gemma4MtpGGUFAdapter,
    Gemma4GGUFAdapter,
]


def register_gguf_weights_adapter(
    adapter_cls: type[BaseGGUFWeightsAdapter],
) -> type[BaseGGUFWeightsAdapter]:
    """Append an adapter; it is consulted after the built-in ones."""
    _ADAPTER_REGISTRY.append(adapter_cls)
    return adapter_cls


def get_weights_adapter(model_config: ModelConfig) -> BaseGGUFWeightsAdapter:
    """Return the first adapter matching the model, or the generic one."""
    for adapter_cls in _ADAPTER_REGISTRY:
        if adapter_cls.matches(model_config):
            return adapter_cls()
    return GenericGGUFWeightsAdapter()


__all__ = [
    "BaseGGUFWeightsAdapter",
    "GGUFWeight",
    "GenericGGUFWeightsAdapter",
    "Gemma4GGUFAdapter",
    "Gemma4MtpGGUFAdapter",
    "Qwen35GGUFAdapter",
    "Qwen35MtpGGUFAdapter",
    "Qwen4ExpGGUFAdapter",
    "build_name_map_via_gguf_py",
    "get_mmproj_file",
    "get_weights_adapter",
    "register_gguf_weights_adapter",
]
