# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .hasher import MultiModalHasher
from .inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
    NestedTensors,
)

try:
    from .inputs import MultiModalDataDict  # type: ignore[attr-defined]
except ImportError:
    from vllm.inputs.llm import MultiModalDataDict  # noqa: F401
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global [`MultiModalRegistry`][vllm.multimodal.registry.MultiModalRegistry]
is used by model runners to dispatch data processing according to the target
model.

Info:
    [mm_processing](../../../design/mm_processing.md)
"""

__all__ = [
    "MultiModalDataDict",
    "MultiModalUUIDDict",
    "BatchedTensorInputs",
    "MultiModalHasher",
    "MultiModalKwargsItems",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
