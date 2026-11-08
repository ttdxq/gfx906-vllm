# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF weights-adapter interface.

Each supported GGUF model family implements one adapter that owns its
GGUF→HF tensor-name mapping (and any family-specific weight transforms).
The generic fallback (``generic.py``) serves every architecture whose HF
names are covered by gguf-py's tensor-name tables.  Adapters keep the
model loader free of per-family branches; adding GGUF support for a new
family means adding one module here and registering it in ``__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import gguf
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from vllm.logger import init_logger
from vllm.transformers_utils.gguf_utils import detect_gguf_multimodal

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.config import ModelConfig

logger = init_logger(__name__)

GGUFWeight = tuple[str, torch.Tensor]

# Architectures whose GGUF directories may carry an mmproj companion file.
_MULTIMODAL_GGUF_ARCHITECTURES = frozenset(
    {
        "Gemma3ForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


def get_mmproj_file(
    model_config: ModelConfig, model_name_or_path: str | None = None
) -> Path | None:
    if model_config.architecture not in _MULTIMODAL_GGUF_ARCHITECTURES:
        return None
    if getattr(model_config.hf_config, "vision_config", None) is None:
        return None
    return detect_gguf_multimodal(model_name_or_path or model_config.model)


def build_name_map_via_gguf_py(
    model_config: ModelConfig,
    *,
    gguf_model_type: str,
    name_map: dict[str, str] | None = None,
    hf_name_preprocess: Callable[[str], str] | None = None,
    base_name_overrides: dict[str, str] | None = None,
    resolve_missing: Callable[[str, str, str], str | None] | None = None,
    dummy_config_override: Callable[[PretrainedConfig, bool], PretrainedConfig]
    | None = None,
) -> dict[str, str]:
    """Cross-validate HF parameter names against gguf-py's tensor tables.

    Builds a dummy model from the config, then maps every parameter in its
    state dict to the GGUF tensor name gguf-py expects.  Family adapters
    customize the process through the hook arguments; anything still
    unmapped at the end is fatal.
    """
    config = model_config.hf_config
    text_config = config.get_text_config()
    is_multimodal = get_mmproj_file(model_config) is not None

    arch = None
    for key, value in gguf.MODEL_ARCH_NAMES.items():
        if value == gguf_model_type:
            arch = key
            break
    if arch is None:
        raise RuntimeError(f"Unknown gguf model_type: {gguf_model_type}")
    text_num_layers = text_config.num_hidden_layers
    text_name_map = gguf.get_tensor_name_map(arch, text_num_layers)

    if is_multimodal:
        vision_num_layers = getattr(
            config.vision_config,
            "num_hidden_layers",
            getattr(config.vision_config, "depth", None),
        )
        if vision_num_layers is None:
            raise RuntimeError("GGUF vision config does not define its layer count")
        vision_name_map = gguf.get_tensor_name_map(
            gguf.MODEL_ARCH.MMPROJ, vision_num_layers
        )
    else:
        vision_name_map = None

    # Create dummy model to extract parameter names.  For multimodal models
    # this covers language + vision + projector params; for text-only models
    # just the language model.
    auto_cls = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
    dummy_config = config
    if dummy_config_override is not None:
        dummy_config = dummy_config_override(config, is_multimodal)
    with torch.device("meta"):
        dummy_model = auto_cls.from_config(
            dummy_config, trust_remote_code=model_config.trust_remote_code
        )

    state_dict = dummy_model.state_dict()
    if hf_checkpoint_map := getattr(
        dummy_model, "_checkpoint_conversion_mapping", None
    ):

        def revert_hf_rename(name: str) -> str:
            for original_name, hf_name in hf_checkpoint_map.items():
                if hf_name in name:
                    name = name.replace(hf_name, original_name).lstrip("^")
            return name

        state_dict = {
            revert_hf_rename(name): tensor for name, tensor in state_dict.items()
        }

    def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
        """Map one HF parameter name to its GGUF tensor name, or None."""
        if hf_name_preprocess is not None:
            hf_name = hf_name_preprocess(hf_name)
        elif hf_name.startswith("language_model."):
            # gguf-py tensor mappings expect parameter names without this
            # prefix.  The 'model.' prefix is KEPT: gguf-py expects it.
            hf_name = hf_name[len("language_model.") :]

        if hf_name.endswith((".weight", ".bias")):
            base_name, suffix = hf_name.rsplit(".", 1)
        else:
            base_name, suffix = hf_name, ""
            # Gemma3 naming: parameters end with '_weight' instead of '.weight'
            if base_name.endswith("_weight"):
                base_name = base_name[: -len("_weight")]
                suffix = "weight"

        if base_name_overrides:
            base_name = base_name_overrides.get(base_name, base_name)

        gguf_name = None
        # Priority 1: vision/projector parameters for multimodal models
        if vision_name_map is not None:
            gguf_name = vision_name_map.get_name(base_name)
        # Priority 2: text backbone parameters
        if gguf_name is None:
            gguf_name = text_name_map.get_name(base_name)

        if gguf_name is None:
            if resolve_missing is not None:
                return resolve_missing(hf_name, base_name, suffix)
            return None

        return gguf_name + "." + suffix if suffix else gguf_name

    if name_map is None:
        name_map = {}

    unmapped_params = []
    for hf_name in state_dict:
        gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)
        if gguf_name_with_suffix is not None:
            name_map[gguf_name_with_suffix] = hf_name
            logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
        elif hf_name not in name_map.values():
            # Parameter not covered by the pre-seeded entries either
            unmapped_params.append(hf_name)

    if unmapped_params:
        raise RuntimeError(
            f"Failed to map GGUF parameters "
            f"({len(unmapped_params)}): "
            f"{unmapped_params}"
        )
    return name_map


class BaseGGUFWeightsAdapter(ABC):
    """Model-family-specific GGUF name mapping and tensor transforms."""

    @classmethod
    @abstractmethod
    def matches(cls, model_config: ModelConfig) -> bool:
        """Whether this adapter handles the model's architecture."""

    @abstractmethod
    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        """Return a GGUF tensor name → HF parameter name map."""

    def transform_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Transform main-file weights after name mapping."""
        del model_config
        yield from weights

    def transform_mmproj_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
        gguf_to_hf_name_map: dict[str, str],
        mmproj_file: Path,
    ) -> Iterable[GGUFWeight]:
        """Transform mmproj (vision/projector) weights after name mapping."""
        del model_config, gguf_to_hf_name_map, mmproj_file
        yield from weights
