# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ColQwen3.5 late-interaction retrieval model."""

from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn
from transformers.models.qwen3_vl import Qwen3VLProcessor

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY

from .interfaces_base import default_pooling_type
from .qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from .utils import AutoWeightsLoader, WeightsMapper


class ColQwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    """Use the standard Qwen3-VL processor for remote ColQwen configs."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    @property
    def _supports_video(self) -> bool:
        return hasattr(self.get_hf_processor(), "video_processor")

    def get_video_processor(self, **kwargs: object):
        if not self._supports_video:
            raise AttributeError(
                f"The processor for {self.ctx.model_config.model} does not "
                "support video inputs (no video_processor attribute)."
            )
        return self.get_hf_processor(**kwargs).video_processor  # type: ignore[attr-defined]

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {"image": None}
        if self._supports_video:
            limits["video"] = None
        return limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        limits = {"image": self.get_max_image_tokens()}
        if self._supports_video:
            limits["video"] = self.get_max_video_tokens(seq_len, mm_counts)
        return limits


@default_pooling_type("ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=ColQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class ColQwen3_5Model(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 backbone with a ColBERT-style per-token projection head."""

    is_pooling_model = True
    score_type = "late-interaction"

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and hasattr(config, "text_config"):
            hidden_size = config.text_config.hidden_size
        if hidden_size is None:
            raise ValueError(
                "Unable to determine text hidden size from config. Expected "
                "'hidden_size' or 'text_config.hidden_size'."
            )

        self.embed_dim: int = (
            getattr(config, "embed_dim", None)
            or getattr(config, "dims", None)
            or getattr(config, "dim", None)
            or getattr(config, "projection_dim", None)
            or getattr(config, "colbert_dim", None)
            or 128
        )
        self.custom_text_proj = nn.Linear(
            hidden_size,
            self.embed_dim,
            bias=True,
            dtype=vllm_config.model_config.head_dtype,
        )
        nn.init.zeros_(self.custom_text_proj.bias)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = Pooler.for_token_embed(pooler_config)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if not isinstance(hidden_states, torch.Tensor):
            return hidden_states  # type: ignore[return-value]

        return self.custom_text_proj(
            hidden_states.to(self.custom_text_proj.weight.dtype)
        )

    _PROJ_LAYER_NAMES = {
        "custom_text_proj",
        "embedding_proj_layer",
    }

    def _is_proj_weight(self, name: str) -> bool:
        return any(proj_name in name for proj_name in self._PROJ_LAYER_NAMES)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        proj_weights: list[tuple[str, torch.Tensor]] = []
        model_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            target = proj_weights if self._is_proj_weight(name) else model_weights
            target.append((name, weight))

        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        loaded = loader.load_weights(model_weights, mapper=self.hf_to_vllm_mapper)

        for name, weight in proj_weights:
            param_name = name.rsplit(".", 1)[-1]
            param = getattr(self.custom_text_proj, param_name, None)
            if param is not None:
                default_weight_loader(
                    param,
                    weight.to(device=param.device, dtype=param.dtype),
                )
                loaded.add(f"custom_text_proj.{param_name}")

        return loaded
