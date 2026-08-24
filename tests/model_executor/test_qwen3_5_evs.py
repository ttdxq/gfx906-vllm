# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch import nn

from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import PPMissingLayer


class _MultiModalConfig:
    def __init__(
        self,
        *,
        language_model_only: bool = False,
        image_limit: int = 1,
        video_limit: int = 1,
        video_pruning_rate: float | None = 0.75,
    ) -> None:
        self.language_model_only = language_model_only
        self.limit_per_prompt = {
            "image": SimpleNamespace(count=image_limit),
            "video": SimpleNamespace(count=video_limit),
        }
        self.video_pruning_rate = video_pruning_rate
        self.mm_encoder_tp_mode = "weights"

    def get_limit_per_prompt(self, modality: str) -> int:
        return self.limit_per_prompt[modality].count

    def is_multimodal_pruning_enabled(self) -> bool:
        return bool(self.video_pruning_rate and self.video_pruning_rate > 0)


class _DummyVision(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class _DummyLanguageModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.make_empty_intermediate_tensors = None


def _make_vllm_config(mm_config: _MultiModalConfig):
    vision_config = SimpleNamespace(out_hidden_size=32)
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(vision_config=vision_config),
        multimodal_config=mm_config,
    )
    return SimpleNamespace(model_config=model_config, quant_config=None)


@pytest.mark.parametrize(
    ("model_cls", "language_model_name"),
    [
        (Qwen3_5ForConditionalGeneration, "Qwen3_5ForCausalLM"),
        (Qwen3_5MoeForConditionalGeneration, "Qwen3_5MoeForCausalLM"),
    ],
)
def test_qwen35_evs_initialization(monkeypatch, model_cls, language_model_name):
    tokenizer = object()

    monkeypatch.setattr(qwen3_5, "Qwen3_VisionTransformer", _DummyVision)
    monkeypatch.setattr(qwen3_5, language_model_name, _DummyLanguageModel)
    tokenizer_mock = Mock(return_value=tokenizer)
    monkeypatch.setattr(qwen3_5, "cached_tokenizer_from_config", tokenizer_mock)
    monkeypatch.setattr(
        model_cls,
        "_mark_tower_model",
        lambda self, *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        model_cls,
        "_mark_language_model",
        lambda self, *args, **kwargs: nullcontext(),
    )
    if model_cls is Qwen3_5MoeForConditionalGeneration:
        monkeypatch.setattr(model_cls, "set_moe_parameters", lambda self: None)

    config = _make_vllm_config(_MultiModalConfig())
    model = model_cls(vllm_config=config)

    assert model.is_multimodal_pruning_enabled
    assert model.video_pruning_rate == 0.75
    assert isinstance(model.visual, _DummyVision)
    assert model._tokenizer is tokenizer
    assert model.visual_dim == 32
    tokenizer_mock.assert_called_once_with(config.model_config)


@pytest.mark.parametrize(
    "mm_config",
    [
        _MultiModalConfig(language_model_only=True),
        _MultiModalConfig(image_limit=0, video_limit=0),
    ],
)
def test_qwen35_evs_disabled_with_multimodal_boundaries(monkeypatch, mm_config):
    monkeypatch.setattr(qwen3_5, "Qwen3_5ForCausalLM", _DummyLanguageModel)
    tokenizer_mock = Mock()
    monkeypatch.setattr(qwen3_5, "cached_tokenizer_from_config", tokenizer_mock)
    monkeypatch.setattr(
        Qwen3_5ForConditionalGeneration,
        "_mark_language_model",
        lambda self, *args, **kwargs: nullcontext(),
    )

    model = Qwen3_5ForConditionalGeneration(vllm_config=_make_vllm_config(mm_config))

    assert not model.is_multimodal_pruning_enabled
    assert model.video_pruning_rate is None
    assert model.multimodal_config is None
    assert isinstance(model.visual, PPMissingLayer)
    assert not hasattr(model, "_tokenizer")
    tokenizer_mock.assert_not_called()


def test_qwen35_inherits_qwen3vl_mrope_recomputation():
    assert (
        Qwen3_5ForConditionalGeneration.recompute_mrope_positions
        is Qwen3VLForConditionalGeneration.recompute_mrope_positions
    )
