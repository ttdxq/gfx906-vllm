# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.model_executor.model_loader import gguf_loader


@pytest.mark.parametrize(
    ("architecture", "has_vision_config", "expected"),
    [
        ("Qwen3_5ForCausalLM", True, None),
        ("Qwen3_5MoeForCausalLM", True, None),
        ("Qwen3_5ForConditionalGeneration", True, Path("mmproj.gguf")),
        ("Qwen3_5MoeForConditionalGeneration", True, Path("mmproj.gguf")),
        ("Gemma3ForConditionalGeneration", True, Path("mmproj.gguf")),
        ("Gemma3ForConditionalGeneration", False, None),
    ],
)
def test_get_mmproj_file_requires_multimodal_architecture(
    monkeypatch, architecture, has_vision_config, expected
):
    detected = Path("mmproj.gguf")
    monkeypatch.setattr(gguf_loader, "detect_gguf_multimodal", lambda _: detected)
    model_config = SimpleNamespace(
        architecture=architecture,
        model="model.gguf",
        hf_config=SimpleNamespace(
            vision_config=object() if has_vision_config else None
        ),
    )

    assert gguf_loader.GGUFModelLoader._get_mmproj_file(model_config) == expected
