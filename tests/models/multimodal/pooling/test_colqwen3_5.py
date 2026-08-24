# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


def _vllm_config(contract=None, *, text_contract=None):
    hf_config = SimpleNamespace()
    text_config = SimpleNamespace(mamba_ssm_dtype=None)
    if contract is not None:
        hf_config.retrieval_attention_contract = contract
    if text_contract is not None:
        text_config.retrieval_attention_contract = text_contract
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=text_config,
    )
    cache_config = SimpleNamespace(mamba_ssm_cache_dtype="auto")
    return SimpleNamespace(model_config=model_config, cache_config=cache_config)


@pytest.mark.parametrize(
    ("contract", "expected_is_causal"),
    [("causal", True), ("bidirectional", False)],
)
def test_colqwen3_5_applies_declared_attention_contract(
    contract: str,
    expected_is_causal: bool,
) -> None:
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        ColQwen3_5Config,
    )

    assert MODELS_CONFIG_MAP["ColQwen3_5"] is ColQwen3_5Config
    vllm_config = _vllm_config(contract)
    ColQwen3_5Config.verify_and_update_config(vllm_config)
    assert vllm_config.model_config.hf_config.is_causal is expected_is_causal
    assert vllm_config.model_config.hf_text_config.is_causal is expected_is_causal


@pytest.mark.parametrize(
    ("contract", "text_contract"),
    [
        (None, None),
        ("unsupported", None),
        ("causal", "bidirectional"),
    ],
)
def test_colqwen3_5_rejects_invalid_attention_contract(
    contract: str | None,
    text_contract: str | None,
) -> None:
    from vllm.model_executor.models.config import ColQwen3_5Config

    with pytest.raises(ValueError, match="retrieval_attention_contract"):
        ColQwen3_5Config.verify_and_update_config(
            _vllm_config(contract, text_contract=text_contract)
        )


def test_encoder_only_attention_has_no_kv_cache_spec() -> None:
    from vllm.attention.backends.abstract import AttentionType
    from vllm.attention.layer import Attention

    attention = SimpleNamespace(attn_type=AttentionType.ENCODER_ONLY)
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    assert Attention.get_kv_cache_spec(attention, vllm_config) is None


def test_triton_backend_supports_encoder_only_attention() -> None:
    from vllm.attention.backends.abstract import AttentionType
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    assert TritonAttentionBackend.supports_attn_type(AttentionType.ENCODER_ONLY)


def test_bidirectional_contract_builds_encoder_only_attention(monkeypatch) -> None:
    from vllm.attention.backends.abstract import AttentionType
    from vllm.model_executor.models import qwen3_next

    captured = {}

    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            captured["attn_type"] = kwargs["attn_type"]

    monkeypatch.setattr(qwen3_next, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        qwen3_next, "QKVParallelLinear", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        qwen3_next, "RowParallelLinear", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        qwen3_next,
        "get_rope",
        lambda *args, **kwargs: SimpleNamespace(is_neox_style=False),
    )
    monkeypatch.setattr(
        qwen3_next, "Qwen3NextRMSNorm", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(qwen3_next, "Attention", FakeAttention)
    monkeypatch.setattr(
        qwen3_next,
        "current_platform",
        SimpleNamespace(
            get_device_capability=lambda: None,
            is_rocm=lambda: False,
            is_cuda=lambda: False,
        ),
    )

    config = SimpleNamespace(
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=4096,
        rope_parameters={},
        partial_rotary_factor=1.0,
        rms_norm_eps=1e-6,
        is_causal=False,
    )
    qwen3_next.Qwen3NextAttention(config)
    assert captured["attn_type"] == AttentionType.ENCODER_ONLY


def test_colqwen3_5_registry_import() -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    model_cls = ModelRegistry._try_load_model_cls("ColQwen3_5")
    assert model_cls is not None
    assert model_cls.__name__ == "ColQwen3_5Model"
    assert model_cls.default_pooling_type == "ALL"
    assert model_cls.score_type == "late-interaction"
