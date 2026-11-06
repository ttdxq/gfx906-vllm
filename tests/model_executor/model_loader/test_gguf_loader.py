# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader import gguf_adapters, gguf_loader
from vllm.model_executor.model_loader.gguf_adapters.qwen4_exp import (
    Qwen4ExpGGUFAdapter,
    Qwen4ExpMTPGGUFAdapter,
)
from vllm.model_executor.layers.quantization.gguf import (
    GGUFConfig,
    is_layer_skipped_gguf,
)
from vllm.models.qwen4_exp.amd.mtp import (
    Qwen4ExpMultiTokenPredictor,
    _inherit_gguf_unquantized_modules,
    _remap_mtp_weight_name,
)


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
    monkeypatch.setattr(
        gguf_adapters.base, "detect_gguf_multimodal", lambda _: detected
    )
    model_config = SimpleNamespace(
        architecture=architecture,
        model="model.gguf",
        hf_config=SimpleNamespace(
            vision_config=object() if has_vision_config else None
        ),
    )

    assert gguf_loader.GGUFModelLoader._get_mmproj_file(model_config) == expected


@pytest.mark.parametrize(
    ("adapter_cls", "architecture", "weight_prefix"),
    [
        (Qwen4ExpGGUFAdapter, "Qwen4ExpForCausalLM", "model."),
        (Qwen4ExpMTPGGUFAdapter, "Qwen4ExpMTP", "mtp."),
    ],
)
def test_qwen4_exp_gguf_adapter_dequantizes_hyper_connection_weights(
    adapter_cls, architecture, weight_prefix
):
    qtype = gguf.GGMLQuantizationType.Q8_0
    quantized = gguf.quants.quantize(
        np.arange(32 * 32, dtype=np.float32).reshape(32, 32), qtype
    )
    name = f"{weight_prefix}hyper_connection_mixer.input_mix_weight_down"
    text_config = SimpleNamespace(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_value_head_dim=1,
        linear_key_head_dim=1,
        ple_layer_ids=[],
    )
    model_config = SimpleNamespace(
        architecture=architecture,
        dtype=torch.float16,
        hf_config=SimpleNamespace(get_text_config=lambda: text_config),
    )

    transformed = list(
        adapter_cls().transform_weights(
            iter(
                [
                    (f"{name}.qweight_type", torch.tensor(qtype)),
                    (f"{name}.qweight", torch.from_numpy(quantized)),
                ]
            ),
            model_config,
        )
    )

    assert [weight_name for weight_name, _ in transformed] == [f"{name}.weight"]
    assert transformed[0][1].dtype == torch.float16
    expected = gguf.quants.dequantize(quantized, qtype)
    torch.testing.assert_close(transformed[0][1], torch.from_numpy(expected).half())


def test_qwen4_exp_mtp_gguf_adapter_dequantizes_and_maps_hc_injection():
    qtype = gguf.GGMLQuantizationType.Q8_0
    quantized = gguf.quants.quantize(
        np.arange(32 * 32, dtype=np.float32).reshape(32, 32), qtype
    )
    name = "mtp.layers.0.attn_hyper_connection.block_inject_weight"
    text_config = SimpleNamespace(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_value_head_dim=1,
        linear_key_head_dim=1,
        ple_layer_ids=[],
    )
    model_config = SimpleNamespace(
        architecture="Qwen4ExpMTP",
        dtype=torch.float16,
        hf_config=SimpleNamespace(get_text_config=lambda: text_config),
    )

    transformed = list(
        Qwen4ExpMTPGGUFAdapter().transform_weights(
            iter(
                [
                    (f"{name}.qweight_type", torch.tensor(qtype)),
                    (f"{name}.qweight", torch.from_numpy(quantized)),
                ]
            ),
            model_config,
        )
    )

    assert [weight_name for weight_name, _ in transformed] == [f"{name}.weight"]
    assert transformed[0][1].dtype == torch.float16

    remapped_name = _remap_mtp_weight_name(transformed[0][0])
    assert remapped_name is not None
    child_name = remapped_name.removeprefix("model.")
    stacked = list(
        Qwen4ExpMultiTokenPredictor.hf_to_vllm_mapper.apply(
            [(child_name, transformed[0][1])]
        )
    )
    assert [weight_name for weight_name, _ in stacked] == [
        "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    ]
    assert stacked[0][1].shard_id == 1


def test_qwen4_exp_mtp_gguf_preserves_dense_module_precision():
    source = GGUFConfig(
        [
            "model.layers.3.self_attn.indexer.index_qk_proj",
            "mtp.layers.0.self_attn.indexer.index_qk_proj",
        ]
    )
    draft = GGUFConfig()

    _inherit_gguf_unquantized_modules(draft, source, mtp_start_layer_idx=48)

    expected = "mtp.layers.48.self_attn.indexer.index_qk_proj"
    assert draft.unquantized_modules == [expected]
    assert is_layer_skipped_gguf(
        "draft_model.mtp.layers.48.self_attn.indexer.index_qk_proj",
        draft.unquantized_modules,
    )
