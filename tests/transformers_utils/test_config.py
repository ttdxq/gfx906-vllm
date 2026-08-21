# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`vllm.LLMEngine._get_eos_token_id`.
"""

import pytest
from transformers import SiglipVisionConfig

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils import config as config_utils
from vllm.transformers_utils import gguf_utils
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config


@pytest.mark.parametrize(
    ("model_type", "expected_class_name"),
    [
        ("qwen3_5_text", "Qwen3_5TextConfig"),
        ("qwen3_5_moe_text", "Qwen3_5MoeTextConfig"),
    ],
)
def test_qwen35_text_only_config_registry(model_type, expected_class_name):
    assert config_utils._CONFIG_REGISTRY[model_type].__name__ == expected_class_name


def test_detect_gguf_multimodal_ignores_unrelated_shared_projector(tmp_path):
    model = tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
    model.touch()
    (tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf").touch()
    (tmp_path / "Qwen3.8-27B-mmproj-F16.gguf").touch()

    assert gguf_utils.detect_gguf_multimodal(str(model)) is None


def test_detect_gguf_multimodal_matches_model_family(tmp_path):
    model = tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    model.touch()
    mmproj = tmp_path / "Qwen3.8-27B-mmproj-F16.gguf"
    mmproj.touch()
    (tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf").touch()

    assert gguf_utils.detect_gguf_multimodal(str(model)) == mmproj


def test_detect_gguf_multimodal_accepts_single_family_directory(tmp_path):
    model = tmp_path / "gemma-3-4b-it-Q4_K_M.gguf"
    model.touch()
    (tmp_path / "gemma-3-4b-it-Q6_K.gguf").touch()
    mmproj = tmp_path / "mmproj-model-f16-4B.gguf"
    mmproj.touch()

    assert gguf_utils.detect_gguf_multimodal(str(model)) == mmproj


def test_patch_qwen35_multimodal_gguf_config(monkeypatch, tmp_path):
    model = tmp_path / "qwen35.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"GGUF")
    mmproj.write_bytes(b"GGUF")
    vision_config = SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=768,
        patch_size=16,
    )
    vision_config.projection_dim = 5120
    vision_config.spatial_merge_size = 2

    monkeypatch.setattr(gguf_utils, "detect_gguf_multimodal", lambda _: mmproj)
    monkeypatch.setattr(
        gguf_utils,
        "extract_vision_config_from_gguf",
        lambda _: vision_config,
    )

    config = Qwen3_5Config(architectures=["Qwen3_5ForCausalLM"])
    patched = gguf_utils.maybe_patch_hf_config_from_gguf(str(model), config)

    assert patched.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert patched.vision_config.depth == 27
    assert patched.vision_config.num_heads == 16
    assert patched.vision_config.out_hidden_size == 5120
    assert patched.vision_config.num_position_embeddings == 2304
    assert patched.vision_config.deepstack_visual_indexes == []


def test_multimodal_gguf_uses_original_processor_repo(monkeypatch, tmp_path):
    model = tmp_path / "qwen35.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"GGUF")
    mmproj.write_bytes(b"GGUF")

    monkeypatch.setattr(gguf_utils, "detect_gguf_multimodal", lambda _: mmproj)
    monkeypatch.setattr(gguf_utils.gguf, "GGUFReader", lambda _: object())
    monkeypatch.setattr(
        gguf_utils,
        "_read_gguf_scalar",
        lambda _reader, key, default=None: (
            "https://huggingface.co/Qwen/Qwen3.8-27B"
            if key == "general.base_model.0.repo_url"
            else default
        ),
    )

    assert gguf_utils.gguf_multimodal_processor_repo(str(model)) == "Qwen/Qwen3.8-27B"


def test_maybe_override_with_speculators_skips_unsupported_local_gguf_arch(
    monkeypatch,
    tmp_path,
):
    model = tmp_path / "qwen35.gguf"
    model.write_bytes(b"GGUF")

    def fail_get_config_dict(*args, **kwargs):
        raise ValueError("GGUF model with architecture qwen35 is not supported yet.")

    monkeypatch.setattr(
        config_utils.PretrainedConfig,
        "get_config_dict",
        fail_get_config_dict,
    )

    assert config_utils.maybe_override_with_speculators(
        model=str(model),
        tokenizer="/tmp/tokenizer",
        trust_remote_code=False,
    ) == (str(model), "/tmp/tokenizer", None)


def test_maybe_override_with_speculators_reraises_hf_config_path_gguf_errors(
    monkeypatch,
    tmp_path,
):
    model = tmp_path / "qwen35.gguf"
    model.write_bytes(b"GGUF")

    def fail_get_config_dict(*args, **kwargs):
        raise ValueError("GGUF model with architecture qwen35 is not supported yet.")

    monkeypatch.setattr(
        config_utils.PretrainedConfig,
        "get_config_dict",
        fail_get_config_dict,
    )

    with pytest.raises(ValueError, match="architecture qwen35"):
        config_utils.maybe_override_with_speculators(
            model=str(model),
            tokenizer="/tmp/tokenizer",
            trust_remote_code=False,
            hf_config_path="/tmp/hf-config",
        )


def test_get_config_builds_qwen35_config_from_local_gguf_metadata(
    monkeypatch,
    tmp_path,
):
    model = tmp_path / "qwen35.gguf"
    model.write_bytes(b"GGUF")

    def fail_get_config_dict(*args, **kwargs):
        raise ValueError("GGUF model with architecture qwen35 is not supported yet.")

    config_dict = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 65,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "head_dim": 256,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 24,
            "full_attention_interval": 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "mrope_section": [11, 11, 10, 0],
                "mrope_interleaved": True,
            },
            "bos_token_id": 248044,
            "eos_token_id": 248046,
            "pad_token_id": 248055,
            "tie_word_embeddings": False,
        },
    }

    monkeypatch.setattr(
        config_utils.PretrainedConfig,
        "get_config_dict",
        fail_get_config_dict,
    )
    monkeypatch.setattr(
        config_utils,
        "qwen35_gguf_config_dict",
        lambda model: config_dict,
    )

    config = config_utils.get_config(str(model), trust_remote_code=False)

    assert config.model_type == "qwen3_5"
    assert config.architectures == ["Qwen3_5ForCausalLM"]
    assert config.text_config.linear_num_value_heads == 24


def test_try_get_safetensors_metadata_skips_local_files(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")

    def fail_get_safetensors_metadata(*args, **kwargs):
        raise AssertionError("local model files should not query HF metadata")

    monkeypatch.setattr(
        config_utils,
        "get_safetensors_metadata",
        fail_get_safetensors_metadata,
    )

    assert config_utils.try_get_safetensors_metadata(str(model)) is None


def test_qwen35_gguf_config_derives_value_heads_from_attn_qkv(
    monkeypatch,
    tmp_path,
):
    model = tmp_path / "qwen35-4b.gguf"
    model.write_bytes(b"GGUF")

    class FakeField:
        def __init__(self, value):
            self.value = value

        def contents(self):
            return self.value

    class FakeTensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class FakeReader:
        tensors = [
            FakeTensor("blk.0.attn_qkv.weight", (2560, 8192)),
            FakeTensor("output.weight", (151936, 2560)),
        ]

        def get_field(self, key):
            fields = {
                "general.architecture": "qwen35",
                "tokenizer.ggml.tokens": ["<pad>", "x"],
                "qwen35.ssm.state_size": 128,
                "qwen35.ssm.group_count": 16,
                "qwen35.block_count": 1,
                "qwen35.embedding_length": 2560,
                "qwen35.feed_forward_length": 9728,
                "qwen35.attention.head_count": 16,
                "qwen35.attention.head_count_kv": 8,
                "qwen35.context_length": 32768,
                "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
                "qwen35.attention.key_length": 256,
                "qwen35.ssm.conv_kernel": 4,
                "qwen35.full_attention_interval": 4,
                "qwen35.rope.freq_base": 1000000.0,
                "qwen35.rope.dimension_sections": [8, 8, 8, 0],
                "tokenizer.ggml.bos_token_id": 0,
                "tokenizer.ggml.eos_token_id": 1,
                "tokenizer.ggml.padding_token_id": 0,
            }
            if key not in fields:
                return None
            return FakeField(fields[key])

    monkeypatch.setattr(gguf_utils.gguf, "GGUFReader", lambda _: FakeReader())

    config_dict = gguf_utils.qwen35_gguf_config_dict(str(model))

    assert config_dict is not None
    assert config_dict["text_config"]["linear_num_value_heads"] == 32


def test_qwen35_gguf_config_excludes_dense_mtp_layer(
    monkeypatch,
    tmp_path,
):
    model = tmp_path / "qwen35-mtp.gguf"
    model.write_bytes(b"GGUF")

    class FakeField:
        def __init__(self, value):
            self.value = value

        def contents(self):
            return self.value

    class FakeTensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class FakeReader:
        tensors = [
            FakeTensor("blk.0.attn_qkv.weight", (2560, 8192)),
            FakeTensor("blk.1.attn_k.weight", (2560, 1024)),
            FakeTensor("blk.1.nextn.eh_proj.weight", (2560, 5120)),
            FakeTensor("output.weight", (248320, 2560)),
        ]

        def get_field(self, key):
            fields = {
                "general.architecture": "qwen35",
                "tokenizer.ggml.tokens": ["<pad>", "x"],
                "qwen35.ssm.state_size": 128,
                "qwen35.ssm.group_count": 16,
                "qwen35.block_count": 2,
                "qwen35.nextn_predict_layers": 1,
                "qwen35.embedding_length": 2560,
                "qwen35.feed_forward_length": 9728,
                "qwen35.attention.head_count": 16,
                "qwen35.attention.head_count_kv": 8,
                "qwen35.context_length": 32768,
                "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
                "qwen35.attention.key_length": 256,
                "qwen35.ssm.conv_kernel": 4,
                "qwen35.full_attention_interval": 4,
                "qwen35.rope.freq_base": 1000000.0,
                "qwen35.rope.dimension_sections": [8, 8, 8, 0],
                "tokenizer.ggml.bos_token_id": 0,
                "tokenizer.ggml.eos_token_id": 1,
                "tokenizer.ggml.padding_token_id": 0,
            }
            value = fields.get(key)
            return None if value is None else FakeField(value)

    monkeypatch.setattr(gguf_utils.gguf, "GGUFReader", lambda _: FakeReader())

    config_dict = gguf_utils.qwen35_gguf_config_dict(str(model))

    assert config_dict is not None
    text_config = config_dict["text_config"]
    assert text_config["num_hidden_layers"] == 1
    assert len(text_config["layer_types"]) == 1
    assert text_config["num_nextn_predict_layers"] == 1


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118
