# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig


def _mtp_config(model_type: str) -> PretrainedConfig:
    return PretrainedConfig(
        model_type=model_type,
        architectures=["SomeArch"],
        mtp_num_hidden_layers=1,
    )


@pytest.mark.parametrize(
    ("model_type", "expected_arch"),
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
        ("qwen3_5_text", "Qwen3_5MTP"),
        ("qwen3_5_moe_text", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_recognizes_text_only_types(model_type, expected_arch):
    config = SpeculativeConfig.hf_config_override(_mtp_config(model_type))

    assert config.model_type == "qwen3_5_mtp"
    assert config.architectures == [expected_arch]
    assert config.n_predict == 1
