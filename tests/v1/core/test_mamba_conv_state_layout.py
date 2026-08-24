# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba import mamba_utils


@pytest.fixture(autouse=True)
def clear_conv_state_layout_cache():
    mamba_utils.get_conv_state_layout.cache_clear()
    yield
    mamba_utils.get_conv_state_layout.cache_clear()


def _conv_shapes() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    calculator = mamba_utils.MambaStateShapeCalculator
    return [
        (calculator.mamba1_state_shape(2, 64, 8, 4)[0], (3, 32)),
        (calculator.mamba2_state_shape(2, 64, 2, 8, 16, 8, 4)[0], (3, 48)),
        (calculator.short_conv_state_shape(2, 64, 4)[0], (3, 32)),
        (
            calculator.gated_delta_net_state_shape(2, 4, 4, 16, 16, 4, 2)[0],
            (5, 96),
        ),
        (calculator.kda_state_shape(2, 4, 16, 2, 8, 4)[0], (3, 32)),
        (calculator.kda_state_shape(2, 4, 16, 2, 8, 4)[1], (3, 8)),
    ]


@pytest.mark.parametrize("layout", ["SD", "DS"])
def test_conv_state_shapes_follow_layout(monkeypatch, layout):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)

    for actual, sd_shape in _conv_shapes():
        expected = sd_shape if layout == "SD" else sd_shape[::-1]
        assert actual == expected


def test_conv_state_layout_defaults_to_sd(monkeypatch):
    monkeypatch.delenv("VLLM_SSM_CONV_STATE_LAYOUT", raising=False)

    assert mamba_utils.get_conv_state_layout() == "SD"
    assert not mamba_utils.is_conv_state_dim_first()


def test_conv_state_layout_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "INVALID")

    with pytest.raises(ValueError, match="VLLM_SSM_CONV_STATE_LAYOUT"):
        mamba_utils.get_conv_state_layout()


def test_sd_conv_copy_spec_offsets_state_length(monkeypatch):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "SD")
    state = torch.empty(2, 4, 3)

    spec = mamba_utils.get_conv_copy_spec(state, [1], 0, 2)

    assert spec.start_addr == state[1, 1:].data_ptr()
    assert spec.num_elements == 9


def test_ds_conv_copy_spec(monkeypatch):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    state = torch.empty(2, 3, 4)

    spec = mamba_utils.get_conv_copy_spec(state, [1], 0, 1)
    assert spec.start_addr == state[1].data_ptr()
    assert spec.num_elements == 12

    with pytest.raises(NotImplementedError, match="num_accepted_tokens > 1"):
        mamba_utils.get_conv_copy_spec(state, [1], 0, 2)
