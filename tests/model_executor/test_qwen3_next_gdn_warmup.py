# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

import vllm.model_executor.models.qwen3_next as qwen3_next_module
from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet


def test_gdn_profile_run_warms_prefill_kernels_once(monkeypatch):
    layer = object.__new__(Qwen3NextGatedDeltaNet)
    nn.Module.__init__(layer)
    calls: list[torch.Tensor] = []

    monkeypatch.setattr(
        qwen3_next_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    monkeypatch.setattr(
        layer,
        "_warmup_prefill_kernels",
        lambda mixed_qkv: calls.append(mixed_qkv),
    )

    mixed_qkv = torch.empty(2, 8)
    layer._forward_core(
        mixed_qkv=mixed_qkv,
        b=torch.empty(2, 1),
        a=torch.empty(2, 1),
        core_attn_out=torch.empty(2, 1, 1),
    )

    assert calls == [mixed_qkv]


def test_gdn_prefill_kernel_warmup_covers_autotune_sizes_once(monkeypatch):
    layer = object.__new__(Qwen3NextGatedDeltaNet)
    nn.Module.__init__(layer)
    layer.num_k_heads = 1
    layer.num_v_heads = 2
    layer.tp_size = 1
    layer.head_k_dim = 4
    layer.head_v_dim = 4
    layer.prefix = "model.layers.0.linear_attn"
    layer.get_state_dtype = lambda: (torch.float32, torch.float32)
    warmed_sizes: list[int] = []

    def fake_chunk_gated_delta_rule(**kwargs):
        warmed_sizes.append(kwargs["q"].shape[1])
        return torch.empty(0), torch.empty(0)

    monkeypatch.setattr(
        qwen3_next_module,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    mixed_qkv = torch.empty(1, 16)
    layer._warmup_prefill_kernels(mixed_qkv)
    layer._warmup_prefill_kernels(mixed_qkv)

    assert warmed_sizes == [16, 32, 64]
