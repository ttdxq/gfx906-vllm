# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.models.config import (
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from ...utils import spawn_new_process_for_each_test


def _text_config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        # `Qwen4ExpTextConfig` requires an EOS token whenever PLE is enabled.
        "eos_token_id": 1,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


@pytest.mark.parametrize("keep_quantized", [False, True])
def test_gguf_hc_up_preserves_checkpoint_precision(
    tmp_path, monkeypatch, keep_quantized
) -> None:
    import gguf

    from vllm.model_executor.model_loader.gguf_adapters import Qwen4ExpGGUFAdapter

    monkeypatch.setenv("VLLM_QWEN4EXP_GGUF_HC_UP_QUANT", str(int(keep_quantized)))
    _text_config(architectures=["Qwen4ExpForCausalLM"]).save_pretrained(tmp_path)
    model_config = ModelConfig(model=str(tmp_path), dtype="half", max_model_len=4096)
    qtype = gguf.GGMLQuantizationType.Q8_0
    packed = torch.from_numpy(
        gguf.quantize(
            torch.arange(1024, dtype=torch.float32).view(32, 32).numpy(), qtype
        )
    )
    base = "model.layers.0.attn_hyper_connection"
    weights = []
    for suffix in ("input_mix_weight_up", "input_mix_weight_down"):
        weights.extend(
            [
                (f"{base}.{suffix}.qweight_type", torch.tensor(int(qtype))),
                (f"{base}.{suffix}.qweight", packed),
            ]
        )
    restored = dict(
        Qwen4ExpGGUFAdapter().transform_weights(iter(weights), model_config)
    )
    expected = torch.from_numpy(gguf.dequantize(packed.numpy(), qtype)).half()
    torch.testing.assert_close(
        restored[f"{base}.input_mix_weight_down.weight"], expected
    )
    if keep_quantized:
        assert restored[f"{base}.input_mix_weight_up.qweight"] is packed
        assert restored[f"{base}.input_mix_weight_up.qweight_type"].item() == int(qtype)
        assert f"{base}.input_mix_weight_up.weight" not in restored
    else:
        torch.testing.assert_close(
            restored[f"{base}.input_mix_weight_up.weight"], expected
        )


def test_gguf_adapter_restores_gdn_head_order_and_norm_weights(tmp_path) -> None:
    """GGUF value heads and zero-centred gammas retain their HF meaning."""
    import gguf

    from vllm.model_executor.model_loader.gguf_adapters import Qwen4ExpGGUFAdapter

    config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        ple_layer_ids=[],
        output_gate_type="sigmoid",
    )
    config.save_pretrained(tmp_path)
    model_config = ModelConfig(model=str(tmp_path), dtype="half", max_model_len=4096)
    torch.manual_seed(0)
    qtype = gguf.GGMLQuantizationType.Q4_0
    qkv = gguf.quantize(torch.randn(256, 32).numpy(), qtype)
    out = gguf.quantize(torch.randn(32, 128).numpy(), qtype)
    prefix = "model.layers.0.linear_attn"
    decay = -torch.exp(torch.tensor([0.25, -1.0, 2.0, 0.5]))
    weights = [
        (f"{prefix}.in_proj_qkv.qweight_type", torch.tensor(int(qtype))),
        (f"{prefix}.out_proj.qweight_type", torch.tensor(int(qtype))),
        (f"{prefix}.in_proj_qkv.qweight", torch.from_numpy(qkv)),
        (f"{prefix}.out_proj.qweight", torch.from_numpy(out)),
        (f"{prefix}.A_log", decay),
        (f"{prefix}.norm.weight", torch.ones(32)),
        ("model.layers.0.attn_hyper_connection.hc_norm.weight", torch.ones(64)),
    ]
    restored = dict(
        Qwen4ExpGGUFAdapter().transform_weights(iter(weights), model_config)
    )
    permutation = torch.tensor([0, 2, 1, 3])
    qkv_ref = torch.from_numpy(gguf.dequantize(qkv, qtype))
    qkv_ref = torch.cat(
        [qkv_ref[:128], qkv_ref[128:].view(4, 32, 32)[permutation].flatten(0, 1)]
    )
    actual_qkv = torch.from_numpy(
        gguf.dequantize(restored[f"{prefix}.in_proj_qkv.qweight"].numpy(), qtype)
    )
    torch.testing.assert_close(actual_qkv, qkv_ref, rtol=0, atol=0)
    out_ref = torch.from_numpy(gguf.dequantize(out, qtype))
    out_ref = out_ref.view(32, 4, 32)[:, permutation].flatten(1).half()
    torch.testing.assert_close(
        restored[f"{prefix}.out_proj.qweight"], out_ref, rtol=0, atol=0
    )
    assert (
        int(restored[f"{prefix}.out_proj.qweight_type"])
        == gguf.GGMLQuantizationType.F16
    )
    torch.testing.assert_close(
        restored[f"{prefix}.A_log"], torch.log(-decay)[permutation], rtol=0, atol=0
    )
    assert torch.equal(restored[f"{prefix}.norm.weight"], torch.ones(32))
    assert (
        torch.count_nonzero(
            restored["model.layers.0.attn_hyper_connection.hc_norm.weight"]
        )
        == 0
    )


def test_gguf_mtp_adapter_maps_standalone_checkpoint() -> None:
    from vllm.model_executor.model_loader.gguf_adapters import (
        Qwen4ExpMTPGGUFAdapter,
    )

    text_config = _text_config(
        num_hidden_layers=48,
        layer_types=["indexed_attention"] * 48,
        ple_layer_ids=[],
    )
    name_map = Qwen4ExpMTPGGUFAdapter.build_name_map_for_text_config(text_config)

    assert name_map["blk.48.nextn.eh_proj.weight"] == "mtp.eh_proj.weight"
    assert (
        name_map["blk.48.nextn.enorm.weight"]
        == "mtp.pre_fc_norm_embedding.weight"
    )
    assert (
        name_map["blk.48.nextn.hnorm.weight"]
        == "mtp.pre_fc_norm_hidden.weight"
    )
    assert (
        name_map["blk.48.nextn.hc_head_down.weight"]
        == "mtp.hyper_connection_mixer.input_mix_weight_down.weight"
    )
    assert (
        name_map["blk.48.attn_q.weight"]
        == "mtp.layers.0.self_attn.q_proj.weight"
    )
    assert (
        name_map["blk.48.ffn_gate_exps.weight"]
        == "mtp.layers.0.mlp.experts.0.gate_proj.weight"
    )
    assert "blk.49.nextn.eh_proj.weight" not in name_map


def test_moe_sp_rejects_pipeline_parallel() -> None:
    """Automatic MoE SP must validate HC topology even without --enable-hc-sp."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(num_experts=4, ple_layer_ids=[]),
        ),
        cache_config=CacheConfig(),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
        ),
    )
    with (
        patch("vllm.platforms.current_platform.is_cuda_alike", return_value=True),
        pytest.raises(ValueError, match="MoE SP requires PP=1"),
    ):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["amd", "nvidia"])
@pytest.mark.parametrize("use_sp", [False, True])
def test_qwen4_exp_mtp_returns_sample_and_multi_streams(
    backend: str, use_sp: bool
) -> None:
    """MTP returns complete, contiguous outputs after gathering padded SP shards."""
    mtp_module = f"vllm.models.qwen4_exp.{backend}.mtp"
    model = object.__new__(import_module(mtp_module).Qwen4ExpMultiTokenPredictor)
    model.use_hc_sequence_parallel = use_sp
    model.use_sequence_parallel = False
    torch.nn.Module.__init__(model)
    model.hc_count = 2
    model.hidden_size = 4
    model.num_mtp_layers = 1
    model.pre_fc_norm_embedding = torch.nn.Identity()
    model.pre_fc_norm_hidden = torch.nn.Identity()

    if backend == "amd":

        class SelectHiddenProjection(torch.nn.Module):
            def forward(self, eh_input):
                return eh_input[..., model.hidden_size :], None

        model.eh_proj = SelectHiddenProjection()
    else:
        model.fc_embedding = torch.nn.Identity()
        model.fc_hidden = torch.nn.Identity()

    def decoder(**kwargs):
        """Check that the embedding residual shares the hidden state's token slice."""
        hidden = kwargs["hidden_states"]
        if backend == "nvidia":
            prev_block_output = kwargs["prev_block_output"]
            assert prev_block_output.shape == (hidden.shape[0], model.hidden_size)
        return hidden, hidden, torch.zeros(hidden.shape[0], model.hc_count)

    model.layers = [decoder]
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda hidden_states, block_output, injection: (
            hidden_states,
            hidden_states.unflatten(-1, (2, 4)).mean(-2),
            None,
        ),
    )
    multi_hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    expected_sample = multi_hidden.unflatten(-1, (2, 4)).mean(-2)
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

    def gather(local):
        """Supply rank 1's last token and padding for a simulated two-rank gather."""
        assert local.shape[0] == 2
        remote = torch.cat([expected_sample[2:], multi_hidden[2:]], dim=-1)
        return torch.cat([local, remote, torch.full_like(remote, float("nan"))])

    with (
        patch(f"{mtp_module}.get_pp_group", return_value=pp_group),
        patch(f"{mtp_module}.sp_shard", side_effect=lambda x: x[:2]),
        patch(f"{mtp_module}.sp_all_gather", side_effect=gather),
    ):
        sample_hidden, returned_multi_hidden = model.forward(
            input_ids=None,
            positions=torch.arange(3),
            hidden_states=multi_hidden,
            inputs_embeds=torch.zeros(3, 4),
        )

    torch.testing.assert_close(sample_hidden, expected_sample)
    torch.testing.assert_close(returned_multi_hidden, multi_hidden)
    assert sample_hidden.is_contiguous()
    assert returned_multi_hidden.is_contiguous()


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["amd", "nvidia"])
def test_qwen4_exp_mtp_remaps_mixed_precision_layer_indices(backend: str) -> None:
    mtp_module = import_module(f"vllm.models.qwen4_exp.{backend}.mtp")

    quantized_layers = {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.0.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }

    assert mtp_module._remap_quantized_layers(quantized_layers, 48) == {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.48.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }


@pytest.mark.parametrize("wrapped_config", [False, True])
def test_qwen4_exp_mtp_override_sets_draft_config(
    wrapped_config: bool,
) -> None:
    text_config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        index_share_for_mtp_iteration=True,
    )
    config = (
        Qwen4ExpConfig(
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=text_config,
        )
        if wrapped_config
        else text_config
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.index_share_for_mtp_iteration is True
    assert draft_config.model_type == "qwen4_exp_mtp"
    assert draft_config.architectures == ["Qwen4ExpMTP"]
    assert draft_config.hc_mult == 2
    assert draft_config.n_predict == 1


def test_qwen4_exp_model_state_prepares_ngram_context() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)
    model_state.disk_offload_modules = []

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc = torch.full((9,), 3, dtype=torch.int32)
    expected_query_start_loc[0] = 0
    expected_query_start_loc[1] = 2
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context = torch.full((8, 3), 99, dtype=torch.int32)
    expected_context[:2] = torch.tensor([[99, 99, 20], [1, 2, 3]])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)

    # Retain the views to detect reallocations as the request layout changes.
    query_start_loc = model_inputs["query_start_loc"]
    ngram_context = model_inputs["ngram_context"]
    input_batch.num_reqs = 1
    input_batch.num_reqs_after_padding = 1
    input_batch.idx_mapping = torch.tensor([0])
    input_batch.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc.fill_(3)
    expected_query_start_loc[0] = 0
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context.fill_(99)
    expected_context[0] = torch.tensor([1, 2, 3])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)
    assert model_inputs["query_start_loc"].data_ptr() == query_start_loc.data_ptr()
    assert model_inputs["ngram_context"].data_ptr() == ngram_context.data_ptr()


def test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    with patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}):
        first = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)
        # Dummy runs establish the addresses used during CUDA graph capture.
        query_start_loc_ptr = first["query_start_loc"].data_ptr()
        ngram_context_ptr = first["ngram_context"].data_ptr()
        second = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)

    expected_query_start_loc = torch.full((9,), 4, dtype=torch.int32)
    expected_query_start_loc[:4] = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    torch.testing.assert_close(second["query_start_loc"], expected_query_start_loc)
    torch.testing.assert_close(
        second["ngram_context"], torch.full((8, 3), 99, dtype=torch.int32)
    )
    assert second["query_start_loc"].data_ptr() == query_start_loc_ptr
    assert second["ngram_context"].data_ptr() == ngram_context_ptr


@pytest.mark.parametrize(
    "model_cls", [Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration]
)
def test_qkv_proj_mapping_allows_bf16_indexer(model_cls) -> None:
    """Regression test for https://github.com/vllm-project/vllm/issues/62308.

    Layout of e.g. ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ: q/k/v are
    quantized and only the indexer projection is in `ignore`.
    """
    assert not should_ignore_layer(
        "model.layers.3.self_attn.qkv_proj",
        ignore=["model.layers.3.self_attn.indexer.index_qk_proj"],
        fused_mapping=model_cls.packed_modules_mapping,
    )
