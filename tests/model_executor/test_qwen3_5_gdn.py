from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as gating_module
import vllm.model_executor.layers.mamba.gdn_linear_attn as gdn_linear_attn
import vllm.model_executor.models.qwen3_5 as qwen3_5_module
from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    GatedDeltaNetAttention,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5GatedDeltaNet,
    _make_qwen35_fused_expert_params_mapping,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock,
    _gdn_convert_state_layout,
    _gdn_recurrent_state_to_cache,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)


def test_gfx906_mtp_gdn_uses_custom_update(monkeypatch):
    captured: dict[str, object] = {}

    def fake_mtp_update(**kwargs):
        captured.update(kwargs)
        q = kwargs["q"]
        v = kwargs["v"]
        return torch.zeros(
            (q.shape[1], v.shape[2], v.shape[3]), dtype=q.dtype, device=q.device
        )

    monkeypatch.setattr(gating_module, "_is_gfx906_rocm", lambda: True)
    monkeypatch.setattr(
        gating_module.ops,
        "fused_sigmoid_gating_delta_rule_gfx906_mtp_update",
        fake_mtp_update,
    )

    q = torch.randn(1, 3, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn(1, 3, 4, 5)
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    state = torch.randn(4, 4, 5, 4)
    state_indices = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([2], dtype=torch.int32)

    output, final_state = gating_module.fused_sigmoid_gating_delta_rule_update(
        A_log=torch.randn(4),
        a=a,
        b=b,
        dt_bias=torch.randn(4),
        q=q,
        k=k,
        v=v,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    assert output.shape == (1, 3, 4, 5)
    assert final_state is state
    assert captured["state"] is state
    assert captured["state_indices"] is state_indices
    assert captured["num_accepted_tokens"] is num_accepted_tokens


def test_gdn_mixed_batch_builds_prefill_only_metadata():
    builder = object.__new__(GDNAttentionMetadataBuilder)
    builder.use_spec_decode = False
    builder.use_full_cuda_graph = False

    query_start_loc = torch.tensor([0, 1, 2, 5], dtype=torch.int32)
    common = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        num_computed_tokens_cpu=torch.tensor([8, 9, 0], dtype=torch.int32),
        block_table_tensor=torch.tensor([[3], [4], [5]], dtype=torch.int32),
        max_query_len=3,
        num_reqs=3,
        num_actual_tokens=5,
    )

    metadata = builder.build(0, common)

    assert metadata.num_decodes == 2
    assert metadata.num_prefills == 1
    assert torch.equal(
        metadata.prefill_query_start_loc,
        torch.tensor([0, 3], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.prefill_state_indices,
        torch.tensor([5], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.prefill_has_initial_state,
        torch.tensor([False]),
    )


def test_qwen3_5_gdn_prefill_state_uses_cache_layout():
    state = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    cached = _gdn_recurrent_state_to_cache(state)

    assert cached.shape == (2, 3, 5, 4)
    assert torch.equal(cached, state.transpose(-1, -2))


def test_qwen3_5_chunk_state_keeps_packed_decode_layout():
    state = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

    cached = _gdn_convert_state_layout(
        state,
        source_uses_kv_layout=True,
        target_uses_kv_layout=True,
    )

    assert torch.equal(cached, state)
    assert cached.is_contiguous()


def test_qwen3_5_chunk_state_transposes_for_standard_cache_layout():
    state = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

    cached = _gdn_convert_state_layout(
        state,
        source_uses_kv_layout=True,
        target_uses_kv_layout=False,
    )

    assert torch.equal(cached, state.transpose(-1, -2))
    assert cached.is_contiguous()


def test_qwen3_5_text_model_declares_hybrid_cache_interface():
    assert Qwen3_5ForCausalLMBase.is_hybrid
    assert Qwen3_5ForCausalLMBase.supports_mrope
    assert callable(Qwen3_5ForCausalLMBase.get_mamba_state_dtype_from_config)
    assert callable(Qwen3_5ForCausalLMBase.get_mamba_state_shape_from_config)
    assert callable(Qwen3_5ForCausalLMBase.get_mamba_state_copy_func)


def test_qwen3_5_text_mrope_positions():
    model = object.__new__(Qwen3_5ForCausalLMBase)

    positions, delta = model.get_mrope_input_positions([11, 12, 13], [])

    assert positions.dtype == torch.long
    assert positions.shape == (3, 3)
    assert torch.equal(positions, torch.tensor([[0, 1, 2]]).expand(3, -1))
    assert delta == 0


def test_qwen3_next_moe_sequence_parallel_restores_full_tokens(monkeypatch):
    module = object.__new__(Qwen3NextSparseMoeBlock)
    torch.nn.Module.__init__(module)
    module.is_sequence_parallel = True
    module.shared_expert = None
    module.tp_size = 2

    class _FakeExperts:
        is_internal_router = True

        def __call__(self, hidden_states, router_logits):
            assert router_logits is hidden_states
            return hidden_states + 1

    module.experts = _FakeExperts()

    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.sequence_parallel_chunk",
        lambda hidden_states: hidden_states[:2],
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.tensor_model_parallel_all_gather",
        lambda hidden_states, dim: torch.cat(
            [hidden_states, hidden_states + 10], dim=dim
        ),
    )

    hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    output = module(hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.equal(output[:2], hidden_states[:2] + 1)
    assert torch.equal(output[2:], hidden_states[:2] + 11)


class _FakeQuantConfig:
    def get_name(self) -> str:
        return "gguf"


def test_qwen3_5_load_uses_native_checkpoint_mapper(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeLoader:
        def __init__(self, model, skip_prefixes):
            del model, skip_prefixes

        def load_weights(self, weights, mapper=None):
            del weights
            captured["mapper"] = mapper
            return set()

    monkeypatch.setattr(qwen3_5_module, "AutoWeightsLoader", _FakeLoader)

    model = object.__new__(Qwen3_5ForCausalLMBase)
    torch.nn.Module.__init__(model)
    model.quant_config = None

    model.load_weights([])

    assert captured["mapper"] is model.hf_to_vllm_mapper


def test_qwen3_5_fused_expert_mapping_supports_base_layer_layout():
    class _FakeModel:
        def named_parameters(self):
            return [("model.layers.0.mlp.experts.base_layer.w13_weight", object())]

    mapping = _make_qwen35_fused_expert_params_mapping(_FakeModel())

    assert any(target == "experts.base_layer.w13_weight" for target, *_ in mapping)
    assert any(target == "experts.base_layer.w2_weight" for target, *_ in mapping)


class _FakeLinear:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output

    def __call__(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        del hidden_states
        return self.output, None


class _IdentityNorm:
    weight = None
    eps = 1e-6
    norm_before_gate = True

    def __call__(self, core_attn_out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del z
        return core_attn_out


class _FakeOutProj:
    def __call__(self, core_attn_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        return core_attn_out, None


def test_qwen3_5_split_gdn_calls_core_with_b_then_a(monkeypatch):
    module = object.__new__(Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(module)

    module.split_projections = True
    module.key_dim = 4
    module.value_dim = 4
    module.tp_size = 1
    module.head_v_dim = 2
    module.num_v_heads = 2
    module.prefix = "model.layers.0.linear_attn"
    module._qwen35_linear_attn_profile_enabled = False
    module.use_grouped_gguf_mmvq = False
    module.use_grouped_gguf_ba_mmvq = False
    module.use_empty_core_attn_out_for_single_token = False
    module.enable_packed_recurrent_decode = False
    module._qwen35_is_gfx906_rocm = False
    module.call_b_first = True
    module.norm = _IdentityNorm()
    module.out_proj = _FakeOutProj()
    module._maybe_log_debug_stats = lambda **_: None

    hidden_states = torch.randn(3, 8)
    q = torch.randn(3, 4)
    k = torch.randn(3, 4)
    v = torch.randn(3, 4)
    mixed_qkv = torch.cat((q, k, v), dim=-1)
    z = torch.randn(3, 4)
    b = torch.full((3, 2), 11.0)
    a = torch.full((3, 2), 22.0)

    module.in_proj_qkv = _FakeLinear(mixed_qkv)
    module.in_proj_z = _FakeLinear(z)
    module.in_proj_b = _FakeLinear(b)
    module.in_proj_a = _FakeLinear(a)

    captured: dict[str, torch.Tensor] = {}

    def fake_gdn_attention_core(
        passed_mixed_qkv: torch.Tensor,
        passed_b: torch.Tensor,
        passed_a: torch.Tensor,
        core_attn_out: torch.Tensor,
        layer_name: str,
    ) -> None:
        captured["mixed_qkv"] = passed_mixed_qkv
        captured["b"] = passed_b
        captured["a"] = passed_a
        captured["layer_name"] = torch.tensor([len(layer_name)])
        core_attn_out.zero_()

    monkeypatch.setattr(torch.ops.vllm, "gdn_attention_core", fake_gdn_attention_core)

    output = torch.empty(3, 4)
    module.forward(hidden_states, output)

    assert torch.equal(captured["mixed_qkv"], mixed_qkv)
    assert torch.equal(captured["b"], b)
    assert torch.equal(captured["a"], a)


def test_shared_gdn_calls_registered_core_op(monkeypatch):
    module = object.__new__(GatedDeltaNetAttention)
    torch.nn.Module.__init__(module)

    module.gqa_interleaved_layout = False
    module.key_dim = 4
    module.value_dim = 4
    module.tp_size = 1
    module.head_v_dim = 2
    module.num_v_heads = 2
    module.prefix = "model.layers.0.linear_attn"
    module.in_proj_qkv = _FakeLinear(torch.randn(3, 12))
    module.in_proj_z = _FakeLinear(torch.randn(3, 4))
    module.in_proj_ba = _FakeLinear(torch.randn(3, 4))
    module.norm = _IdentityNorm()
    module.out_proj = _FakeOutProj()
    module._can_use_empty_core_attn_out = lambda _: False

    captured: dict[str, object] = {}

    def fake_gdn_attention_core(
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        layer_name: str,
    ) -> None:
        captured["mixed_qkv"] = mixed_qkv
        captured["b"] = b
        captured["a"] = a
        captured["layer_name"] = layer_name
        core_attn_out.zero_()

    monkeypatch.setattr(torch.ops.vllm, "gdn_attention_core", fake_gdn_attention_core)

    module.forward(torch.randn(3, 8), torch.empty(3, 4))

    assert isinstance(captured["mixed_qkv"], torch.Tensor)
    assert isinstance(captured["b"], torch.Tensor)
    assert isinstance(captured["a"], torch.Tensor)
    assert captured["layer_name"] == module.prefix


@pytest.mark.parametrize(
    ("interleaved", "expected_qkvz", "expected_ba"),
    [
        (False, [4, 4, 8, 8], [6, 6]),
        (True, [24], [12]),
    ],
)
def test_shared_gdn_projection_shards(
    monkeypatch, interleaved, expected_qkvz, expected_ba
):
    captured: list[list[int]] = []

    def fake_merged_column_parallel_linear(*, output_sizes, **kwargs):
        captured.append(output_sizes)
        return object()

    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.gdn_linear_attn.MergedColumnParallelLinear",
        fake_merged_column_parallel_linear,
    )
    module = SimpleNamespace(gqa_interleaved_layout=interleaved)

    GatedDeltaNetAttention.create_qkvz_proj(module, 16, 4, 8, None, "qkvz")
    GatedDeltaNetAttention.create_ba_proj(module, 16, 6, None, "ba")

    assert captured == [expected_qkvz, expected_ba]


def test_qwen3_5_gguf_split_gdn_expands_qk_in_tiled_v_head_order():
    module = object.__new__(Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(module)
    module.split_projections = True
    module.quant_config = _FakeQuantConfig()
    module.num_k_heads = 2
    module.num_v_heads = 4
    module.use_tiled_qk_expand = True

    query = torch.tensor([[[[10.0], [20.0]]]])
    key = torch.tensor([[[[30.0], [40.0]]]])

    expanded_query, expanded_key = module._expand_qk_heads_for_gdn(query, key)

    assert expanded_query.flatten().tolist() == [10.0, 20.0, 10.0, 20.0]
    assert expanded_key.flatten().tolist() == [30.0, 40.0, 30.0, 40.0]


def test_qwen3_5_non_gguf_gdn_keeps_grouped_qk_expansion():
    module = object.__new__(Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(module)
    module.split_projections = False
    module.quant_config = None
    module.num_k_heads = 2
    module.num_v_heads = 4
    module.use_tiled_qk_expand = False

    query = torch.tensor([[[[10.0], [20.0]]]])
    key = torch.tensor([[[[30.0], [40.0]]]])

    expanded_query, expanded_key = module._expand_qk_heads_for_gdn(query, key)

    assert expanded_query.flatten().tolist() == [10.0, 10.0, 20.0, 20.0]
    assert expanded_key.flatten().tolist() == [30.0, 30.0, 40.0, 40.0]


def _run_speculative_gate_selection(
    monkeypatch,
    *,
    num_prefills: int,
    spec_token_indx: torch.Tensor,
    non_spec_token_indx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = 4
    num_spec_tokens = spec_token_indx.numel()
    metadata = GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=non_spec_token_indx.numel(),
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=num_spec_tokens,
        num_actual_tokens=num_tokens,
        has_initial_state=torch.zeros(1, dtype=torch.bool),
        prefill_query_start_loc=torch.tensor([0, non_spec_token_indx.numel()]),
        prefill_state_indices=torch.tensor([0]),
        prefill_has_initial_state=torch.zeros(1, dtype=torch.bool),
        spec_query_start_loc=torch.tensor([0, num_spec_tokens]),
        non_spec_query_start_loc=torch.tensor([0, non_spec_token_indx.numel()]),
        spec_state_indices_tensor=torch.zeros(
            (1, num_spec_tokens), dtype=torch.long
        ),
        non_spec_state_indices_tensor=torch.zeros(
            non_spec_token_indx.numel(), dtype=torch.long
        ),
        spec_sequence_masks=torch.tensor([True]),
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=torch.ones(1, dtype=torch.long),
    )
    monkeypatch.setattr(
        gdn_linear_attn,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"layer": metadata}, virtual_engine=0
        ),
    )
    monkeypatch.setattr(
        gdn_linear_attn,
        "causal_conv1d_update",
        lambda value, *args, **kwargs: value,
    )
    monkeypatch.setattr(
        gdn_linear_attn,
        "causal_conv1d_fn",
        lambda value, *args, **kwargs: value,
    )
    monkeypatch.setattr(gdn_linear_attn, "_is_gfx906_rocm", lambda: False)

    captured_gates = []

    def fake_recurrent_update(
        *, a: torch.Tensor, b: torch.Tensor, q: torch.Tensor, **kwargs
    ):
        captured_gates.append((a.clone(), b.clone()))
        return q, torch.zeros((1, 1, 1))

    monkeypatch.setattr(
        gdn_linear_attn,
        "fused_sigmoid_gating_delta_rule_update",
        fake_recurrent_update,
    )

    def fake_post_conv_prep(*, conv_output: torch.Tensor, **kwargs):
        output = conv_output.unsqueeze(1)
        gate = torch.zeros((conv_output.size(0), 1))
        return output, output, output, gate, gate

    monkeypatch.setattr(
        gdn_linear_attn, "fused_post_conv_prep", fake_post_conv_prep
    )

    def rearrange(value):
        if value is None:
            return None, None, None
        output = value.unsqueeze(0).unsqueeze(2)
        return output, output, output

    layer = SimpleNamespace(
        _log_projection_debug_once=lambda: None,
        prefix="layer",
        enable_packed_recurrent_decode=False,
        kv_cache=(torch.zeros((1, 1, 1)), torch.zeros((1, 1, 1))),
        conv1d=SimpleNamespace(weight=torch.ones((1, 1, 1)), bias=None),
        activation="silu",
        rearrange_mixed_qkv=rearrange,
        A_log=torch.ones(1),
        dt_bias=torch.ones(1),
        num_k_heads=1,
        tp_size=1,
        head_k_dim=1,
        head_v_dim=1,
        chunk_gated_delta_rule=lambda **kwargs: (
            kwargs["q"].transpose(1, 2),
            torch.zeros((1, 1, 1)),
        ),
    )
    mixed_qkv = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    a = torch.arange(10, 10 + num_tokens, dtype=torch.float32).unsqueeze(1)
    b = torch.arange(20, 20 + num_tokens, dtype=torch.float32).unsqueeze(1)
    core_attn_out = torch.empty((num_tokens, 1, 1))

    GatedDeltaNetAttention._forward_core(
        layer, mixed_qkv, b, a, core_attn_out
    )

    assert len(captured_gates) == 1
    actual_a, actual_b = captured_gates[0]
    torch.testing.assert_close(actual_a, a.index_select(0, spec_token_indx))
    torch.testing.assert_close(actual_b, b.index_select(0, spec_token_indx))
    return actual_a, actual_b


def test_qwen3_5_all_speculative_tokens_keep_all_gates(monkeypatch):
    spec_token_indx = torch.arange(4)

    actual_a, actual_b = _run_speculative_gate_selection(
        monkeypatch,
        num_prefills=0,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=torch.empty(0, dtype=torch.long),
    )

    assert actual_a.shape[0] == 4
    assert actual_b.shape[0] == 4


def test_qwen3_5_mixed_batch_selects_speculative_gates(monkeypatch):
    spec_token_indx = torch.tensor([1, 3])

    actual_a, actual_b = _run_speculative_gate_selection(
        monkeypatch,
        num_prefills=1,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=torch.tensor([0, 2]),
    )

    assert actual_a.flatten().tolist() == [11.0, 13.0]
    assert actual_b.flatten().tolist() == [21.0, 23.0]
