import torch

from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet


class _FakeQuantConfig:
    def get_name(self) -> str:
        return "gguf"


class _FakeLinear:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output

    def __call__(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        del hidden_states
        return self.output, None


class _IdentityNorm:
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


def test_qwen3_5_gguf_split_gdn_expands_qk_in_tiled_v_head_order():
    module = object.__new__(Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(module)
    module.split_projections = True
    module.quant_config = _FakeQuantConfig()
    module.num_k_heads = 2
    module.num_v_heads = 4

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

    query = torch.tensor([[[[10.0], [20.0]]]])
    key = torch.tensor([[[[30.0], [40.0]]]])

    expanded_query, expanded_key = module._expand_qk_heads_for_gdn(query, key)

    assert expanded_query.flatten().tolist() == [10.0, 10.0, 20.0, 20.0]
    assert expanded_key.flatten().tolist() == [30.0, 30.0, 40.0, 40.0]
