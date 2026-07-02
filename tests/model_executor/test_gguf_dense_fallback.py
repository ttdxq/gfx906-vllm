from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization import gguf
from vllm.model_executor.layers.quantization.gguf import GGUFLinearMethod


def test_gfx906_qwen35_linear_attn_gguf_uses_packed_weight(monkeypatch):
    method = GGUFLinearMethod(quant_config=SimpleNamespace())
    method.params_dtype = torch.float16
    monkeypatch.setattr(gguf, "on_gfx906", lambda: True)

    qweight = torch.nn.Parameter(
        torch.empty(0, dtype=torch.uint8),
        requires_grad=False,
    )
    qweight.shard_id = ["qkv", "z"]
    qweight.shard_id_map = {"qkv": 0, "z": 1}
    qweight.data_container = [
        torch.ones((2, 3), dtype=torch.uint8),
        torch.ones((4, 5), dtype=torch.uint8) * 2,
    ]

    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.linear_attn.in_proj_qkv"
    layer.register_parameter("qweight", qweight)
    layer.qweight_type = SimpleNamespace(weight_type=gguf.WeightType.Q4_K)

    method.process_weights_after_loading(layer)

    assert not getattr(layer, "use_dense_gguf_fallback", False)
    assert layer.qweight is not None
    assert not hasattr(layer, "weight")
    assert layer.qweight.shape == (6, 5)
    assert layer.qweight.dtype == torch.uint8
    assert layer.qweight.shard_offset_map == {
        "qkv": (0, 2, 3),
        "z": (2, 6, 5),
    }
    assert set(layer._gguf_contiguous_shard_cache) == {"qkv"}
    torch.testing.assert_close(
        layer._gguf_contiguous_shard_cache["qkv"],
        torch.ones((2, 3), dtype=torch.uint8),
    )


def test_gfx906_non_linear_attn_gguf_uses_packed_weight(monkeypatch):
    method = GGUFLinearMethod(quant_config=SimpleNamespace())
    method.params_dtype = torch.float16
    monkeypatch.setattr(gguf, "on_gfx906", lambda: True)

    qweight = torch.nn.Parameter(
        torch.empty(0, dtype=torch.uint8),
        requires_grad=False,
    )
    qweight.shard_id = [0, 1]
    qweight.shard_id_map = {0: 0, 1: 1}
    qweight.data_container = [
        torch.ones((2, 3), dtype=torch.uint8),
        torch.ones((4, 5), dtype=torch.uint8) * 2,
    ]

    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.gate_up_proj"
    layer.register_parameter("qweight", qweight)
    layer.qweight_type = SimpleNamespace(weight_type=gguf.WeightType.Q4_K)

    method.process_weights_after_loading(layer)

    assert not getattr(layer, "use_dense_gguf_fallback", False)
    assert layer.qweight is not None
    assert not hasattr(layer, "weight")
    assert layer.qweight.shape == (6, 5)
    assert layer.qweight.dtype == torch.uint8
    assert layer.qweight.shard_offset_map == {
        0: (0, 2, 3),
        1: (2, 6, 5),
    }
    assert set(layer._gguf_contiguous_shard_cache) == {0}
    torch.testing.assert_close(
        layer._gguf_contiguous_shard_cache[0],
        torch.ones((2, 3), dtype=torch.uint8),
    )
