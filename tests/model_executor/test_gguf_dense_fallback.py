from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization import gguf
from vllm.model_executor.layers.quantization.gguf import (
    GGUFLinearMethod,
    GGUFUninitializedParameter,
)


def _make_method() -> GGUFLinearMethod:
    method = GGUFLinearMethod(quant_config=SimpleNamespace())
    method.params_dtype = torch.float16
    return method


def _make_merged_layer(prefix: str) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.prefix = prefix
    qweight = GGUFUninitializedParameter(requires_grad=False)
    qweight.shard_id = [0, 1]
    qweight.shard_id_map = {0: 0, 1: 1}
    qweight.data_container = [
        torch.ones((2, 3), dtype=torch.uint8),
        torch.ones((4, 5), dtype=torch.uint8) * 2,
    ]
    layer.register_parameter("qweight", qweight)
    layer.qweight_type = SimpleNamespace(
        weight_type=gguf.WeightType.Q4_K,
        shard_weight_type={
            0: int(gguf.WeightType.Q4_K),
            1: int(gguf.WeightType.Q5_K),
        },
    )
    return layer


def test_merged_mixed_width_gguf_shards_use_standalone_shard_cache():
    method = _make_method()
    layer = _make_merged_layer("model.layers.0.linear_attn.in_proj_qkv")

    method.process_weights_after_loading(layer)

    assert not getattr(layer, "use_dense_gguf_fallback", False)
    assert not hasattr(layer, "weight")
    # Mixed packed widths keep the loader's standalone shard tensors as the
    # single resident copy instead of materializing a zero-padded concat.
    assert isinstance(layer.qweight, GGUFUninitializedParameter)
    assert len(layer.qweight.data_container) == 0
    assert not hasattr(layer.qweight, "shard_offset_map")
    assert set(layer._gguf_shard_cache) == {0, 1}
    torch.testing.assert_close(
        layer._gguf_shard_cache[0],
        torch.ones((2, 3), dtype=torch.uint8),
    )
    torch.testing.assert_close(
        layer._gguf_shard_cache[1],
        torch.ones((4, 5), dtype=torch.uint8) * 2,
    )


def test_merged_mixed_width_gguf_shards_padded_compat(monkeypatch):
    monkeypatch.setattr(gguf, "ENABLE_GGUF_MERGED_PADDED_COMPAT", True)
    method = _make_method()
    layer = _make_merged_layer("model.layers.0.mlp.gate_up_proj")

    method.process_weights_after_loading(layer)

    assert not getattr(layer, "use_dense_gguf_fallback", False)
    assert not hasattr(layer, "weight")
    assert isinstance(layer.qweight, torch.nn.Parameter)
    assert layer.qweight.shape == (6, 5)
    assert layer.qweight.dtype == torch.uint8
    assert layer.qweight.shard_offset_map == {
        0: (0, 2, 3),
        1: (2, 6, 5),
    }
    assert set(layer._gguf_shard_cache) == {0, 1}
    torch.testing.assert_close(
        layer._gguf_shard_cache[0],
        torch.ones((2, 3), dtype=torch.uint8),
    )
    torch.testing.assert_close(
        layer._gguf_shard_cache[1],
        torch.ones((4, 5), dtype=torch.uint8) * 2,
    )
