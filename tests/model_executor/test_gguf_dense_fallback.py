from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization import gguf
from vllm.model_executor.layers.quantization.gguf import GGUFLinearMethod


def test_dense_gguf_fallback_matches_input_dtype_for_weight_and_bias():
    method = GGUFLinearMethod(quant_config=SimpleNamespace())
    layer = SimpleNamespace(
        use_dense_gguf_fallback=True,
        weight=torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float16,
        ),
    )
    x = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
    bias = torch.tensor([0.5, -0.5], dtype=torch.float16)

    out = method.apply(layer, x, bias)

    expected = x @ layer.weight.to(dtype=x.dtype).T + bias.to(dtype=x.dtype)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, expected)


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
