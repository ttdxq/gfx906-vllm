# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization import gguf as gguf_layer
from vllm.model_executor.layers.quantization.gguf import (
    GGUFLinearMethod,
    _get_full_mmvq_shard_weight,
)


def _make_layer(
    shard_offset_map: dict[int, tuple[int, int, int]],
    shard_weight_type: dict[int, int],
) -> SimpleNamespace:
    qweight = torch.empty((5, 4), dtype=torch.uint8)
    qweight.shard_id = list(shard_offset_map)
    qweight.shard_offset_map = shard_offset_map
    qweight_type = SimpleNamespace(shard_weight_type=shard_weight_type)
    return SimpleNamespace(qweight=qweight, qweight_type=qweight_type)


def test_full_mmvq_shard_weight_accepts_contiguous_same_type_shards():
    qtype = int(gguf.GGMLQuantizationType.Q5_K)
    layer = _make_layer(
        {0: (0, 2, 4), 1: (2, 5, 4)},
        {0: qtype, 1: qtype},
    )

    assert _get_full_mmvq_shard_weight(layer, [qtype, qtype]) == (
        layer.qweight,
        qtype,
    )


def test_full_mmvq_shard_weight_rejects_mixed_types():
    q5 = int(gguf.GGMLQuantizationType.Q5_K)
    q4 = int(gguf.GGMLQuantizationType.Q4_K)
    layer = _make_layer(
        {0: (0, 2, 4), 1: (2, 5, 4)},
        {0: q5, 1: q4},
    )

    assert _get_full_mmvq_shard_weight(layer, [q5, q4]) is None


def test_full_mmvq_shard_weight_rejects_padded_or_gapped_shards():
    qtype = int(gguf.GGMLQuantizationType.Q5_K)
    padded = _make_layer(
        {0: (0, 2, 4), 1: (2, 5, 3)},
        {0: qtype, 1: qtype},
    )
    gapped = _make_layer(
        {0: (0, 2, 4), 1: (3, 5, 4)},
        {0: qtype, 1: qtype},
    )

    assert _get_full_mmvq_shard_weight(padded, [qtype, qtype]) is None
    assert _get_full_mmvq_shard_weight(gapped, [qtype, qtype]) is None


@pytest.mark.parametrize(
    "qtype",
    [
        gguf.GGMLQuantizationType.Q6_K,
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.Q4_K,
        gguf.GGMLQuantizationType.IQ4_XS,
    ],
)
def test_compiled_full_shards_keep_one_runtime_dispatch(monkeypatch, qtype):
    """Large-batch tracing must not bake two shard calls into decode."""
    layer = _make_layer({0: (0, 2, 4), 1: (2, 5, 4)}, {0: qtype, 1: qtype})
    layer.qweight.copy_(torch.arange(20, dtype=torch.uint8).reshape(5, 4))
    method = GGUFLinearMethod(SimpleNamespace())
    method._cache_gguf_shard_metadata(layer)
    calls = []

    # CPU surrogates isolate dispatch and compiler behavior from GPU math.
    def matmul(weight, x, weight_type, rows):
        calls.append((x.shape[0], rows))
        return x.sum(dim=1, keepdim=True) + weight[:, 0].to(x)

    monkeypatch.setattr(gguf_layer.ops, "ggml_mul_mat_vec_a8", matmul)
    monkeypatch.setattr(gguf_layer.ops, "ggml_mul_mat_a8", matmul)
    bias = torch.arange(5, dtype=torch.float32)
    lib = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        lib.impl("_fused_mul_mat_gguf", gguf_layer._fused_mul_mat_gguf)
        compiled = torch.compile(
            lambda x: method.apply(layer, x, bias),
            backend="eager",
            dynamic=True,
            fullgraph=True,
        )
        for batch in (32, 8, 5, 4, 1, 0):
            calls.clear()
            x = torch.ones((batch, 4))
            actual = compiled(x)
            expected = x.sum(dim=1, keepdim=True) + layer.qweight[:, 0].to(x) + bias
            torch.testing.assert_close(actual, expected)
            assert calls == ([(batch, 5)] if batch else [])
    finally:
        torch._dynamo.reset()
        lib._destroy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@torch.inference_mode()
def test_planar_merged_weights_have_no_original_layout_shard_views():
    """Every consumer must use the whole planar buffer after packing."""
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 planar routing")
    sample_dir = Path(snapshot_download("Isotr0py/test-gguf-sample"))
    method = GGUFLinearMethod(SimpleNamespace())
    keepalive = []
    for qtype in (
        gguf.GGMLQuantizationType.Q4_K,
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.Q6_K,
    ):
        sample = gguf.GGUFReader(sample_dir / f"Quant_{qtype.name}_1024.gguf")
        raw = torch.tensor(sample.tensors[0].data[:6].copy(), device="cuda")
        keepalive.append(raw)
        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(raw.clone(), requires_grad=False)
        layer.qweight.shard_id = [0, 1]
        layer.qweight.shard_offset_map = {
            0: (0, 3, raw.shape[1]),
            1: (3, 6, raw.shape[1]),
        }
        layer.qweight_type = SimpleNamespace(
            weight_type=0, shard_weight_type={0: int(qtype), 1: int(qtype)}
        )
        method._maybe_planar_pack_kquants(layer)
        method._cache_gguf_shards(layer)
        method._cache_gguf_shard_metadata(layer)
        keepalive.append(layer)
        weights, types = gguf_layer._collect_gguf_linear_shards(layer)
        assert weights == [layer.qweight]
        assert types == [int(qtype)]
        assert layer.qweight.shard_id == []
        assert not getattr(layer, "_gguf_shard_cache", None)
        for batch in (0, 1, 4, 5, 8, 32):
            x = torch.randn((batch, 1024), device="cuda", dtype=torch.float16)
            actual = method.apply(layer, x)
            expected = gguf_layer.fused_mul_mat_gguf(x, raw, int(qtype))
            torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.002)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@torch.inference_mode()
def test_planar_k45_matches_explicit_quantized_dot_product(monkeypatch):
    """Late scaling must preserve the dot product, including cancellation."""
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 planar routing")
    monkeypatch.setenv("VLLM_GGUF_K45_WIDE", "1")
    sample_dir = Path(snapshot_download("Isotr0py/test-gguf-sample"))
    keepalive = []
    torch.manual_seed(27)
    for qtype in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K):
        sample = gguf.GGUFReader(sample_dir / f"Quant_{qtype.name}_1024.gguf")
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        source = sample.tensors[0].data[:64]
        for cols in (768, 5120, 17408):
            width = cols // block * size
            repeats = (width + source.shape[1] - 1) // source.shape[1]
            data = np.tile(source, (1, repeats))[:, :width].copy()
            raw = torch.tensor(data, device="cuda")
            packed = torch.ops._C.ggml_planar_pack(raw, int(qtype), len(data), cols)
            keepalive.extend((raw, packed))
            weight = torch.from_numpy(gguf.dequantize(data, qtype)).double()
            x = torch.randn((4, cols), device="cuda", dtype=torch.float32)
            x[1].fill_(1)
            x[2].fill_(1)
            x[2, 1::2] = -1
            x[3].zero_()
            for batch in (1, 4):
                quant_x = ops.ggml_quantize_row_q8_1(x[:batch])
                blocks = quant_x.cpu().view(torch.uint8).reshape(batch, -1, 36)
                scale = blocks[..., :2].contiguous().view(torch.float16).double()
                values = blocks[..., 4:].contiguous().view(torch.int8).double()
                dequant_x = (scale * values).reshape(batch, -1)[:, :cols]
                expected = dequant_x @ weight.T
                for variant in ("0", "1"):
                    monkeypatch.setenv("VLLM_GGUF_K45_INT_ACC", variant)
                    actual = (
                        ops.ggml_mul_mat_vec_q8(
                            packed, quant_x, int(qtype), len(data), cols, torch.float32
                        )
                        .cpu()
                        .double()
                    )
                    torch.testing.assert_close(actual, expected, atol=0.001, rtol=2e-5)
                    if batch == 4:
                        assert torch.count_nonzero(actual[3]).item() == 0
