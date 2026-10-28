# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
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

_NATIVE_WEIGHT_TYPES = [
    gguf.GGMLQuantizationType[name]
    for name in (
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q8_0",
        "Q2_K",
        "Q3_K",
        "Q4_K",
        "Q5_K",
        "Q6_K",
        "IQ1_S",
        "IQ1_M",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_S",
        "IQ4_NL",
        "IQ4_XS",
    )
]
_TEST_WEIGHT_STORAGE = []


@lru_cache(maxsize=1)
def _quantized_sample_dir():
    return Path(snapshot_download("Isotr0py/test-gguf-sample"))


def _quantized_test_rows(qtype, rows, cols):
    if qtype in (gguf.GGMLQuantizationType.Q4_1, gguf.GGMLQuantizationType.Q5_1):
        rng = np.random.default_rng(2710 + int(qtype))
        return gguf.quantize(
            rng.standard_normal((rows, cols)).astype(np.float32), qtype
        )
    sample_dir = _quantized_sample_dir()
    sample = gguf.GGUFReader(sample_dir / f"Quant_{qtype.name}_1024.gguf")
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    width = cols // block * size
    source = sample.tensors[0].data[:rows]
    return np.tile(source, (1, (width + source.shape[1] - 1) // source.shape[1]))[
        :, :width
    ].copy()


def _dequantized_reference(data, qtype):
    decoded = gguf.dequantize(data, qtype)
    if qtype not in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K):
        return torch.from_numpy(decoded).double()
    # The dequantizer rounds coefficient products in FP16 even for FP32
    # output. HIP contracts its final half multiply/subtract into an FMA.
    _, size = gguf.GGML_QUANT_SIZES[qtype]
    raw = data.reshape(-1, size)
    d = raw[:, :2].copy().view(np.float16).astype(np.float32)
    minimum = raw[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, m = gguf.quants.Q4_K.get_scale_min(raw[:, 4:16].copy())
    scale = (d * sc).reshape(-1, 8, 1)
    offset = (minimum * m).reshape(-1, 8, 1)
    values = decoded.reshape(-1, 8, 32)
    integer = np.divide(
        values + offset, scale, out=np.zeros_like(values), where=scale != 0
    ).round()
    product = scale.astype(np.float16).astype(np.float32) * integer
    if not gguf_layer.current_platform.is_rocm():
        product = product.astype(np.float16).astype(np.float32)
    rounded = (product - offset.astype(np.float16).astype(np.float32)).astype(
        np.float16
    )
    return torch.from_numpy(rounded.reshape(decoded.shape)).double()


def _quantized_dot_reference(data, qtype, quant_x, cols, *, mmq=False):
    """CPU decoded weights plus each format's documented Q8 correction."""
    batch = quant_x.shape[0]
    blocks = quant_x.cpu().view(torch.uint8).reshape(batch, -1, 36)
    ds = blocks[..., :4].contiguous().view(torch.float16).double()
    qs = blocks[..., 4:].contiguous().view(torch.int8).double()
    dq_x = (ds[..., :1] * qs).reshape(batch, -1)[:, :cols]
    weight = torch.from_numpy(gguf.dequantize(data, qtype)).double()
    result = dq_x @ weight.T
    if mmq and qtype == gguf.GGMLQuantizationType.Q5_0:
        # MMQ centers the 5-bit values in its tile; MMVQ uses the original sum.
        return result
    count = cols // 32
    scale_x = ds[:, :count, 0]
    sums_x = ds[:, :count, 1]
    sum_quant = scale_x * qs[:, :count].sum(dim=-1)
    if qtype in (gguf.GGMLQuantizationType.Q4_1, gguf.GGMLQuantizationType.Q5_1):
        _, size = gguf.GGML_QUANT_SIZES[qtype]
        metadata = torch.from_numpy(data.reshape(len(data), -1, size)[..., :4].copy())
        metadata = metadata.view(torch.float16).double()
        dw, minimum = metadata[..., 0], metadata[..., 1]
        unsigned = torch.where(
            dw[..., None] != 0,
            (weight.reshape(len(data), count, 32) - minimum[..., None]) / dw[..., None],
            0,
        ).round()
        dots = torch.einsum("nbi,mbi->mnb", unsigned, qs[:, :count])
        # These two existing kernels multiply the half2 metadata in FP16.
        coefficient = (dw[None] * scale_x[:, None]).half().double()
        offset = (minimum[None] * sums_x[:, None]).half().double()
        return (coefficient * dots + offset).sum(dim=-1)
    if mmq and qtype in (
        gguf.GGMLQuantizationType.Q4_K,
        gguf.GGMLQuantizationType.Q5_K,
    ):
        _, size = gguf.GGML_QUANT_SIZES[qtype]
        raw = data.reshape(-1, size)
        minimum = raw[:, 2:4].copy().view(np.float16).astype(np.float64)
        _, minima = gguf.quants.Q4_K.get_scale_min(raw[:, 4:16].copy())
        offset = torch.from_numpy(-(minimum * minima).reshape(len(data), count))
    elif qtype in (gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q5_0):
        _, size = gguf.GGML_QUANT_SIZES[qtype]
        scales = torch.from_numpy(data.reshape(len(data), -1, size)[..., :2].copy())
        offset = scales.view(torch.float16).double().squeeze(-1)
        offset *= -8 if qtype == gguf.GGMLQuantizationType.Q4_0 else -16
    elif qtype == gguf.GGMLQuantizationType.IQ1_S:
        raw = data.reshape(len(data), -1, 50)
        d = raw[..., :2].copy().view(np.float16).astype(np.float64)
        high = raw[..., 34:].copy().view(np.uint16)
        multiplier = 2 * ((high >> 12) & 7).astype(np.float64) + 1
        delta = np.where((high & 0x8000) == 0, 0.125, -0.125) - 1
        offset = torch.from_numpy((d * multiplier * delta).reshape(len(data), count))
    else:
        return result
    return result + torch.einsum("nb,mb->mn", offset, sums_x - sum_quant)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize("qtype", _NATIVE_WEIGHT_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_all_gguf_weight_formats_match_cpu_reference_and_graph(qtype, dtype):
    """Format coverage must include affine sums, odd rows and Q8 padding."""
    torch.manual_seed(2710 + int(qtype))
    for cols in (768, 1024):
        data = _quantized_test_rows(qtype, 3, cols)
        raw = torch.tensor(data, device="cuda")
        _TEST_WEIGHT_STORAGE.append(raw)
        dequant = ops.ggml_dequantize(raw, int(qtype), 3, cols, torch.float32).cpu()
        torch.testing.assert_close(
            dequant.double(),
            _dequantized_reference(data, qtype),
            atol=0.001,
            rtol=0.001,
        )
        empty = gguf_layer.fused_mul_mat_gguf(
            torch.empty((0, cols), device="cuda", dtype=dtype), raw, int(qtype)
        )
        assert empty.shape == (0, 3)
        for batch in (1, 4, 5, 8):
            x = torch.randn((batch, cols), device="cuda", dtype=dtype) * 0.2
            quant_x = ops.ggml_quantize_row_q8_1(x)
            expected = _quantized_dot_reference(data, qtype, quant_x, cols)
            actual = ops.ggml_mul_mat_vec_q8(raw, quant_x, int(qtype), 3, cols, dtype)
            atol, rtol = (0.001, 2e-4) if dtype == torch.float32 else (0.02, 0.01)
            torch.testing.assert_close(
                actual.cpu().double(), expected, atol=atol, rtol=rtol
            )
            if batch == 1:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = ops.ggml_mul_mat_vec_q8(
                        raw, quant_x, int(qtype), 3, cols, dtype
                    )
                    observed = output.clone()
                    output.fill_(float("nan"))
                for _ in range(2):
                    graph.replay()
                    torch.testing.assert_close(observed, actual, atol=0, rtol=0)
                quant_x.copy_(ops.ggml_quantize_row_q8_1(torch.zeros_like(x)))
                for _ in range(2):
                    graph.replay()
                    torch.testing.assert_close(
                        observed, torch.zeros_like(actual), atol=0, rtol=0
                    )
        if cols == 1024:
            for batch in (17, 65):
                x = torch.randn((batch, cols), device="cuda", dtype=dtype) * 0.2
                quant_x = ops.ggml_quantize_row_q8_1(x)
                expected = _quantized_dot_reference(
                    data, qtype, quant_x, cols, mmq=True
                )
                actual = ops.ggml_mul_mat_a8(raw, x, int(qtype), 3)
                torch.testing.assert_close(
                    actual.cpu().double(), expected, atol=atol, rtol=rtol
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize(
    ("qtype", "wide_env"),
    [
        (gguf.GGMLQuantizationType.IQ2_XXS, "VLLM_GGUF_IQ2_XXS_WIDE"),
        (gguf.GGMLQuantizationType.IQ1_S, "VLLM_GGUF_IQ1_S_WIDE"),
    ],
    ids=["IQ2_XXS", "IQ1_S"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("variant", ["1", "2", "3"])
@torch.inference_mode()
def test_iq_wide_candidates_match_reference_and_graph(
    monkeypatch, qtype, wide_env, dtype, variant
):
    """Exercise IQ decoders against generic MMVQ and CPU math."""
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 experimental IQ routing")
    torch.manual_seed(817 + int(qtype))
    monkeypatch.setenv("VLLM_GGUF_IQ_WIDE", "1")
    monkeypatch.delenv("VLLM_GGUF_IQ2_XXS_WIDE", raising=False)
    monkeypatch.delenv("VLLM_GGUF_IQ1_S_WIDE", raising=False)

    for rows, cols in ((3, 768), (65, 4096), (5, 17408)):
        if rows == 3:
            data = _quantized_test_rows(qtype, rows, cols)
        else:
            block, size = gguf.GGML_QUANT_SIZES[qtype]
            rng = np.random.default_rng(817 + int(qtype) + cols)
            data = rng.integers(
                0, 256, size=(rows, cols // block * size), dtype=np.uint8
            )
            metadata = data.reshape(-1, size)
            metadata[:, :2] = np.array([0.03125], dtype=np.float16).view(np.uint8)
        raw = torch.tensor(data, device="cuda")
        _TEST_WEIGHT_STORAGE.append(raw)
        for batch in (1, 2, 3, 4, 5, 8):
            x = torch.randn((batch, cols), device="cuda", dtype=dtype) * 0.2
            quant_x = ops.ggml_quantize_row_q8_1(x)
            expected = _quantized_dot_reference(data, qtype, quant_x, cols)

            monkeypatch.setenv(wide_env, "0")
            generic = ops.ggml_mul_mat_vec_q8(
                raw, quant_x, int(qtype), rows, cols, dtype
            )
            monkeypatch.setenv(wide_env, variant)
            candidate = ops.ggml_mul_mat_vec_q8(
                raw, quant_x, int(qtype), rows, cols, dtype
            )
            atol, rtol = (0.001, 2e-4) if dtype == torch.float32 else (0.02, 0.01)
            torch.testing.assert_close(
                generic.cpu().double(), expected, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                candidate.cpu().double(), expected, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(candidate, generic, atol=atol, rtol=rtol)

            if batch == 1:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = ops.ggml_mul_mat_vec_q8(
                        raw, quant_x, int(qtype), rows, cols, dtype
                    )
                    observed = output.clone()
                    output.fill_(float("nan"))
                for _ in range(2):
                    graph.replay()
                    torch.testing.assert_close(observed, candidate, atol=0, rtol=0)
                quant_x.copy_(ops.ggml_quantize_row_q8_1(torch.zeros_like(x)))
                for _ in range(2):
                    graph.replay()
                    torch.testing.assert_close(
                        observed, torch.zeros_like(candidate), atol=0, rtol=0
                    )
            monkeypatch.delenv(wide_env, raising=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize(
    ("qtype", "wide_env"),
    [
        (gguf.GGMLQuantizationType.IQ2_XXS, "VLLM_GGUF_IQ2_XXS_WIDE"),
        (gguf.GGMLQuantizationType.IQ1_S, "VLLM_GGUF_IQ1_S_WIDE"),
    ],
    ids=["IQ2_XXS", "IQ1_S"],
)
@torch.inference_mode()
def test_iq_candidate_gates_launch_selected_kernel(monkeypatch, qtype, wide_env):
    """Numerical equivalence alone must not accept an unused candidate gate."""
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 experimental IQ routing")
    raw = torch.tensor(_quantized_test_rows(qtype, 3, 768), device="cuda")
    _TEST_WEIGHT_STORAGE.append(raw)
    x = torch.randn((1, 768), device="cuda", dtype=torch.float16)
    quant_x = ops.ggml_quantize_row_q8_1(x)
    for master, variant, expected in (
        ("1", "0", "mul_mat_vec_q<"),
        ("1", "1", "mul_mat_vec_iq_wide<"),
        ("1", "2", "mul_mat_vec_iq_lut<"),
        ("1", "3", "mul_mat_vec_iq_lut<"),
        ("0", "2", "mul_mat_vec_q<"),
    ):
        monkeypatch.setenv("VLLM_GGUF_IQ_WIDE", master)
        monkeypatch.setenv(wide_env, variant)
        ops.ggml_mul_mat_vec_q8(raw, quant_x, int(qtype), 3, 768, x.dtype)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            ops.ggml_mul_mat_vec_q8(raw, quant_x, int(qtype), 3, 768, x.dtype)
            torch.cuda.synchronize()
        kernels = [
            event.name
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert any(expected in name for name in kernels), kernels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize("qtype", _NATIVE_WEIGHT_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_all_format_candidates_preserve_reference_and_poisoned_output(
    monkeypatch, qtype, dtype
):
    """Candidate scheduling must retain each format's scale and sum semantics."""
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 candidate routing")
    iq = qtype.name.startswith("IQ")
    monkeypatch.setenv("VLLM_GGUF_IQ_WIDE", "1")
    monkeypatch.setenv("VLLM_GGUF_IQ_LUT_TYPE", str(int(qtype)))
    monkeypatch.setenv("VLLM_GGUF_MMVQ_SCHEDULE_TYPE", str(int(qtype)))
    variants = (
        [("2", "0"), ("3", "0")]
        if iq
        else [("4", "0"), ("4", "1"), ("8", "0"), ("8", "1")]
    )
    torch.manual_seed(4100 + int(qtype))
    for cols in (768, 4096, 17408):
        data = _quantized_test_rows(qtype, 5, cols)
        raw = torch.tensor(data, device="cuda")
        _TEST_WEIGHT_STORAGE.append(raw)
        for batch in (1, 4, 5):
            x = torch.randn((batch, cols), device="cuda", dtype=dtype) * 0.2
            quant_x = ops.ggml_quantize_row_q8_1(x)
            expected = _quantized_dot_reference(data, qtype, quant_x, cols)
            atol, rtol = (0.001, 2e-4) if dtype == torch.float32 else (0.02, 0.01)
            monkeypatch.setenv("VLLM_GGUF_IQ_LUT", "0")
            monkeypatch.setenv("VLLM_GGUF_MMVQ_SCHEDULE", "0")
            baseline = ops.ggml_mul_mat_vec_q8(
                raw, quant_x, int(qtype), 5, cols, dtype
            )
            for variant, shared in variants:
                monkeypatch.setenv("VLLM_GGUF_IQ_LUT", variant if iq else "0")
                monkeypatch.setenv("VLLM_GGUF_MMVQ_SCHEDULE", "0" if iq else variant)
                monkeypatch.setenv("VLLM_GGUF_MMVQ_SHARED_X", shared)
                output = torch.full(
                    (batch, 5), float("nan"), device="cuda", dtype=dtype
                )
                ops.ggml_mul_mat_vec_q8_out(raw, quant_x, output, int(qtype), 5, cols)
                torch.testing.assert_close(
                    output.cpu().double(), expected, atol=atol, rtol=rtol
                )
                if int(qtype) in (17, 18, 21, 22, 29):
                    torch.testing.assert_close(output, baseline, atol=0, rtol=0)
                if batch == 1:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output.fill_(float("nan"))
                        ops.ggml_mul_mat_vec_q8_out(
                            raw, quant_x, output, int(qtype), 5, cols
                        )
                        observed = output.clone()
                    for _ in range(2):
                        graph.replay()
                        torch.testing.assert_close(
                            observed.cpu().double(), expected, atol=atol, rtol=rtol
                        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize("qtype", _NATIVE_WEIGHT_TYPES, ids=lambda t: t.name)
@torch.inference_mode()
def test_all_format_candidate_gates_launch_selected_kernel(monkeypatch, qtype):
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 candidate routing")
    iq = qtype.name.startswith("IQ")
    monkeypatch.setenv("VLLM_GGUF_IQ_WIDE", "1")
    monkeypatch.delenv("VLLM_GGUF_IQ_LUT_TYPE", raising=False)
    monkeypatch.delenv("VLLM_GGUF_MMVQ_SCHEDULE_TYPE", raising=False)
    raw = torch.tensor(_quantized_test_rows(qtype, 5, 768), device="cuda")
    _TEST_WEIGHT_STORAGE.append(raw)
    x = torch.randn((1, 768), device="cuda", dtype=torch.float16)
    quant_x = ops.ggml_quantize_row_q8_1(x)
    expected = "mul_mat_vec_iq_lut<" if iq else "mul_mat_vec_q_schedule<"
    for enabled in (False, True):
        monkeypatch.setenv("VLLM_GGUF_IQ_LUT", "2" if iq and enabled else "0")
        monkeypatch.setenv(
            "VLLM_GGUF_MMVQ_SCHEDULE", "4" if not iq and enabled else "0"
        )
        ops.ggml_mul_mat_vec_q8(raw, quant_x, int(qtype), 5, 768, x.dtype)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]) as profile:
            ops.ggml_mul_mat_vec_q8(raw, quant_x, int(qtype), 5, 768, x.dtype)
            torch.cuda.synchronize()
        kernels = [event.name for event in profile.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        if enabled or int(qtype) not in (16, 19):
            assert any(expected in name for name in kernels) == enabled, kernels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_q6_planar_schedule_preserves_reference_and_poisoned_output(monkeypatch, dtype):
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 candidate routing")
    qt = gguf.GGMLQuantizationType.Q6_K
    monkeypatch.setenv("VLLM_GGUF_MMVQ_SCHEDULE_TYPE", "14")
    for rows, cols in ((5, 768), (65, 17408)):
        data = _quantized_test_rows(qt, rows, cols)
        raw = torch.tensor(data, device="cuda")
        packed = torch.ops._C.ggml_planar_pack(raw, int(qt), rows, cols)
        _TEST_WEIGHT_STORAGE.extend([raw, packed])
        x = torch.randn((1, cols), device="cuda", dtype=dtype) * 0.1
        q = ops.ggml_quantize_row_q8_1(x)
        expected = _quantized_dot_reference(data, qt, q, cols)
        for variant, shared in (("4", "0"), ("4", "1"), ("8", "0"), ("8", "1")):
            monkeypatch.setenv("VLLM_GGUF_MMVQ_SCHEDULE", variant)
            monkeypatch.setenv("VLLM_GGUF_MMVQ_SHARED_X", shared)
            y = torch.empty((1, rows), device="cuda", dtype=dtype)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y.fill_(float("nan"))
                ops.ggml_mul_mat_vec_q8_out(packed, q, y, int(qt), rows, cols)
                observed = y.clone()
            for _ in range(2):
                graph.replay()
                atol, rtol = (0.001, 2e-4) if dtype == torch.float32 else (0.02, 0.01)
                torch.testing.assert_close(
                    observed.cpu().double(), expected, atol=atol, rtol=rtol
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_mixed_format_candidates_preserve_sharded_stride_and_graph(monkeypatch, dtype):
    """One output stride must survive mixed formats and odd shard tails."""
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 candidate routing")
    monkeypatch.setenv("VLLM_GGUF_IQ_WIDE", "1")
    monkeypatch.setenv("VLLM_GGUF_IQ_LUT", "2")
    monkeypatch.setenv("VLLM_GGUF_MMVQ_SCHEDULE", "8")
    monkeypatch.setenv("VLLM_GGUF_MMVQ_SHARED_X", "1")
    monkeypatch.delenv("VLLM_GGUF_IQ_LUT_TYPE", raising=False)
    monkeypatch.delenv("VLLM_GGUF_MMVQ_SCHEDULE_TYPE", raising=False)
    types = [
        gguf.GGMLQuantizationType[name]
        for name in ("IQ1_M", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "Q8_0")
    ]
    data = [_quantized_test_rows(t, 3 + 2 * (i % 2), 768) for i, t in enumerate(types)]
    weights = [torch.tensor(d, device="cuda") for d in data]
    _TEST_WEIGHT_STORAGE.extend(weights)
    rows = sum(d.shape[0] for d in data)
    for batch in (1, 2, 3, 4, 5):
        x = torch.randn((batch, 768), device="cuda", dtype=dtype) * 0.1
        q = ops.ggml_quantize_row_q8_1(x)
        expected = torch.cat(
            [_quantized_dot_reference(d, t, q, 768) for d, t in zip(data, types)],
            dim=1,
        )
        y = torch.empty((batch, rows), device="cuda", dtype=dtype)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y.fill_(float("nan"))
            ops.ggml_mul_mat_vec_q8_sharded_out(weights, q, y, types, 768)
            observed = y.clone()
        atol, rtol = (0.001, 2e-4) if dtype == torch.float32 else (0.02, 0.01)
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(
                observed.cpu().double(), expected, atol=atol, rtol=rtol
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_shared_activation_preserves_all_formats_and_graph(monkeypatch, dtype):
    """Sharing must retain single-kernel math, strides and replay overwrite."""
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 shared activation")
    monkeypatch.setenv("VLLM_GGUF_SHARED_ACTIVATION", "1")
    torch.manual_seed(2810)
    for cols in (768, 5120, 17408):
        weights = []
        for qtype in _NATIVE_WEIGHT_TYPES:
            rows = 2049 if qtype == 8 else (65 if qtype in (12, 13, 14) else 5)
            data = _quantized_test_rows(qtype, rows, cols)
            if len(data) < rows:
                data = np.tile(data, ((rows + len(data) - 1) // len(data), 1))[:rows]
            raw = torch.tensor(data, device="cuda")
            _TEST_WEIGHT_STORAGE.append(raw)
            weight = raw
            if qtype in (12, 13, 14):
                weight = torch.ops._C.ggml_planar_pack(raw, int(qtype), rows, cols)
                _TEST_WEIGHT_STORAGE.append(weight)
            weights.append(weight)
        x = torch.randn((4, cols), device="cuda", dtype=dtype) * 0.1
        for batch in (1, 2, 4):
            expected = torch.cat(
                [
                    ops.ggml_mul_mat_vec_a8(w, x[:batch], int(t), w.shape[0])
                    for w, t in zip(weights, _NATIVE_WEIGHT_TYPES)
                ],
                dim=1,
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = ops.ggml_mul_mat_vec_a8_sharded(
                    weights, x[:batch], _NATIVE_WEIGHT_TYPES
                )
                observed = output.clone()
                output.fill_(float("nan"))
            for _ in range(2):
                graph.replay()
                torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        # Exercise both single-representation producers as well as the dual one.
        for indices in ([7, 8], [0, 4, 18]):
            selected = [weights[i] for i in indices]
            types = [_NATIVE_WEIGHT_TYPES[i] for i in indices]
            for prepared in ("0", "1"):
                monkeypatch.setenv("VLLM_GGUF_K45_PREPARED", prepared)
                expected = torch.cat(
                    [
                        ops.ggml_mul_mat_vec_a8(w, x[:1], int(t), w.shape[0])
                        for w, t in zip(selected, types)
                    ],
                    dim=1,
                )
                actual = ops.ggml_mul_mat_vec_a8_sharded(selected, x[:1], types)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        x.zero_()
        graph.replay()
        torch.testing.assert_close(observed, torch.zeros_like(observed), atol=0, rtol=0)


def _make_layer(
    shard_offset_map: dict[int, tuple[int, int, int]],
    shard_weight_type: dict[int, int],
) -> SimpleNamespace:
    qweight = torch.empty((5, 4), dtype=torch.uint8)
    qweight.shard_id = list(shard_offset_map)
    qweight.shard_offset_map = shard_offset_map
    qweight_type = SimpleNamespace(shard_weight_type=shard_weight_type)
    return SimpleNamespace(qweight=qweight, qweight_type=qweight_type)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize(
    "qtype", [gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_prepared_k45_preserves_math_graph_replay_and_fallback(
    monkeypatch, qtype, dtype
):
    """The private producer/consumer must match standard Q8 and fully write tails."""
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 prepared routing")
    torch.manual_seed(2710)
    monkeypatch.setenv("VLLM_GGUF_K45_WIDE", "1")
    monkeypatch.setenv("VLLM_GGUF_K45_INT_ACC", "1")
    for rows, cols in (
        (65, 768),
        (64, 5120),
        (65, 6144),
        (65, 17408),
        (63, 1024),
        (65, 29440),
    ):
        data = _quantized_test_rows(qtype, rows, cols)
        raw = torch.tensor(data, device="cuda")
        packed = torch.ops._C.ggml_planar_pack(raw, int(qtype), rows, cols)
        _TEST_WEIGHT_STORAGE.extend((raw, packed))
        x = torch.randn((4, cols), device="cuda", dtype=dtype) * 0.2
        original_x = x.clone()
        for batch in (1, 2, 4):
            monkeypatch.setenv("VLLM_GGUF_K45_PREPARED", "0")
            quant_x = ops.ggml_quantize_row_q8_1(x[:batch])
            expected = ops.ggml_mul_mat_vec_a8(packed, x[:batch], int(qtype), rows)
            monkeypatch.setenv("VLLM_GGUF_K45_PREPARED", "1")
            actual = ops.ggml_mul_mat_vec_a8(packed, x[:batch], int(qtype), rows)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(
                ops.ggml_quantize_row_q8_1(x[:batch]), quant_x, atol=0, rtol=0
            )
            if batch != 1:
                continue
            reference = _quantized_dot_reference(data, qtype, quant_x, cols)
            atol, rtol = (0.003, 3e-4) if dtype == torch.float32 else (0.03, 0.01)
            torch.testing.assert_close(
                actual.cpu().double(), reference, atol=atol, rtol=rtol
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = ops.ggml_mul_mat_vec_a8(packed, x[:1], int(qtype), rows)
                observed = output.clone()
                output.fill_(float("nan"))
            for zero_input in (False, True):
                if zero_input:
                    x.zero_()
                for _ in range(2):
                    graph.replay()
                    torch.testing.assert_close(
                        observed,
                        torch.zeros_like(expected) if zero_input else expected,
                        atol=0,
                        rtol=0,
                    )
            x.copy_(original_x)
        if cols == 768:
            for alternating in (False, True):
                probe = torch.ones_like(x[:1])
                if alternating:
                    probe[:, 1::2] = -1
                monkeypatch.setenv("VLLM_GGUF_K45_PREPARED", "0")
                expected = ops.ggml_mul_mat_vec_a8(packed, probe, int(qtype), rows)
                monkeypatch.setenv("VLLM_GGUF_K45_PREPARED", "1")
                actual = ops.ggml_mul_mat_vec_a8(packed, probe, int(qtype), rows)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for gate in ("VLLM_GGUF_K45_INT_ACC", "VLLM_GGUF_K45_WIDE"):
            monkeypatch.setenv(gate, "0")
            monkeypatch.setenv("VLLM_GGUF_K45_PREPARED", "0")
            expected = ops.ggml_mul_mat_vec_a8(packed, x[:1], int(qtype), rows)
            monkeypatch.setenv("VLLM_GGUF_K45_PREPARED", "1")
            actual = ops.ggml_mul_mat_vec_a8(packed, x[:1], int(qtype), rows)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            monkeypatch.setenv(gate, "1")
        expected = ops.ggml_mul_mat_vec_a8(packed, x[:1], int(qtype), rows)
        monkeypatch.setenv("VLLM_GGUF_MMVQ_ZERO_OUTPUT", "1")
        torch.testing.assert_close(
            ops.ggml_mul_mat_vec_a8(packed, x[:1], int(qtype), rows),
            expected,
            atol=0,
            rtol=0,
        )
        monkeypatch.delenv("VLLM_GGUF_MMVQ_ZERO_OUTPUT")


def test_q8_1_weight_type_is_rejected():
    """Q8_1 activation records do not implement the GGUF weight format."""
    qtype = gguf.GGMLQuantizationType.Q8_1
    assert set(_NATIVE_WEIGHT_TYPES) == gguf_layer.MMVQ_QUANT_TYPES
    assert set(_NATIVE_WEIGHT_TYPES) == gguf_layer.MMQ_QUANT_TYPES
    assert set(_NATIVE_WEIGHT_TYPES) == gguf_layer.DEQUANT_TYPES
    assert qtype not in gguf_layer.MMVQ_QUANT_TYPES
    assert qtype not in gguf_layer.MMQ_QUANT_TYPES
    assert qtype not in gguf_layer.DEQUANT_TYPES
    with pytest.raises(NotImplementedError, match="Unsupported GGUF"):
        gguf_layer._fused_mul_mat_gguf(
            torch.ones((1, 32)), torch.zeros((3, 40), dtype=torch.uint8), qtype
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@torch.inference_mode()
def test_native_q8_1_weights_fail_explicitly():
    raw = torch.zeros((3, 40), device="cuda", dtype=torch.uint8)
    x = torch.ones((1, 32), device="cuda")
    quant_x = ops.ggml_quantize_row_q8_1(x)
    with pytest.raises(RuntimeError, match="Unsupported GGUF MMVQ"):
        ops.ggml_mul_mat_vec_q8(raw, quant_x, 9, 3, 32, x.dtype)
    with pytest.raises(RuntimeError, match="Unsupported GGUF MMQ"):
        ops.ggml_mul_mat_a8(raw, x, 9, 3)
    with pytest.raises(RuntimeError, match="Unsupported GGUF dequantization"):
        ops.ggml_dequantize(raw, 9, 3, 32, x.dtype)


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


@pytest.mark.parametrize("grouped", [False, True])
def test_compiled_shared_activation_keeps_runtime_batch_selection(monkeypatch, grouped):
    """Prefill tracing must keep sharing available for mixed decode shards."""
    monkeypatch.setattr(gguf_layer, "ENABLE_GGUF_SHARED_ACTIVATION", True)
    types = [gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K]
    layer = _make_layer({0: (0, 2, 4), 1: (2, 5, 4)}, dict(enumerate(types)))
    layer.qweight.copy_(torch.arange(20, dtype=torch.uint8).reshape(5, 4))
    method = GGUFLinearMethod(SimpleNamespace())
    method._cache_gguf_shard_metadata(layer)
    weights = [layer.qweight[:2], layer.qweight[2:]]
    calls = []

    def matmul(weight, x, weight_type, rows):
        calls.append(("single", x.shape[0], rows))
        return x.sum(dim=1, keepdim=True) + weight[:, 0].to(x)

    def sharded(qweights, x, qtypes):
        calls.append(("shared", x.shape[0], sum(w.shape[0] for w in qweights)))
        return x.sum(dim=1, keepdim=True) + torch.cat(qweights)[:, 0].to(x)

    monkeypatch.setattr(gguf_layer.ops, "ggml_mul_mat_vec_a8", matmul)
    monkeypatch.setattr(gguf_layer.ops, "ggml_mul_mat_a8", matmul)
    monkeypatch.setattr(gguf_layer.ops, "ggml_mul_mat_vec_a8_sharded", sharded)
    if grouped:
        layers = []
        for w, t in zip(weights, types):
            sublayer = torch.nn.Module()
            sublayer.qweight = torch.nn.Parameter(w, requires_grad=False)
            sublayer.qweight_type = SimpleNamespace(weight_type=t)
            layers.append(sublayer)

        def apply(x):
            result = gguf_layer.try_grouped_gguf_linear_mmvq(x, layers)
            return torch.cat(result, dim=1)
    else:

        def apply(x):
            return method.apply(layer, x)

    lib = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        lib.impl(
            "_fused_mul_mat_gguf_sharded_mmvq",
            gguf_layer._fused_mul_mat_gguf_sharded_mmvq,
        )
        compiled = torch.compile(apply, backend="eager", dynamic=True, fullgraph=True)
        for batch in (32, 8, 5, 4, 1, 0):
            calls.clear()
            x = torch.ones((batch, 4))
            actual = compiled(x)
            expected = x.sum(dim=1, keepdim=True) + layer.qweight[:, 0].to(x)
            torch.testing.assert_close(actual, expected)
            if 0 < batch <= 4:
                assert calls == [("shared", batch, 5)]
            else:
                assert calls == (
                    [("single", batch, 2), ("single", batch, 3)] if batch else []
                )
        monkeypatch.setattr(gguf_layer, "ENABLE_GGUF_Q8_0_FAST_MMVQ", True)
        calls.clear()
        compiled(torch.ones((1, 4)))
        assert calls == [("single", 1, 2), ("single", 1, 3)]
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.parametrize("qtype", [12, 13])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_planar_k45_overwrites_poisoned_output_on_graph_replay(
    monkeypatch, qtype, dtype
):
    """Every output must be written even when a prior replay left NaNs."""
    if "gfx906" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
        pytest.skip("gfx906 planar routing")
    monkeypatch.setenv("VLLM_GGUF_K45_WIDE", "1")
    quant_type = gguf.GGMLQuantizationType(qtype)
    sample_dir = Path(snapshot_download("Isotr0py/test-gguf-sample"))
    sample = gguf.GGUFReader(sample_dir / f"Quant_{quant_type.name}_1024.gguf")
    block, size = gguf.GGML_QUANT_SIZES[quant_type]
    keepalive = []
    for rows, cols in ((1, 256), (3, 768), (65, 17408)):
        width = cols // block * size
        source = sample.tensors[0].data[:rows]
        repeats = (width + source.shape[1] - 1) // source.shape[1]
        raw = torch.tensor(
            np.tile(source, (1, repeats))[:, :width].copy(), device="cuda"
        )
        packed = torch.ops._C.ggml_planar_pack(raw, qtype, rows, cols)
        keepalive.extend((raw, packed))
        for batch in (0, 1, 2, 3, 4):
            x = torch.randn((batch, cols), device="cuda", dtype=dtype)
            quant_x = ops.ggml_quantize_row_q8_1(x) if batch else raw[:0]
            if not batch:
                assert ops.ggml_mul_mat_vec_q8(
                    packed, quant_x, qtype, rows, cols, dtype
                ).shape == (0, rows)
                continue
            monkeypatch.setenv("VLLM_GGUF_MMVQ_ZERO_OUTPUT", "1")
            expected = ops.ggml_mul_mat_vec_q8(
                packed, quant_x, qtype, rows, cols, dtype
            )
            monkeypatch.setenv("VLLM_GGUF_MMVQ_ZERO_OUTPUT", "0")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = ops.ggml_mul_mat_vec_q8(
                    packed, quant_x, qtype, rows, cols, dtype
                )
                observed = output.clone()
                output.fill_(float("nan"))
            for zero_input in (False, True):
                if zero_input:
                    quant_x.copy_(ops.ggml_quantize_row_q8_1(torch.zeros_like(x)))
                for _ in range(2):
                    graph.replay()
                    assert torch.isfinite(observed).all()
                    torch.testing.assert_close(
                        observed,
                        torch.zeros_like(expected) if zero_input else expected,
                        atol=0,
                        rtol=0,
                    )
