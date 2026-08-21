from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, dequantize
from huggingface_hub import snapshot_download

import vllm._custom_ops as ops
from vllm.platforms import current_platform


def _is_gfx906() -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_rocm()
        and capability is not None
        and capability.major == 9
        and capability.minor == 0
    )


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific GGUF path")
@pytest.mark.parametrize("cols", [2048, 2560, 4096, 5120, 6144, 9216, 17408])
@torch.inference_mode()
def test_q4_k_fixed_cols_matches_dequantized_reference(cols: int):
    torch.manual_seed(0)
    sample_dir = Path(snapshot_download("Isotr0py/test-gguf-sample"))
    sample_file = sample_dir / "Quant_Q4_K_1024.gguf"
    sample = GGUFReader(sample_file).tensors[0].data[:64]

    bytes_per_block = 144
    bytes_per_row = cols // 256 * bytes_per_block
    repeats = (bytes_per_row + sample.shape[1] - 1) // sample.shape[1]
    quantized = np.tile(sample, (1, repeats))[:, :bytes_per_row].copy()

    qweight = torch.tensor(quantized, device="cuda")
    weight = torch.tensor(
        dequantize(quantized, GGMLQuantizationType.Q4_K),
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.randn((1, cols), device="cuda", dtype=torch.float16)

    output = ops.ggml_mul_mat_vec_a8(
        qweight, x, GGMLQuantizationType.Q4_K, qweight.shape[0]
    ).to(torch.float16)
    reference = x @ weight.T

    torch.testing.assert_close(output, reference, atol=1, rtol=1e-1)


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific GGUF path")
@pytest.mark.parametrize("cols", [2560, 5120, 6144, 9216, 17408])
@torch.inference_mode()
def test_q5_k_fixed_cols_matches_dequantized_reference(cols: int):
    torch.manual_seed(0)
    sample_dir = Path(snapshot_download("Isotr0py/test-gguf-sample"))
    sample_file = sample_dir / "Quant_Q5_K_1024.gguf"
    sample = GGUFReader(sample_file).tensors[0].data[:64]

    bytes_per_block = 176
    bytes_per_row = cols // 256 * bytes_per_block
    repeats = (bytes_per_row + sample.shape[1] - 1) // sample.shape[1]
    quantized = np.tile(sample, (1, repeats))[:, :bytes_per_row].copy()

    qweight = torch.tensor(quantized, device="cuda")
    weight = torch.tensor(
        dequantize(quantized, GGMLQuantizationType.Q5_K),
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.randn((1, cols), device="cuda", dtype=torch.float16)

    output = ops.ggml_mul_mat_vec_a8(
        qweight, x, GGMLQuantizationType.Q5_K, qweight.shape[0]
    ).to(torch.float16)
    reference = x @ weight.T

    torch.testing.assert_close(output, reference, atol=1, rtol=1e-1)


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific GGUF path")
@torch.inference_mode()
def test_q6_k_fixed_cols_matches_dequantized_reference(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setenv("VLLM_GGUF_Q6_K_FIXED_COLS_MIN_ROWS", "1")
    cols = 5120
    sample_dir = Path(snapshot_download("Isotr0py/test-gguf-sample"))
    sample_file = sample_dir / "Quant_Q6_K_1024.gguf"
    sample = GGUFReader(sample_file).tensors[0].data[:64]

    bytes_per_block = 210
    bytes_per_row = cols // 256 * bytes_per_block
    repeats = (bytes_per_row + sample.shape[1] - 1) // sample.shape[1]
    quantized = np.tile(sample, (1, repeats))[:, :bytes_per_row].copy()

    qweight = torch.tensor(quantized, device="cuda")
    weight = torch.tensor(
        dequantize(quantized, GGMLQuantizationType.Q6_K),
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.randn((1, cols), device="cuda", dtype=torch.float16)

    output = ops.ggml_mul_mat_vec_a8(
        qweight, x, GGMLQuantizationType.Q6_K, qweight.shape[0]
    ).to(torch.float16)
    reference = x @ weight.T

    torch.testing.assert_close(output, reference, atol=1, rtol=1e-1)
