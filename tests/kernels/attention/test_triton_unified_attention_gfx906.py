import pytest
import torch

import vllm.attention.ops.triton_unified_attention as unified_attn
from vllm.attention.ops.triton_unified_attention import (
    _unified_attention_decode_eager,
    _unified_attention_eager,
)
from vllm.platforms import current_platform


def _is_gfx906() -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_rocm()
        and capability is not None
        and capability.major == 9
        and capability.minor == 0
    )


def test_decode_num_warps_defaults_to_two(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_TRITON_ATTN_DECODE_NUM_WARPS", raising=False)
    monkeypatch.setattr(unified_attn, "_is_gfx906_rocm", lambda: False)
    assert unified_attn._decode_num_warps(256, 4096) == 2


def test_decode_num_warps_uses_four_for_gfx906_wide_head_long_context(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_TRITON_ATTN_DECODE_NUM_WARPS", raising=False)
    monkeypatch.setattr(unified_attn, "_is_gfx906_rocm", lambda: True)
    assert unified_attn._decode_num_warps(129, 1536) == 4
    assert unified_attn._decode_num_warps(160, 4096) == 4
    assert unified_attn._decode_num_warps(192, 4096) == 4
    assert unified_attn._decode_num_warps(256, 4096) == 4


def test_decode_num_warps_keeps_two_for_narrow_or_short_gfx906_decode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_TRITON_ATTN_DECODE_NUM_WARPS", raising=False)
    monkeypatch.setattr(unified_attn, "_is_gfx906_rocm", lambda: True)
    assert unified_attn._decode_num_warps(128, 16384) == 2
    assert unified_attn._decode_num_warps(256, 1535) == 2


def test_decode_num_warps_honors_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_TRITON_ATTN_DECODE_NUM_WARPS", "8")
    monkeypatch.setattr(unified_attn, "_is_gfx906_rocm", lambda: False)
    assert unified_attn._decode_num_warps(128, 1) == 8


def test_decode_num_warps_rejects_invalid_override(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_TRITON_ATTN_DECODE_NUM_WARPS", "3")
    with pytest.raises(ValueError, match="must be one of"):
        unified_attn._decode_num_warps(128, 1)


def test_decode_block_m_uses_eight_for_gfx906_wide_head_gqa6(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(unified_attn, "_is_gfx906_rocm", lambda: True)
    assert unified_attn._decode_block_m(256, 6) == 8
    assert unified_attn._decode_block_m(256, 4) is None
    assert unified_attn._decode_block_m(128, 6) is None


def test_decode_block_m_keeps_default_for_non_gfx906(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(unified_attn, "_is_gfx906_rocm", lambda: False)
    assert unified_attn._decode_block_m(256, 6) is None


def test_num_query_blocks_is_exact_for_single_sequence_decode():
    assert unified_attn._num_query_blocks(1, 1, 1) == 1


@pytest.mark.parametrize("num_seqs", [2, 17])
def test_num_query_blocks_keeps_sequence_mapping_gaps_for_batched_decode(
    num_seqs: int,
):
    assert (
        unified_attn._num_query_blocks(num_seqs, num_seqs, 1) == 2 * num_seqs
    )


@pytest.mark.parametrize(
    ("num_query_tokens", "block_q", "expected"),
    [(2, 1, 2), (3, 1, 3), (3, 2, 2), (8, 3, 3)],
)
def test_num_query_blocks_is_exact_for_single_sequence(
    num_query_tokens: int,
    block_q: int,
    expected: int,
):
    assert (
        unified_attn._num_query_blocks(num_query_tokens, 1, block_q)
        == expected
    )


def test_num_query_blocks_keeps_safe_mixed_batch_upper_bound():
    assert unified_attn._num_query_blocks(7, 3, 2) == 6


def test_unified_attention_decode_eager_matches_gfx906_fallback():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    num_seqs, num_q_heads, num_kv_heads, head_size = 3, 4, 2, 16
    block_size, max_seqlen_k, num_blocks = 8, 19, 32
    q = torch.randn(num_seqs, num_q_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    out_decode = torch.empty_like(q)
    out_eager = torch.empty_like(q)
    cu_seqlens_q = torch.arange(num_seqs + 1, device=device, dtype=torch.int32)
    seqused_k = torch.tensor([19, 7, 13], device=device, dtype=torch.int32)
    block_table = torch.randint(
        0,
        num_blocks,
        (num_seqs, (max_seqlen_k + block_size - 1) // block_size),
        device=device,
        dtype=torch.int32,
    )
    scale = head_size**-0.5

    _unified_attention_eager(
        q=q,
        k=k,
        v=v,
        out=out_eager,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
    )
    _unified_attention_decode_eager(
        q=q,
        k=k,
        v=v,
        out=out_decode,
        seqused_k=seqused_k,
        block_table=block_table,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
    )

    torch.testing.assert_close(out_decode, out_eager, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific Triton path")
@pytest.mark.parametrize(
    ("head_size", "num_queries_per_kv"),
    [(160, 1), (192, 4), (256, 6), (256, 8)],
)
@pytest.mark.parametrize("num_seqs", [1, 2])
@torch.inference_mode()
def test_unified_attention_wide_head_decode_matches_eager(
    head_size: int,
    num_queries_per_kv: int,
    num_seqs: int,
):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    num_kv_heads = 2
    num_q_heads = num_kv_heads * num_queries_per_kv
    block_size = 16
    kv_len = 1536
    num_blocks_per_seq = kv_len // block_size
    num_blocks = num_seqs * num_blocks_per_seq

    q = torch.randn(num_seqs, num_q_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    out_3d = torch.empty_like(q)
    out_eager = torch.empty_like(q)
    cu_seqlens_q = torch.arange(num_seqs + 1, device=device, dtype=torch.int32)
    seqused_k = torch.full((num_seqs,), kv_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        num_seqs, -1
    )
    scale = head_size**-0.5

    _unified_attention_eager(
        q=q,
        k=k,
        v=v,
        out=out_eager,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
    )
    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out_3d,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    torch.testing.assert_close(out_3d, out_eager, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific Triton path")
@pytest.mark.parametrize("query_len", [2, 3])
@pytest.mark.parametrize("kv_len", [512, 4096])
@torch.inference_mode()
def test_unified_attention_multi_query_3d_matches_eager(
    monkeypatch: pytest.MonkeyPatch,
    query_len: int,
    kv_len: int,
):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    num_q_heads, num_kv_heads, head_size = 24, 4, 256
    block_size = 16
    num_blocks = (kv_len + block_size - 1) // block_size

    q = torch.randn(query_len, num_q_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    out_3d = torch.empty_like(q)
    out_eager = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    )
    scale = head_size**-0.5

    _unified_attention_eager(
        q=q,
        k=k,
        v=v,
        out=out_eager,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
    )

    monkeypatch.setattr(unified_attn, "ENABLE_GFX906_ATTN_MULTI_QUERY_3D", True)
    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out_3d,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=query_len,
        seqused_k=seqused_k,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    torch.testing.assert_close(out_3d, out_eager, rtol=1e-3, atol=1e-3)
