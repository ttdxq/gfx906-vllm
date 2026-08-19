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

    q = torch.randn(
        query_len, num_q_heads, head_size, device=device, dtype=dtype
    )
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
    cu_seqlens_q = torch.tensor(
        [0, query_len], device=device, dtype=torch.int32
    )
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).reshape(1, -1)
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
