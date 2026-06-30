import torch

from vllm.attention.ops.triton_unified_attention import (
    _unified_attention_decode_eager,
    _unified_attention_eager,
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
