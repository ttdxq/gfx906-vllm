import torch

from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as fsg


def test_gfx906_decode_sigmoid_gating_matches_eager_fallback():
    torch.manual_seed(0)
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens, heads, kv_heads, key_dim, value_dim = 5, 2, 4, 8, 3
    q = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(1, tokens, kv_heads, value_dim, device=device, dtype=dtype)
    a = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    b = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    A_log = torch.randn(kv_heads, device=device, dtype=dtype)
    dt_bias = torch.randn(kv_heads, device=device, dtype=dtype)
    initial_state = torch.randn(
        tokens, kv_heads, value_dim, key_dim, device=device, dtype=dtype
    )

    eager_state = initial_state.clone()
    decode_state = initial_state.clone()
    cu_seqlens = torch.arange(tokens + 1, device=device, dtype=torch.int32)
    scale = key_dim**-0.5

    eager_out, eager_final = fsg._fused_sigmoid_gating_delta_rule_update_gfx906_eager(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        scale=scale,
        initial_state=eager_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )
    decode_out, decode_final = fsg._fused_sigmoid_gating_delta_rule_decode_gfx906(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        scale=scale,
        initial_state=decode_state,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(decode_out, eager_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(decode_final, eager_final, rtol=1e-3, atol=1e-3)
