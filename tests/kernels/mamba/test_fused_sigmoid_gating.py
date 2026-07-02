import torch

from vllm import _custom_ops as ops
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


def test_gfx906_decode_sigmoid_gating_accepts_float32_state():
    torch.manual_seed(0)
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens, heads, kv_heads, key_dim, value_dim = 3, 2, 4, 8, 3
    q = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(1, tokens, kv_heads, value_dim, device=device, dtype=dtype)
    a = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    b = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    A_log = torch.randn(kv_heads, device=device, dtype=dtype)
    dt_bias = torch.randn(kv_heads, device=device, dtype=dtype)
    initial_state = torch.randn(
        tokens,
        kv_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
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


def test_gfx906_indexed_decode_sigmoid_gating_updates_source_state():
    torch.manual_seed(0)
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens, slots, heads, kv_heads, key_dim, value_dim = 3, 7, 2, 4, 8, 3
    q = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(1, tokens, kv_heads, value_dim, device=device, dtype=dtype)
    a = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    b = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    A_log = torch.randn(kv_heads, device=device, dtype=dtype)
    dt_bias = torch.randn(kv_heads, device=device, dtype=dtype)
    state_indices = torch.tensor([3, 1, 5], device=device, dtype=torch.int32)
    source_state = torch.randn(
        slots,
        kv_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )

    gathered_state = source_state[state_indices].clone()
    direct_state = source_state.clone()
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
        initial_state=gathered_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )
    direct_out, direct_final = (
        fsg._fused_sigmoid_gating_delta_rule_indexed_decode_gfx906(
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
            initial_state=direct_state,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
    )

    torch.testing.assert_close(direct_out, eager_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        direct_final[state_indices], eager_final, rtol=1e-3, atol=1e-3
    )
    untouched = torch.ones(slots, device=device, dtype=torch.bool)
    untouched[state_indices.long()] = False
    torch.testing.assert_close(direct_final[untouched], source_state[untouched])


def test_gfx906_indexed_decode_sigmoid_gating_updates_kv_cache_state():
    torch.manual_seed(0)
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens, slots, heads, kv_heads, key_dim, value_dim = 3, 7, 2, 4, 8, 3
    q = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(1, tokens, kv_heads, value_dim, device=device, dtype=dtype)
    a = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    b = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    A_log = torch.randn(kv_heads, device=device, dtype=dtype)
    dt_bias = torch.randn(kv_heads, device=device, dtype=dtype)
    state_indices = torch.tensor([3, 1, 5], device=device, dtype=torch.int32)
    cache_state = torch.randn(
        slots,
        kv_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )

    gathered_state = cache_state[state_indices].transpose(-1, -2).contiguous()
    direct_state = cache_state.clone()
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
        initial_state=gathered_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )
    direct_out, direct_final = (
        fsg.fused_sigmoid_gating_delta_rule_update_kv_cache_gfx906(
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
            initial_state=direct_state,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
    )

    torch.testing.assert_close(direct_out, eager_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        direct_final[state_indices].transpose(-1, -2),
        eager_final,
        rtol=1e-3,
        atol=1e-3,
    )
    untouched = torch.ones(slots, device=device, dtype=torch.bool)
    untouched[state_indices.long()] = False
    torch.testing.assert_close(direct_final[untouched], cache_state[untouched])


def test_gfx906_packed_decode_matches_indexed_decode():
    torch.manual_seed(0)
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens, slots, heads, kv_heads, key_dim, value_dim = 3, 7, 2, 4, 8, 3
    q = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(1, tokens, kv_heads, value_dim, device=device, dtype=dtype)
    a = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    b = torch.randn(tokens, kv_heads, device=device, dtype=dtype)
    A_log = torch.randn(kv_heads, device=device, dtype=dtype)
    dt_bias = torch.randn(kv_heads, device=device, dtype=dtype)
    state_indices = torch.tensor([3, 1, 5], device=device, dtype=torch.int32)
    source_state = torch.randn(
        slots,
        kv_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )
    mixed_qkv = torch.cat(
        (
            q.squeeze(0).reshape(tokens, -1),
            k.squeeze(0).reshape(tokens, -1),
            v.squeeze(0).reshape(tokens, -1),
        ),
        dim=-1,
    ).contiguous()
    scale = key_dim**-0.5

    indexed_state = source_state.clone()
    indexed_out, _ = fsg._fused_sigmoid_gating_delta_rule_indexed_decode_gfx906(
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
        initial_state=indexed_state,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    packed_state = source_state.clone()
    packed_out = torch.empty(
        tokens,
        1,
        kv_heads,
        value_dim,
        device=device,
        dtype=dtype,
    )
    ops.fused_recurrent_gated_delta_rule_gfx906_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        state=packed_state,
        out=packed_out,
        state_indices=state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(
        packed_out.squeeze(1), indexed_out.squeeze(0), rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(packed_state, indexed_state, rtol=1e-3, atol=1e-3)
