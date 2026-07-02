import torch

from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)


def _packed_decode_ref(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    ssm_state_indices,
    use_qk_l2norm_in_kernel,
):
    batch = mixed_qkv.shape[0]
    hv, value_dim, key_dim = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - hv * value_dim
    heads = qk_dim // (2 * key_dim)
    head_ratio = hv // heads
    output = torch.zeros(
        batch, 1, hv, value_dim, device=mixed_qkv.device, dtype=mixed_qkv.dtype
    )
    final_state = initial_state.clone()

    for batch_idx in range(batch):
        state_idx = int(ssm_state_indices[batch_idx].item())
        if state_idx < 0:
            continue
        state = final_state[state_idx].float().clone()
        for head_idx in range(hv):
            q_head_idx = head_idx // head_ratio
            q_offset = q_head_idx * key_dim
            k_offset = heads * key_dim + q_head_idx * key_dim
            v_offset = 2 * heads * key_dim + head_idx * value_dim
            q_t = mixed_qkv[batch_idx, q_offset : q_offset + key_dim].float()
            k_t = mixed_qkv[batch_idx, k_offset : k_offset + key_dim].float()
            v_t = mixed_qkv[batch_idx, v_offset : v_offset + value_dim].float()
            if use_qk_l2norm_in_kernel:
                q_t = q_t * torch.rsqrt(torch.sum(q_t * q_t) + 1e-6)
                k_t = k_t * torch.rsqrt(torch.sum(k_t * k_t) + 1e-6)
            q_t = q_t * scale
            x = a[batch_idx, head_idx].float() + dt_bias[head_idx].float()
            softplus_x = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
            g_val = -torch.exp(A_log[head_idx].float()) * softplus_x
            beta_val = torch.sigmoid(b[batch_idx, head_idx].float())
            state_head = state[head_idx] * torch.exp(g_val)
            v_residual = v_t - torch.sum(state_head * k_t[None, :], dim=1)
            state_head = state_head + (v_residual * beta_val)[:, None] * k_t[None, :]
            output[batch_idx, 0, head_idx] = torch.sum(
                state_head * q_t[None, :], dim=1
            ).to(output.dtype)
            state[head_idx] = state_head
        final_state[state_idx] = state.to(final_state.dtype)
    return output, final_state


def _packed_decode_transposed_state_ref(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    ssm_state_indices,
    use_qk_l2norm_in_kernel,
    use_tiled_qk_head_mapping,
):
    batch = mixed_qkv.shape[0]
    hv, value_dim, key_dim = initial_state.shape[-3:]
    assert value_dim == key_dim
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - hv * value_dim
    heads = qk_dim // (2 * key_dim)
    head_ratio = hv // heads
    output = torch.zeros(
        batch, 1, hv, value_dim, device=mixed_qkv.device, dtype=mixed_qkv.dtype
    )
    final_state = initial_state.clone()

    for batch_idx in range(batch):
        state_idx = int(ssm_state_indices[batch_idx].item())
        if state_idx < 0:
            continue
        state = final_state[state_idx].float().clone()
        for head_idx in range(hv):
            q_head_idx = (
                head_idx % heads
                if use_tiled_qk_head_mapping
                else head_idx // head_ratio
            )
            q_offset = q_head_idx * key_dim
            k_offset = heads * key_dim + q_head_idx * key_dim
            v_offset = 2 * heads * key_dim + head_idx * value_dim
            q_t = mixed_qkv[batch_idx, q_offset : q_offset + key_dim].float()
            k_t = mixed_qkv[batch_idx, k_offset : k_offset + key_dim].float()
            v_t = mixed_qkv[batch_idx, v_offset : v_offset + value_dim].float()
            if use_qk_l2norm_in_kernel:
                q_t = q_t * torch.rsqrt(torch.sum(q_t * q_t) + 1e-6)
                k_t = k_t * torch.rsqrt(torch.sum(k_t * k_t) + 1e-6)
            q_t = q_t * scale
            x = a[batch_idx, head_idx].float() + dt_bias[head_idx].float()
            softplus_x = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
            g_val = -torch.exp(A_log[head_idx].float()) * softplus_x
            beta_val = torch.sigmoid(b[batch_idx, head_idx].float())
            state_head = state[head_idx] * torch.exp(g_val)
            v_residual = v_t - torch.sum(state_head * k_t[:, None], dim=0)
            state_head = state_head + k_t[:, None] * (v_residual * beta_val)[None, :]
            output[batch_idx, 0, head_idx] = torch.sum(
                state_head * q_t[:, None], dim=0
            ).to(output.dtype)
            state[head_idx] = state_head
        final_state[state_idx] = state.to(final_state.dtype)
    return output, final_state


def test_gfx906_packed_decode_matches_reference():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    batch, heads, hv, key_dim, value_dim = 4, 2, 4, 8, 3
    mixed_qkv = torch.randn(
        batch,
        2 * heads * key_dim + hv * value_dim,
        device=device,
        dtype=dtype,
    )
    a = torch.randn(batch, hv, device=device, dtype=dtype)
    b = torch.randn(batch, hv, device=device, dtype=dtype)
    A_log = torch.randn(hv, device=device, dtype=dtype)
    dt_bias = torch.randn(hv, device=device, dtype=dtype)
    initial_state = torch.randn(6, hv, value_dim, key_dim, device=device, dtype=dtype)
    indices = torch.tensor([2, -1, 4, 1], device=device, dtype=torch.int32)
    scale = key_dim**-0.5

    out = torch.empty(batch, 1, hv, value_dim, device=device, dtype=dtype)
    actual_out, actual_state = fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state.clone(),
        out=out,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    expected_out, expected_state = _packed_decode_ref(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-3, atol=1e-3)


def test_gfx906_packed_decode_matches_transposed_qwen_cache_reference():
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    batch, heads, hv, key_dim, value_dim = 4, 2, 4, 8, 8
    mixed_qkv = torch.randn(
        batch,
        2 * heads * key_dim + hv * value_dim,
        device=device,
        dtype=dtype,
    )
    a = torch.randn(batch, hv, device=device, dtype=dtype)
    b = torch.randn(batch, hv, device=device, dtype=dtype)
    A_log = torch.randn(hv, device=device, dtype=dtype)
    dt_bias = torch.randn(hv, device=device, dtype=dtype)
    initial_state = torch.randn(6, hv, value_dim, key_dim, device=device, dtype=torch.float32)
    indices = torch.tensor([2, -1, 4, 1], device=device, dtype=torch.int32)
    scale = key_dim**-0.5

    out = torch.empty(batch, 1, hv, value_dim, device=device, dtype=dtype)
    actual_out, actual_state = fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state.clone(),
        out=out,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        use_tiled_qk_head_mapping=True,
        use_transposed_state=True,
    )
    expected_out, expected_state = _packed_decode_transposed_state_ref(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        use_tiled_qk_head_mapping=True,
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-3, atol=1e-3)


def test_gfx906_packed_decode_matches_qwen35_shared_qk_reference():
    torch.manual_seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    batch, heads, hv, key_dim, value_dim = 3, 16, 32, 128, 128
    mixed_qkv = torch.randn(
        batch,
        2 * heads * key_dim + hv * value_dim,
        device=device,
        dtype=dtype,
    )
    a = torch.randn(batch, hv, device=device, dtype=dtype)
    b = torch.randn(batch, hv, device=device, dtype=dtype)
    A_log = torch.randn(hv, device=device, dtype=dtype)
    dt_bias = torch.randn(hv, device=device, dtype=dtype)
    initial_state = torch.randn(
        6, hv, value_dim, key_dim, device=device, dtype=torch.float32
    )
    indices = torch.tensor([2, -1, 4], device=device, dtype=torch.int32)
    scale = key_dim**-0.5

    out = torch.empty(batch, 1, hv, value_dim, device=device, dtype=dtype)
    actual_out, actual_state = fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state.clone(),
        out=out,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        use_tiled_qk_head_mapping=True,
        use_transposed_state=True,
    )
    expected_out, expected_state = _packed_decode_transposed_state_ref(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        use_tiled_qk_head_mapping=True,
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-3, atol=1e-3)
