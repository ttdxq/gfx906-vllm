# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import os
import sys
from functools import lru_cache

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .op import exp

ENABLE_QWEN35_RATIO2_FUSED_DECODE = os.getenv(
    "VLLM_QWEN35_RATIO2_FUSED_DECODE", "1"
).lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _is_gfx906_rocm() -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_rocm()
        and capability is not None
        and capability.major == 9
        and capability.minor == 0
    )


def _lookup_state_index(
    ssm_state_indices: torch.Tensor | None,
    seq_idx: int,
    token_idx: int,
) -> int:
    if ssm_state_indices is None:
        return seq_idx
    if ssm_state_indices.ndim == 1:
        return int(ssm_state_indices[seq_idx].item())
    return int(ssm_state_indices[seq_idx, token_idx].item())


def _maybe_l2norm(x: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    return (
        x_float * torch.rsqrt(torch.sum(x_float * x_float, dim=-1, keepdim=True) + 1e-6)
    ).to(x.dtype)


def _fused_recurrent_gated_delta_rule_fwd_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_qk_l2norm_in_kernel:
        q = _maybe_l2norm(q)
        k = _maybe_l2norm(k)

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    head_ratio = HV // H
    if H == 0 or HV % H != 0:
        raise ValueError(f"Invalid GDN head mapping: H={H}, HV={HV}")

    o = torch.empty_like(v)
    if inplace_final_state:
        final_state = initial_state
    else:
        state_dtype = initial_state.dtype if initial_state is not None else q.dtype
        final_state = q.new_empty(T, HV, V, K, dtype=state_dtype)

    if cu_seqlens is None:
        seq_ranges = [(b, b, 0, T) for b in range(B)]
    else:
        seq_ranges = [
            (0, i, int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item()))
            for i in range(len(cu_seqlens) - 1)
        ]

    for batch_idx, seq_idx, seq_start, seq_end in seq_ranges:
        if initial_state is None:
            state = torch.zeros((HV, V, K), device=q.device, dtype=torch.float32)
        else:
            init_token_idx = 0
            if num_accepted_tokens is not None:
                init_token_idx = int(num_accepted_tokens[seq_idx].item()) - 1
                init_token_idx = max(init_token_idx, 0)
            state_index = _lookup_state_index(
                ssm_state_indices, seq_idx, init_token_idx
            )
            state = initial_state[state_index].float().clone()

        for token_idx in range(seq_start, seq_end):
            local_token_idx = token_idx - seq_start
            for head_idx in range(HV):
                q_head_idx = head_idx // head_ratio
                q_t = q[batch_idx, token_idx, q_head_idx].float() * scale
                k_t = k[batch_idx, token_idx, q_head_idx].float()
                v_t = v[batch_idx, token_idx, head_idx].float()
                state_head = state[head_idx]

                if g is not None:
                    state_head = state_head * torch.exp(
                        g[batch_idx, token_idx, head_idx].float()
                    )

                v_residual = v_t - torch.sum(state_head * k_t[None, :], dim=1)
                beta_t = beta[batch_idx, token_idx, head_idx].float()
                v_residual = v_residual * beta_t
                state_head = state_head + v_residual[:, None] * k_t[None, :]

                o[batch_idx, token_idx, head_idx] = torch.sum(
                    state_head * q_t[None, :], dim=1
                ).to(o.dtype)
                state[head_idx] = state_head

            if inplace_final_state:
                state_index = _lookup_state_index(
                    ssm_state_indices, seq_idx, local_token_idx
                )
                final_state[state_index] = state.to(final_state.dtype)
            else:
                final_state[token_idx] = state.to(final_state.dtype)

    return o, final_state


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv

    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * HV + i_hv) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = (
                h0
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                * stride_init_state_token
            )
        else:
            p_h0 = h0 + bos * HV * K * V
        p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[:, None])
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            p_ht = (
                ht
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                * stride_final_state_token
            )
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_gfx906_rocm():
        return _fused_recurrent_gated_delta_rule_fwd_eager(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, K, V, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state


@triton.jit
def fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    SPLIT_BATCH_HEAD_GRID: tl.constexpr,
):
    if SPLIT_BATCH_HEAD_GRID:
        i_v, i_hv, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    else:
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    use_tiled_qk_head_mapping: bool = False,
    use_transposed_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_gfx906_rocm():
        try:
            output = ops.fused_recurrent_gated_delta_rule_gfx906_packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                state=initial_state,
                out=out,
                state_indices=ssm_state_indices,
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                use_tiled_qk_head_mapping=use_tiled_qk_head_mapping,
                use_transposed_state=use_transposed_state,
            )
            return output, initial_state
        except (AttributeError, RuntimeError) as exc:
            if os.getenv("VLLM_QWEN35_RATIO2_FUSED_DECODE_DEBUG", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                print(
                    "QWEN35_RATIO2_FUSED_DECODE_FALLBACK "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            pass

        B = mixed_qkv.shape[0]
        HV, V, K = initial_state.shape[-3:]
        qkv_dim = mixed_qkv.shape[1]
        qk_dim = qkv_dim - HV * V
        if qk_dim <= 0 or qk_dim % 2 != 0:
            raise ValueError(
                f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
            )
        q_dim = qk_dim // 2
        if q_dim % K != 0:
            raise ValueError(
                f"Invalid packed Q size {q_dim}: must be divisible by K={K}."
            )
        H = q_dim // K
        if H <= 0 or HV % H != 0:
            raise ValueError(
                f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
            )
        if use_transposed_state and K != V:
            raise ValueError(
                "Packed decode with transposed cache state requires K == V "
                f"(got K={K}, V={V})."
            )

        state_indices = ssm_state_indices.to(dtype=torch.long)
        valid_mask = state_indices >= 0
        safe_indices = state_indices.clamp(0, initial_state.shape[0] - 1)
        head_ratio = HV // H
        output = torch.zeros_like(out)
        final_state = initial_state

        for head_idx in range(HV):
            if use_tiled_qk_head_mapping:
                q_head_idx = head_idx % H
            else:
                q_head_idx = head_idx // head_ratio
            q_offset = q_head_idx * K
            k_offset = H * K + q_head_idx * K
            v_offset = 2 * H * K + head_idx * V

            q_t = mixed_qkv[:, q_offset : q_offset + K].float()
            k_t = mixed_qkv[:, k_offset : k_offset + K].float()
            v_t = mixed_qkv[:, v_offset : v_offset + V].float()

            if use_qk_l2norm_in_kernel:
                q_t = _maybe_l2norm(q_t)
                k_t = _maybe_l2norm(k_t)
            q_t = q_t * scale

            x = a[:, head_idx].float() + dt_bias[head_idx].float()
            softplus_x = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
            g_val = -torch.exp(A_log[head_idx].float()) * softplus_x
            beta_val = torch.sigmoid(b[:, head_idx].float())

            state_view = final_state[:, head_idx]
            state_head = state_view.index_select(0, safe_indices).float()
            state_head = state_head * torch.exp(g_val).view(B, 1, 1)
            if use_transposed_state:
                v_residual = v_t - torch.sum(state_head * k_t[:, :, None], dim=1)
                v_residual = v_residual * beta_val.view(B, 1)
                state_head = state_head + k_t[:, :, None] * v_residual[:, None, :]
                head_out = torch.sum(state_head * q_t[:, :, None], dim=1).to(
                    output.dtype
                )
            else:
                v_residual = v_t - torch.sum(state_head * k_t[:, None, :], dim=2)
                v_residual = v_residual * beta_val.view(B, 1)
                state_head = state_head + v_residual[:, :, None] * k_t[:, None, :]
                head_out = torch.sum(state_head * q_t[:, None, :], dim=2).to(
                    output.dtype
                )
            output[:, 0, head_idx] = torch.where(
                valid_mask.view(B, 1), head_out, torch.zeros_like(head_out)
            )

            current_state = state_view.index_select(0, safe_indices)
            updated_state = torch.where(
                valid_mask.view(B, 1, 1),
                state_head.to(final_state.dtype),
                current_state,
            )
            state_view.index_copy_(0, safe_indices, updated_state)

        return output, final_state

    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    dev = mixed_qkv.device
    if (
        a.device != dev
        or b.device != dev
        or A_log.device != dev
        or dt_bias.device != dev
        or initial_state.device != dev
        or out.device != dev
        or ssm_state_indices.device != dev
    ):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements (got A_log.numel()={A_log.numel()}, dt_bias.numel()={dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(
            f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK})."
        )
    BV = min(triton.next_power_of_2(V), 32)
    num_stages = 3
    num_warps = 1

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    NV = triton.cdiv(V, BV)
    # CUDA limits grid Y/Z dimensions to 65535.
    split_batch_head_grid = B * HV > 65535
    grid = (NV, HV, B) if split_batch_head_grid else (NV, B * HV)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        SPLIT_BATCH_HEAD_GRID=split_batch_head_grid,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state


def causal_conv1d_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    pad_slot_id: int,
    silu_activation: bool,
    use_qk_l2norm_in_kernel: bool = False,
    use_tiled_qk_head_mapping: bool = False,
    use_transposed_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_gfx906_rocm():
        raise RuntimeError("combined packed decode is only available on gfx906 ROCm")

    qkv_dim = mixed_qkv.shape[1]
    kv_heads, value_dim, key_dim = initial_state.shape[-3:]
    qk_dim = qkv_dim - kv_heads * value_dim
    heads = qk_dim // (2 * key_dim) if qk_dim > 0 else 0
    can_use_ratio2_fused_decode = (
        ENABLE_QWEN35_RATIO2_FUSED_DECODE
        and mixed_qkv.ndim == 2
        and conv_state.ndim == 3
        and conv_weight.ndim == 2
        and a.ndim == 2
        and b.ndim == 2
        and initial_state.ndim == 4
        and out.ndim == 4
        and ssm_state_indices.ndim == 1
        and use_qk_l2norm_in_kernel
        and key_dim == value_dim == 128
        and qk_dim > 0
        and qk_dim % (2 * key_dim) == 0
        and heads > 0
        and kv_heads == 2 * heads
        and conv_state.shape[1] == qkv_dim
        and conv_weight.shape[0] == qkv_dim
        and conv_state.dtype == initial_state.dtype
        and (not use_transposed_state or key_dim == value_dim)
    )
    if can_use_ratio2_fused_decode:
        try:
            output = ops.causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode(
                mixed_qkv=mixed_qkv,
                conv_state=conv_state,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                state=initial_state,
                out=out,
                state_indices=ssm_state_indices,
                pad_slot_id=pad_slot_id,
                scale=scale,
                silu_activation=silu_activation,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                use_tiled_qk_head_mapping=use_tiled_qk_head_mapping,
                use_transposed_state=use_transposed_state,
            )
            return output, initial_state
        except (AttributeError, RuntimeError):
            pass

    output = ops.causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode(
        mixed_qkv=mixed_qkv,
        conv_state=conv_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        state=initial_state,
        out=out,
        state_indices=ssm_state_indices,
        pad_slot_id=pad_slot_id,
        scale=scale,
        silu_activation=silu_activation,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_tiled_qk_head_mapping=use_tiled_qk_head_mapping,
        use_transposed_state=use_transposed_state,
    )
    return output, initial_state


class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

        return o, final_state


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        inplace_final_state: bool:
            Whether to store the final state in-place to save memory.
            Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        ssm_state_indices (Optional[torch.Tensor]):
            Indices to map the input sequences to the initial/final states.
        num_accepted_tokens (Optional[torch.Tensor]):
            Number of accepted tokens for each sequence during decoding.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state
