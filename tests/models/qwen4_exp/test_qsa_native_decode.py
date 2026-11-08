# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.amd.ops.qsa import qsa_sparse_paged_attention
from vllm.models.qwen4_exp.amd.ops.qsa_decode import run_qsa_decode
from vllm.models.qwen4_exp.amd.qsa_indexer import QSAIndexer
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="HIP native QSA submission"
)


def _attention(mrope, context_length=32):
    pages = (context_length + 7) // 16 + 1

    def norm(size):
        return SimpleNamespace(
            weight=torch.randn(size, device="cuda", dtype=torch.float16) * 0.1,
            variance_epsilon=1e-6,
        )

    raw = torch.randn(
        2, 4, 1, 140 if mrope else 128, device="cuda", dtype=torch.float16
    )
    if mrope:
        raw[..., 128:].zero_()
        raw[1, :, 0, 128:].view(torch.int64).copy_(
            torch.arange(context_length - 4, context_length, device="cuda")[
                :, None
            ].expand(4, 3)
        )
    rope = torch.randn(8192, 64, device="cuda", dtype=torch.float16)
    indexer = SimpleNamespace(
        index_n_heads=4,
        index_head_dim=128,
        indexer_dtype=torch.float16,
        compress_ratio=4,
        raw_key_cache=SimpleNamespace(
            kv_cache=raw,
            cache_rope_positions=mrope,
            rope_position_offset=128,
        ),
        compressed_key_cache=SimpleNamespace(
            kv_cache=torch.randn(pages, 4, 1, 128, device="cuda", dtype=torch.float16),
        ),
        q_layernorm=norm(128),
        k_layernorm=norm(128),
        rotary_emb=SimpleNamespace(cos_sin_cache=rope),
        project=lambda hidden: (hidden[:, :512], hidden[:, 512:]),
    )
    if mrope:
        indexer.rotary_emb.mrope_section = (11, 11, 10)
    indexer._fused_prepare = MethodType(QSAIndexer._fused_prepare, indexer)
    indexer._select = MethodType(QSAIndexer._select, indexer)
    indexer.token_topk = 32
    indexer.output_width = 35
    attn = SimpleNamespace(
        indexer=indexer,
        kv_cache=torch.randn(pages, 1, 16, 512, device="cuda", dtype=torch.float16),
        kv_cache_dtype="auto",
        _k_scale_float=1.0,
        _v_scale_float=1.0,
        q_norm=norm(256),
        k_norm=norm(256),
        topk_indices_buffer=torch.empty(1, 35, device="cuda", dtype=torch.int32),
    )

    def forward(layer, query, kv, metadata, output, *, token_to_req):
        key, value = kv.transpose(1, 2).split(256, dim=-1)
        qsa_sparse_paged_attention(
            query,
            key,
            value,
            layer.topk_indices_buffer[:1],
            metadata.block_table,
            token_to_req,
            output,
        )

    attn.impl = SimpleNamespace(forward=forward)
    return attn


@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("token_topk", [32, 2048])
@pytest.mark.parametrize("context_length", [32, 4096])
@torch.inference_mode()
def test_native_qsa_rebinds_inputs_metadata_and_preserves_cache_updates(
    mrope, token_topk, context_length
):
    """Replay exact kernels over advancing state with freshly allocated inputs."""
    import copy

    torch.manual_seed(4047)
    attn = _attention(mrope, context_length)
    attn.indexer.token_topk = token_topk
    attn.indexer.output_width = token_topk + 3
    attn.topk_indices_buffer = torch.empty(
        1, token_topk + 3, device="cuda", dtype=torch.int32
    )
    # Keep parameter/cache layouts identical, with independent state storage.
    reference = copy.copy(attn)
    reference.indexer = copy.copy(attn.indexer)
    reference.indexer.raw_key_cache = copy.copy(attn.indexer.raw_key_cache)
    reference.indexer.compressed_key_cache = copy.copy(
        attn.indexer.compressed_key_cache
    )
    reference.indexer._fused_prepare = MethodType(
        QSAIndexer._fused_prepare, reference.indexer
    )
    reference.indexer._select = MethodType(QSAIndexer._select, reference.indexer)
    reference.topk_indices_buffer = torch.empty_like(attn.topk_indices_buffer)
    for step, position in enumerate(range(context_length - 1, context_length + 7)):
        reference.kv_cache = attn.kv_cache.clone()
        reference.indexer.raw_key_cache.kv_cache = (
            attn.indexer.raw_key_cache.kv_cache.clone()
        )
        reference.indexer.compressed_key_cache.kv_cache = (
            attn.indexer.compressed_key_cache.kv_cache.clone()
        )
        hidden = torch.randn(1, 640, device="cuda", dtype=torch.float16)
        qkv = torch.randn(1, 2 * 13 * 256, device="cuda", dtype=torch.float16)
        pos = torch.tensor([position], device="cuda")
        positions = pos.repeat(3, 1) if mrope else pos
        table = torch.arange(attn.kv_cache.shape[0], device="cuda", dtype=torch.int32)[
            None
        ]
        mapping = torch.tensor(
            [0 if step != 6 else -1], device="cuda", dtype=torch.int32
        )
        raw = SimpleNamespace(
            num_actual_tokens=1,
            slot_mapping=torch.tensor([4 + position % 4], device="cuda"),
            block_table=torch.tensor([[1]], device="cuda", dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            logical_positions=pos,
            token_to_req=mapping,
        )
        compressed = SimpleNamespace(
            slot_mapping=torch.tensor(
                [position // 4 if position % 4 == 3 else -1], device="cuda"
            ),
            k_work_metadata=torch.tensor([[0, 0]], device="cuda", dtype=torch.int32),
            block_table=table,
            token_to_req=mapping,
            logical_positions=pos,
            seq_lens=torch.tensor([position + 1], device="cuda", dtype=torch.int32),
        )
        main = SimpleNamespace(
            slot_mapping=pos,
            block_table=table,
            num_actual_tokens=1,
        )
        output = torch.full(
            (1, 12, 256), float("nan"), device="cuda", dtype=torch.float16
        )
        expected = torch.empty_like(output)
        gate = torch.empty_like(output)
        expected_gate = torch.empty_like(output)
        q, key = reference.indexer.project(hidden)
        iq, (mq, _) = reference.indexer._fused_prepare(
            q,
            key,
            positions,
            raw,
            compressed,
            reference,
            qkv,
            main.slot_mapping,
            expected_gate,
        )
        reference.indexer._select(iq, compressed, reference.topk_indices_buffer)
        reference.impl.forward(
            reference, mq, reference.kv_cache, main, expected, token_to_req=mapping
        )
        run_qsa_decode(
            attn, hidden, positions, qkv, output, gate, main, raw, compressed
        )
        for actual, wanted in [
            (output, expected),
            (gate, expected_gate),
            (attn.kv_cache, reference.kv_cache),
            (attn.topk_indices_buffer, reference.topk_indices_buffer),
            (
                attn.indexer.raw_key_cache.kv_cache,
                reference.indexer.raw_key_cache.kv_cache,
            ),
            (
                attn.indexer.compressed_key_cache.kv_cache,
                reference.indexer.compressed_key_cache.kv_cache,
            ),
        ]:
            if actual is attn.topk_indices_buffer:
                # The existing atomic top-k emits an unordered selection.
                actual = actual.sort().values
                wanted = wanted.sort().values
            if actual is output and context_length > 32:
                # Different valid index orders change FP16 reduction rounding.
                torch.testing.assert_close(actual, wanted, rtol=2e-3, atol=2e-3)
            else:
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert attn._native_decode_plans.builds == 1
    assert attn._native_decode_plans.replays == 7
