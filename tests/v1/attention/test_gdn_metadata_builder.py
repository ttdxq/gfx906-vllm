# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch
from transformers import LlamaConfig

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


def _create_stateless_builder(full_cuda_graph: bool = False):
    builder = object.__new__(GDNAttentionMetadataBuilder)
    builder.use_spec_decode = False
    builder.use_full_cuda_graph = full_cuda_graph
    builder._building_for_capture = False
    return builder


def test_decode_does_not_transfer_context_lens():
    builder = _create_stateless_builder()

    batch = BatchSpec(seq_lens=[80, 96], query_lens=[1, 1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    context_lens_to = Mock(wraps=common.num_computed_tokens_cpu.to)
    common.num_computed_tokens_cpu.to = context_lens_to

    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    context_lens_to.assert_not_called()
    assert meta.num_prefills == 0
    assert meta.has_initial_state is None


def test_full_cudagraph_spec_metadata_uses_request_count(tmp_path):
    num_speculative_tokens = 3
    LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=4,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    vllm_config = create_vllm_config(
        model_name=str(tmp_path),
        block_size=BLOCK_SIZE,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=num_speculative_tokens,
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
        num_accepted_tokens=torch.ones(batch.batch_size, dtype=torch.int32),
    )

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def _build_non_spec(batch: BatchSpec):
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    builder = _create_stateless_builder()
    return builder, common, builder.build(common_prefix_len=0, common_attn_metadata=common)


@pytest.mark.parametrize(
    ("seq_len", "query_len", "num_prefills"),
    [
        pytest.param(1, 1, 1, id="first-chunk"),
        pytest.param(65, 1, 0, id="resumed-chunk"),
        pytest.param(0, 0, 0, id="padding"),
    ],
)
def test_one_token_chunk_classification(
    seq_len: int,
    query_len: int,
    num_prefills: int,
):
    """Only a real first chunk with no prior state needs state initialization."""
    _, _, meta = _build_non_spec(
        BatchSpec(seq_lens=[100, seq_len], query_lens=[1, query_len])
    )

    assert meta.num_prefills == num_prefills
    assert meta.num_decodes == 2 - num_prefills
    assert meta.num_prefill_tokens == num_prefills
    assert meta.num_decode_tokens == 1 + query_len - num_prefills
    if num_prefills:
        assert meta.has_initial_state is not None
        assert meta.has_initial_state.tolist() == [True, False]
    else:
        assert meta.has_initial_state is None


def test_one_token_first_chunk_excludes_padding():
    """Neither padding requests nor padding tokens count as prefill work."""
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 1, 0, 0], query_lens=[1, 1, 0, 0]),
        BLOCK_SIZE,
        DEVICE,
    )
    builder = _create_stateless_builder()
    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert meta.num_decodes == 1
    assert meta.num_prefills == 1
    assert meta.num_decode_tokens == 1
    assert meta.num_prefill_tokens == 1


def test_cudagraph_capture_batch_stays_decode_only(tmp_path):
    """Capture rows have no history, but must still select decode kernels."""
    LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=4,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    vllm_config = create_vllm_config(
        model_name=str(tmp_path),
        block_size=BLOCK_SIZE,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )
    batch = BatchSpec(seq_lens=[1] * 4, query_lens=[1] * 4)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    meta = builder.build_for_cudagraph_capture(common)

    assert meta.num_prefills == 0
    assert meta.num_decodes == 4
    assert meta.has_initial_state is None
    staged = meta.non_spec_state_indices_tensor
    assert staged is not None
    assert staged.data_ptr() == builder.non_spec_state_indices_tensor.data_ptr()
    torch.testing.assert_close(staged, common.block_table_tensor[:, 0])
