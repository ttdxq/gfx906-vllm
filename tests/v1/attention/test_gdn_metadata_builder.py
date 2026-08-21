# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
