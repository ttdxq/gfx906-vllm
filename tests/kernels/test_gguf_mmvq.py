# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import gguf
import torch

from vllm.model_executor.layers.quantization.gguf import (
    _get_full_mmvq_shard_weight,
)


def _make_layer(
    shard_offset_map: dict[int, tuple[int, int, int]],
    shard_weight_type: dict[int, int],
) -> SimpleNamespace:
    qweight = torch.empty((5, 4), dtype=torch.uint8)
    qweight.shard_id = list(shard_offset_map)
    qweight.shard_offset_map = shard_offset_map
    qweight_type = SimpleNamespace(shard_weight_type=shard_weight_type)
    return SimpleNamespace(qweight=qweight, qweight_type=qweight_type)


def test_full_mmvq_shard_weight_accepts_contiguous_same_type_shards():
    qtype = int(gguf.GGMLQuantizationType.Q5_K)
    layer = _make_layer(
        {0: (0, 2, 4), 1: (2, 5, 4)},
        {0: qtype, 1: qtype},
    )

    assert _get_full_mmvq_shard_weight(layer, [qtype, qtype]) == (
        layer.qweight,
        qtype,
    )


def test_full_mmvq_shard_weight_rejects_mixed_types():
    q5 = int(gguf.GGMLQuantizationType.Q5_K)
    q4 = int(gguf.GGMLQuantizationType.Q4_K)
    layer = _make_layer(
        {0: (0, 2, 4), 1: (2, 5, 4)},
        {0: q5, 1: q4},
    )

    assert _get_full_mmvq_shard_weight(layer, [q5, q4]) is None


def test_full_mmvq_shard_weight_rejects_padded_or_gapped_shards():
    qtype = int(gguf.GGMLQuantizationType.Q5_K)
    padded = _make_layer(
        {0: (0, 2, 4), 1: (2, 5, 3)},
        {0: qtype, 1: qtype},
    )
    gapped = _make_layer(
        {0: (0, 2, 4), 1: (3, 5, 4)},
        {0: qtype, 1: qtype},
    )

    assert _get_full_mmvq_shard_weight(padded, [qtype, qtype]) is None
    assert _get_full_mmvq_shard_weight(gapped, [qtype, qtype]) is None
