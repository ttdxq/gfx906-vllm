# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import torch

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    _replace_video_token_placeholders,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

IMAGE_TOKEN_ID = 999
VIDEO_TOKEN_ID = 888
VISION_START_TOKEN_ID = 777
VISION_END_TOKEN_ID = 778


@dataclass
class _VisionConfig:
    spatial_merge_size: int = 1


@dataclass
class _Config:
    image_token_id: int = IMAGE_TOKEN_ID
    video_token_id: int = VIDEO_TOKEN_ID
    vision_start_token_id: int = VISION_START_TOKEN_ID
    vision_end_token_id: int = VISION_END_TOKEN_ID
    vision_config: _VisionConfig = field(default_factory=_VisionConfig)


def _video_feature(grid_thw, length):
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    modality="video",
                    key="video_grid_thw",
                    data=torch.tensor(grid_thw),
                    field=None,
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=length),
    )


def test_replace_video_token_placeholders():
    prompt = [1, 10, 11, 2, 10, 11, 3]
    replacements = [[20, 21], [30, 31, 32]]

    assert _replace_video_token_placeholders(prompt, [10, 11], replacements) == [
        1,
        20,
        21,
        2,
        30,
        31,
        32,
        3,
    ]


def test_recompute_qwen3vl_mrope_after_evs_pruning():
    config = _Config()
    grid_thw = (2, 2, 2)
    first_frame = (
        [20, VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * 4 + [VISION_END_TOKEN_ID]
    )
    second_frame = (
        [21, VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * 4 + [VISION_END_TOKEN_ID]
    )
    media_tokens = torch.tensor(first_frame + second_frame)
    retention_mask = torch.ones(len(media_tokens), dtype=torch.bool)
    retention_mask[len(first_frame) + 4 : len(first_frame) + 6] = False
    pruned_media_tokens = media_tokens[retention_mask]

    prefix = [11]
    suffix = [12]
    input_tokens = prefix + media_tokens.tolist() + suffix
    pruned_input_tokens = prefix + pruned_media_tokens.tolist() + suffix

    expected_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=[_video_feature(grid_thw, len(input_tokens))],
        config=config,
    )
    media_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=media_tokens.tolist(),
        mm_features=[_video_feature(grid_thw, len(media_tokens))],
        config=config,
    )

    expanded_positions = torch.zeros((len(pruned_media_tokens), 5), dtype=torch.long)
    expanded_positions[:, :3] = media_mrope[:, retention_mask].T
    expanded_positions[:, 3] = pruned_media_tokens.eq(VISION_START_TOKEN_ID)
    expanded_positions[:, 4] = pruned_media_tokens.eq(VIDEO_TOKEN_ID)
    multimodal_embeddings = [
        torch.cat(
            [torch.zeros((len(pruned_media_tokens), 8)), expanded_positions.float()],
            dim=1,
        )
    ]

    whole_retention_mask = torch.cat(
        [
            torch.ones(1, dtype=torch.bool),
            retention_mask,
            torch.ones(1, dtype=torch.bool),
        ]
    )
    expected_pruned_mrope = expected_mrope[:, whole_retention_mask]
    initial_mrope = torch.zeros_like(expected_pruned_mrope)
    initial_mrope[:, :1] = expected_pruned_mrope[:, :1]

    embeddings, actual_mrope, _ = (
        Qwen3VLForConditionalGeneration._recompute_mrope_positions(
            input_ids=pruned_input_tokens,
            multimodal_embeddings=multimodal_embeddings,
            mrope_positions=initial_mrope,
            num_computed_tokens=1,
            vision_start_token_id=VISION_START_TOKEN_ID,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
        )
    )

    assert embeddings[0].shape == (len(pruned_media_tokens), 8)
    assert torch.equal(actual_mrope, expected_pruned_mrope)
