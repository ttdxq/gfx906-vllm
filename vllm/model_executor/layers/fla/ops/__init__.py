# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from .chunk import chunk_gated_delta_rule
from .fused_gdn_prefill_post_conv import fused_post_conv_prep
from .fused_recurrent import (
    causal_conv1d_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from .fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_kv_cache_gfx906,
)
from .layernorm_guard import RMSNormGated

__all__ = [
    "RMSNormGated",
    "chunk_gated_delta_rule",
    "fused_post_conv_prep",
    "causal_conv1d_recurrent_gated_delta_rule_packed_decode",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_sigmoid_gating_delta_rule_update",
    "fused_sigmoid_gating_delta_rule_update_kv_cache_gfx906",
]
