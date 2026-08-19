# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.models.qwen3_5 import Qwen3_5RMSNorm
from vllm.model_executor.models.qwen3_5_mtp import (
    _maybe_convert_gguf_rms_norm_weight,
)


@pytest.mark.parametrize(
    "is_gguf,module,expected",
    [
        (True, Qwen3_5RMSNorm(2), torch.tensor([0.5, 1.5])),
        (False, Qwen3_5RMSNorm(2), torch.tensor([1.5, 2.5])),
        (True, nn.Linear(2, 2, bias=False), torch.tensor([1.5, 2.5])),
    ],
)
def test_mtp_gguf_rms_norm_weight_conversion(is_gguf, module, expected):
    weight = torch.tensor([1.5, 2.5])

    converted = _maybe_convert_gguf_rms_norm_weight(
        "model.norm.weight", weight, {"model.norm": module}, is_gguf
    )

    torch.testing.assert_close(converted, expected)
