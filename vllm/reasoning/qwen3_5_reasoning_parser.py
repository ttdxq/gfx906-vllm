# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser


class Qwen3_5ReasoningParser(Qwen3ReasoningParser):
    """Backward-compatible alias for Qwen3.5-specific parser selection."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.assume_reasoning_open = self.thinking_enabled
