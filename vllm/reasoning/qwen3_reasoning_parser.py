# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3/Qwen3.5 model family.

    Qwen3.5 chat templates place <think> in the prompt so the model typically
    generates only </think>. When thinking is disabled, the template places a
    closed think block in the prompt and generated tokens should be treated as
    plain content.

    Older templates may still cause the model to emit <think> directly, so this
    parser accepts both styles.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3/Qwen3.5 default to thinking enabled unless explicitly disabled.
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        self.output_token = "<output>"
        self.assume_reasoning_open = False

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    @staticmethod
    def _clean_text(text: str | None) -> str | None:
        if not text:
            return None
        cleaned = text.lstrip()
        if cleaned.startswith("**"):
            cleaned = cleaned[2:].lstrip()
        return cleaned or None

    def extract_reasoning(self, model_output, request):
        if self.output_token in model_output:
            model_output_parts = model_output.partition(self.start_token)
            model_output = (
                model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
            )
            reasoning, _, content = model_output.partition(self.output_token)
            reasoning = self._clean_text(reasoning)
            content = self._clean_text(content.lstrip("\n"))
            if not self.thinking_enabled:
                return None, content
            return reasoning, content

        if self.end_token not in model_output:
            if not self.thinking_enabled:
                return None, self._clean_text(model_output)
            if self.assume_reasoning_open:
                model_output_parts = model_output.partition(self.start_token)
                reasoning_text = (
                    model_output_parts[2]
                    if model_output_parts[1]
                    else model_output_parts[0]
                )
                return self._clean_text(reasoning_text), None
            if self.start_token in model_output:
                return None, model_output
            return None, self._clean_text(model_output)

        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )
        reasoning, _, content = model_output.partition(self.end_token)
        return self._clean_text(reasoning), self._clean_text(content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        start_seen = self.start_token_id in previous_token_ids
        start_in_delta = self.start_token_id in delta_token_ids

        # Strip <think> from delta if present (old template / edge case where
        # the model generates <think> itself).
        if start_in_delta:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        output_idx = delta_text.find(self.output_token)
        if output_idx >= 0:
            reasoning = self._clean_text(delta_text[:output_idx])
            content = self._clean_text(
                delta_text[output_idx + len(self.output_token) :].lstrip("\n")
            )
            if not self.thinking_enabled:
                return DeltaMessage(content=content)
            return DeltaMessage(
                reasoning=reasoning,
                content=content,
            )

        if self.output_token in previous_text:
            return DeltaMessage(content=delta_text or None)

        if self.end_token_id in delta_token_ids:
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index] or None
                content = delta_text[end_index + len(self.end_token) :] or None
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content,
                )
            return None

        if not delta_text:
            return None
        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text or None)
        if not self.thinking_enabled:
            return DeltaMessage(content=delta_text or None)
        if start_seen or start_in_delta or self.assume_reasoning_open:
            return DeltaMessage(reasoning=delta_text or None)
        return DeltaMessage(content=delta_text or None)
