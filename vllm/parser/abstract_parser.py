# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

from vllm.entrypoints.chat_utils import make_tool_call_id

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        DeltaMessage,
        ExtractedToolCallInformation,
        FunctionCall,
        ResponsesRequest,
    )
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
    from vllm.tokenizers import TokenizerLike
    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
        ToolParser,
    )


@dataclass
class StreamState:
    """Mutable state for parser-engine streaming."""

    reasoning_ended: bool = False
    tool_call_text_started: bool = False
    prompt_reasoning_checked: bool = False
    previous_text: str = ""
    previous_token_ids: list[int] = field(default_factory=list)
    history_tool_call_cnt: int = 0
    history_tool_call_cnt_initialized: bool = False
    tool_call_id_type: str = "random"
    function_name_returned: bool = False
    engine_based: bool = False

    def advance(
        self,
        delta_text: str,
        delta_token_ids: list[int],
    ) -> tuple[str, list[int]]:
        if self.engine_based:
            return delta_text, delta_token_ids
        return (
            self.previous_text + delta_text,
            self.previous_token_ids + delta_token_ids,
        )

    def commit(self, current_text: str, current_token_ids: list[int]) -> None:
        if self.engine_based:
            self.previous_text = ""
            self.previous_token_ids = []
        else:
            self.previous_text = current_text
            self.previous_token_ids = current_token_ids


class Parser(ABC):
    """Small compatibility surface used by parser engines."""

    reasoning_parser_cls: type["ReasoningParser"] | None = None
    tool_parser_cls: type["ToolParser"] | None = None

    def __init__(
        self,
        tokenizer: "TokenizerLike",
        tools: list | None = None,
        *args,
        model_config=None,
        **kwargs,
    ) -> None:
        self.model_tokenizer = tokenizer
        self._reasoning_parser: ReasoningParser | None = None
        self._tool_parser: ToolParser | None = None
        self._stream_state = StreamState()

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    @property
    def reasoning_parser(self) -> "ReasoningParser | None":
        return self._reasoning_parser

    @reasoning_parser.setter
    def reasoning_parser(self, parser: "ReasoningParser | None") -> None:
        self._reasoning_parser = parser

    @property
    def tool_parser(self) -> "ToolParser | None":
        return self._tool_parser

    @tool_parser.setter
    def tool_parser(self, parser: "ToolParser | None") -> None:
        self._tool_parser = parser

    def _make_tool_call_id(self, function_name: str) -> str | None:
        state = self._stream_state
        if state.tool_call_id_type != "kimi_k2":
            return None
        tool_call_id = make_tool_call_id(
            id_type=state.tool_call_id_type,
            func_name=function_name,
            idx=state.history_tool_call_cnt,
        )
        state.history_tool_call_cnt += 1
        return tool_call_id

    @abstractmethod
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        ...

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        ...

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        ...

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        ...

    @abstractmethod
    def extract_tool_calls(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> "ExtractedToolCallInformation":
        ...

    @abstractmethod
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> "DeltaMessage | None":
        ...

    @abstractmethod
    def parse(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list["FunctionCall"] | None]:
        ...

    @abstractmethod
    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: "ChatCompletionRequest | ResponsesRequest",
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> "DeltaMessage | None":
        ...
