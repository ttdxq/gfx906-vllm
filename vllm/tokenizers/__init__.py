# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .hf import maybe_make_thread_pool
from .protocol import TokenizerLike
from .registry import (
    TokenizerRegistry,
    cached_get_tokenizer,
    cached_tokenizer_from_config,
    get_tokenizer,
)

__all__ = [
    "MistralTokenizer",
    "TokenizerLike",
    "TokenizerRegistry",
    "cached_get_tokenizer",
    "get_tokenizer",
    "cached_tokenizer_from_config",
    "maybe_make_thread_pool",
]


def __getattr__(name: str):
    # Lazy export: .mistral pulls in entrypoints.chat_utils, which circles
    # back here through config.reasoning during package init.
    if name == "MistralTokenizer":
        from .mistral import MistralTokenizer

        return MistralTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
