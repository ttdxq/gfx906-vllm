from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike


def merge_kwargs(
    defaults: dict[str, object] | None,
    overrides: dict[str, object] | None,
    /,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, object]:
    base = {} if defaults is None else dict(defaults)
    if overrides is None:
        return base
    return base | {
        key: value for key, value in overrides.items() if value not in unset_values
    }


def recursively_merge_kwargs(
    defaults: dict[str, object] | None,
    overrides: dict[str, object] | None,
    /,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, object]:
    merged = {} if defaults is None else dict(defaults)
    if overrides is None:
        return merged
    for key, value in overrides.items():
        if value in unset_values:
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = recursively_merge_kwargs(
                merged[key], value, unset_values=unset_values
            )
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class ChatParams:
    chat_template: str | None = None
    chat_template_content_format: object = "auto"
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, object]] | None = None
    mm_processor_kwargs: dict[str, object] | None = None

    def with_defaults(
        self,
        default_chat_template_kwargs: dict[str, object] | None = None,
        default_media_io_kwargs: dict[str, dict[str, object]] | None = None,
        default_mm_processor_kwargs: dict[str, object] | None = None,
    ) -> ChatParams:
        if (
            not default_chat_template_kwargs
            and not default_media_io_kwargs
            and not default_mm_processor_kwargs
        ):
            return self
        return ChatParams(
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                default_chat_template_kwargs, self.chat_template_kwargs
            ),
            media_io_kwargs=merge_kwargs(default_media_io_kwargs, self.media_io_kwargs),
            mm_processor_kwargs=recursively_merge_kwargs(
                default_mm_processor_kwargs, self.mm_processor_kwargs
            ),
        )

    def get_apply_chat_template_kwargs(self) -> dict[str, object]:
        return merge_kwargs(
            self.chat_template_kwargs,
            {"chat_template": self.chat_template, "return_dict": False},
        )


@dataclass(frozen=True)
class TokenizeParams:
    max_total_tokens: int | None
    max_output_tokens: int = 0
    pad_prompt_tokens: int | None = None
    truncate_prompt_tokens: int | None = None
    truncation_side: str | None = None
    do_lower_case: bool = False
    add_special_tokens: bool = True
    return_token_offsets: bool = False
    needs_detokenization: bool = False
    max_total_tokens_param: str = "max_total_tokens"
    max_output_tokens_param: str = "max_output_tokens"
    truncate_prompt_tokens_param: str = "truncate_prompt_tokens"

    @property
    def max_input_tokens(self) -> int | None:
        if self.max_total_tokens is None:
            return None
        return self.max_total_tokens - self.max_output_tokens

    def __post_init__(self) -> None:
        if self.truncation_side not in (None, "left", "right"):
            raise VLLMValidationError(
                "`truncation_side` must be either 'left' or 'right'.",
                parameter="truncation_side",
                value=self.truncation_side,
            )
        if (
            self.max_total_tokens is not None
            and self.max_output_tokens > self.max_total_tokens
        ):
            raise VLLMValidationError(
                f"{self.max_output_tokens_param}={self.max_output_tokens} cannot be "
                f"greater than {self.max_total_tokens_param}={self.max_total_tokens}. "
                "Please request fewer output tokens.",
                parameter=self.max_output_tokens_param,
                value=self.max_output_tokens,
            )
        if (
            self.max_input_tokens is not None
            and self.truncate_prompt_tokens is not None
            and self.truncate_prompt_tokens > self.max_input_tokens
        ):
            raise VLLMValidationError(
                f"{self.truncate_prompt_tokens_param}={self.truncate_prompt_tokens} "
                f"cannot be greater than {self.max_total_tokens_param} - "
                f"{self.max_output_tokens_param} = {self.max_input_tokens}. "
                "Please request a smaller truncation size.",
                parameter=self.truncate_prompt_tokens_param,
                value=self.truncate_prompt_tokens,
            )

    def with_kwargs(self, **tokenization_kwargs: object) -> TokenizeParams:
        max_length = tokenization_kwargs.pop("max_length", self.max_input_tokens)
        return replace(
            self,
            max_total_tokens=None
            if max_length is None
            else max_length + self.max_output_tokens,
            pad_prompt_tokens=tokenization_kwargs.pop(
                "pad_prompt_tokens", self.pad_prompt_tokens
            ),
            truncate_prompt_tokens=tokenization_kwargs.pop(
                "truncate_prompt_tokens", self.truncate_prompt_tokens
            ),
            truncation_side=tokenization_kwargs.pop(
                "truncation_side", self.truncation_side
            ),
            do_lower_case=bool(
                tokenization_kwargs.pop("do_lower_case", self.do_lower_case)
            ),
            add_special_tokens=bool(
                tokenization_kwargs.pop("add_special_tokens", self.add_special_tokens)
            ),
            return_token_offsets=bool(
                tokenization_kwargs.pop(
                    "return_token_offsets", self.return_token_offsets
                )
            ),
            needs_detokenization=bool(
                tokenization_kwargs.pop(
                    "needs_detokenization", self.needs_detokenization
                )
            ),
        )

    def get_encode_kwargs(self) -> dict[str, object]:
        max_length = self.truncate_prompt_tokens
        if max_length is not None and max_length < 0:
            max_length = self.max_input_tokens
        elif max_length is None and self.max_input_tokens is not None:
            # Keep one extra token so the post-tokenization check can reject
            # overlong prompts without tokenizing the full input.
            max_length = self.max_input_tokens + 1

        if self.truncation_side is not None and self.truncate_prompt_tokens is not None:
            return {
                "truncation": False,
                "add_special_tokens": self.add_special_tokens,
            }

        return {
            "truncation": max_length is not None,
            "max_length": max_length,
            "add_special_tokens": self.add_special_tokens,
        }

    def apply_pre_tokenization(
        self, tokenizer: TokenizerLike | None, prompt: dict[str, object]
    ) -> dict[str, object]:
        text = prompt.get("prompt")
        if not isinstance(text, str):
            return prompt

        max_input_tokens = self.max_input_tokens
        if max_input_tokens is not None and tokenizer is not None:
            max_input_chars = max_input_tokens * tokenizer.max_chars_per_token
            if self.truncate_prompt_tokens is None and len(text) > max_input_chars:
                raise VLLMValidationError(
                    f"This model's maximum context length is "
                    f"{self.max_total_tokens} tokens. However, you requested "
                    f"{self.max_output_tokens} output tokens and your prompt "
                    f"contains {len(text)} characters (more than "
                    f"{max_input_chars} characters, which is the upper bound "
                    f"for {max_input_tokens} input tokens). Please reduce the "
                    "length of the input prompt or the number of requested "
                    "output tokens.",
                    parameter="input_text",
                    value=len(text),
                )
            if (
                self.truncate_prompt_tokens is not None
                and self.truncation_side is not None
                and len(text) > max_input_chars
            ):
                if max_input_chars == 0:
                    text = ""
                elif self.truncation_side == "left":
                    text = text[-max_input_chars:]
                else:
                    text = text[:max_input_chars]

        if self.do_lower_case:
            text = text.lower()

        if text != prompt["prompt"]:
            prompt = dict(prompt)
            prompt["prompt"] = text
        return prompt

    def _pad_tokens(self, tokenizer: TokenizerLike | None, tokens: Any) -> Any:
        pad_length = self.pad_prompt_tokens
        if pad_length is not None and pad_length < 0:
            pad_length = self.max_input_tokens
        if pad_length is None or pad_length <= len(tokens):
            return tokens
        if tokenizer is None:
            raise ValueError("Cannot pad tokens when `skip_tokenizer_init=True`")
        if not isinstance(tokens, list):
            raise ValueError("Cannot pad tokens for embedding inputs")
        return tokens + [tokenizer.pad_token_id] * (pad_length - len(tokens))

    def _truncate_tokens(self, tokenizer: TokenizerLike | None, tokens: Any) -> Any:
        max_length = self.truncate_prompt_tokens
        if max_length is not None and max_length < 0:
            max_length = self.max_input_tokens
        if max_length is None or max_length >= len(tokens):
            return tokens
        if max_length == 0:
            return tokens[:0]

        side = self.truncation_side or (
            tokenizer.truncation_side if tokenizer is not None else None
        )
        return tokens[-max_length:] if side == "left" else tokens[:max_length]

    def _check_token_length(self, tokens: Any) -> Any:
        max_input_tokens = self.max_input_tokens
        if max_input_tokens is not None and len(tokens) > max_input_tokens:
            token_count = len(tokens)
            qualifier = "at least " if token_count == max_input_tokens + 1 else ""
            total = token_count + self.max_output_tokens
            raise VLLMValidationError(
                f"This model's maximum context length is "
                f"{self.max_total_tokens} tokens. However, you requested "
                f"{self.max_output_tokens} output tokens and your prompt "
                f"contains {qualifier}{token_count} input tokens, for a total "
                f"of {qualifier}{total} tokens. Please reduce the length of "
                "the input prompt or the number of requested output tokens.",
                parameter="input_tokens",
                value=token_count,
            )
        return tokens

    def _validate_tokens(self, tokenizer: TokenizerLike | None, tokens: Any) -> Any:
        tokens = self._pad_tokens(tokenizer, tokens)
        tokens = self._truncate_tokens(tokenizer, tokens)
        return self._check_token_length(tokens)

    def apply_post_tokenization(
        self, tokenizer: TokenizerLike | None, prompt: object
    ) -> object:
        if not isinstance(prompt, dict):
            return prompt
        if "prompt_token_ids" in prompt:
            prompt["prompt_token_ids"] = self._validate_tokens(
                tokenizer, prompt["prompt_token_ids"]
            )
        if "prompt_embeds" in prompt:
            prompt["prompt_embeds"] = self._validate_tokens(
                tokenizer, prompt["prompt_embeds"]
            )
        return prompt
