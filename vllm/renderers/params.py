from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

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
    return base | {key: value for key, value in overrides.items() if value not in unset_values}


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
            merged[key] = recursively_merge_kwargs(merged[key], value, unset_values=unset_values)
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
        if not default_chat_template_kwargs and not default_media_io_kwargs and not default_mm_processor_kwargs:
            return self
        return ChatParams(
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
            chat_template_kwargs=merge_kwargs(default_chat_template_kwargs, self.chat_template_kwargs),
            media_io_kwargs=merge_kwargs(default_media_io_kwargs, self.media_io_kwargs),
            mm_processor_kwargs=recursively_merge_kwargs(default_mm_processor_kwargs, self.mm_processor_kwargs),
        )

    def get_apply_chat_template_kwargs(self) -> dict[str, object]:
        return merge_kwargs(self.chat_template_kwargs, {"chat_template": self.chat_template, "return_dict": False})


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
            raise VLLMValidationError("`truncation_side` must be either 'left' or 'right'.", parameter="truncation_side", value=self.truncation_side)
        if self.max_total_tokens is not None and self.max_output_tokens > self.max_total_tokens:
            raise VLLMValidationError(
                f"{self.max_output_tokens_param}={self.max_output_tokens} cannot be greater than {self.max_total_tokens_param}={self.max_total_tokens}. Please request fewer output tokens.",
                parameter=self.max_output_tokens_param,
                value=self.max_output_tokens,
            )
        if self.max_input_tokens is not None and self.truncate_prompt_tokens is not None and self.truncate_prompt_tokens > self.max_input_tokens:
            raise VLLMValidationError(
                f"{self.truncate_prompt_tokens_param}={self.truncate_prompt_tokens} cannot be greater than {self.max_total_tokens_param} - {self.max_output_tokens_param} = {self.max_input_tokens}. Please request a smaller truncation size.",
                parameter=self.truncate_prompt_tokens_param,
                value=self.truncate_prompt_tokens,
            )

    def with_kwargs(self, **tokenization_kwargs: object) -> TokenizeParams:
        max_length = tokenization_kwargs.pop("max_length", self.max_input_tokens)
        return replace(
            self,
            max_total_tokens=None if max_length is None else max_length + self.max_output_tokens,
            pad_prompt_tokens=tokenization_kwargs.pop("pad_prompt_tokens", self.pad_prompt_tokens),
            truncate_prompt_tokens=tokenization_kwargs.pop("truncate_prompt_tokens", self.truncate_prompt_tokens),
            truncation_side=tokenization_kwargs.pop("truncation_side", self.truncation_side),
            do_lower_case=bool(tokenization_kwargs.pop("do_lower_case", self.do_lower_case)),
            add_special_tokens=bool(tokenization_kwargs.pop("add_special_tokens", self.add_special_tokens)),
            return_token_offsets=bool(tokenization_kwargs.pop("return_token_offsets", self.return_token_offsets)),
            needs_detokenization=bool(tokenization_kwargs.pop("needs_detokenization", self.needs_detokenization)),
        )

    def get_encode_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"add_special_tokens": self.add_special_tokens}
        if self.max_input_tokens is not None:
            kwargs["max_length"] = self.max_input_tokens
            kwargs["truncation"] = True
        if self.truncation_side is not None:
            kwargs["truncation_side"] = self.truncation_side
        return kwargs

    def apply_pre_tokenization(self, tokenizer: "TokenizerLike | None", prompt: dict[str, object]) -> dict[str, object]:
        if self.do_lower_case and isinstance(prompt.get("prompt"), str):
            prompt = dict(prompt)
            prompt["prompt"] = prompt["prompt"].lower()
        return prompt

    def apply_post_tokenization(self, tokenizer: "TokenizerLike | None", prompt: object) -> object:
        return prompt
