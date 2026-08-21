from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

from vllm.inputs import (
    EmbedsInput,
    EmbedsPrompt,
    EncoderDecoderInput,
    EngineInput,
    SingletonInput,
    TextPrompt,
    TokensInput,
    TokensPrompt,
    build_enc_dec_input,
    embeds_input,
    tokens_input,
)
from vllm.logger import init_logger

from .embed_utils import safe_load_prompt_embeds, safe_load_prompt_embeds_async
from .inputs.preprocess import (
    EncoderDecoderDictPrompt,
    parse_dec_only_prompt,
    parse_enc_dec_prompt,
)
from .inputs.tokenize import EncoderDecoderTokPrompt, SingletonTokPrompt, TokPrompt
from .params import ChatParams, TokenizeParams

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig
    from vllm.tokenizers import TokenizerLike
    from vllm.v1.metrics.stats import MultiModalCacheStats

logger = init_logger(__name__)

_T = TypeVar("_T")


class BaseRenderer(ABC, Generic[_T]):
    def __init__(self, config: "VllmConfig", tokenizer: _T | None) -> None:
        self.config = config
        self.model_config = config.model_config
        self.tokenizer = tokenizer
        self.mm_processor = None

    def get_tokenizer(self) -> _T:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")
        return tokenizer

    def stat_mm_cache(self) -> "MultiModalCacheStats | None":
        return None

    def clear_mm_cache(self) -> None:
        return None

    def get_dec_start_token_id(self) -> int:
        dec_start_token_id = getattr(
            self.model_config.hf_config, "decoder_start_token_id", None
        )
        if dec_start_token_id is None:
            dec_start_token_id = getattr(self.get_tokenizer(), "bos_token_id", None)
        if dec_start_token_id is None:
            raise RuntimeError("Cannot find decoder start token id or <BOS>")
        return dec_start_token_id

    def get_eos_token_id(self) -> int | None:
        hf_config = getattr(self.model_config, "hf_config", None)
        if hf_config is not None:
            eos_token_id = getattr(hf_config, "eos_token_id", None)
            if eos_token_id is not None:
                return eos_token_id

        tokenizer = self.tokenizer
        if tokenizer is None:
            return None

        return getattr(tokenizer, "eos_token_id", None)

    @cached_property
    def default_cmpl_tok_params(self) -> TokenizeParams:
        encoder_config = getattr(self.model_config, "encoder_config", None) or {}
        return TokenizeParams(
            max_total_tokens=self.model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=True,
        )

    @cached_property
    def default_chat_tok_params(self) -> TokenizeParams:
        encoder_config = getattr(self.model_config, "encoder_config", None) or {}
        return TokenizeParams(
            max_total_tokens=self.model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=False,
        )

    def render_prompt(self, prompt: dict[str, object] | bytes) -> dict[str, object]:
        if isinstance(prompt, bytes):
            embeds = safe_load_prompt_embeds(self.model_config, prompt)
            return EmbedsPrompt(prompt_embeds=embeds)
        return prompt

    def render_prompts(
        self, prompts: Sequence[dict[str, object] | bytes]
    ) -> list[dict[str, object]]:
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")
        return [self.render_prompt(prompt) for prompt in prompts]

    async def _render_prompt_async(
        self, prompt: dict[str, object] | bytes
    ) -> dict[str, object]:
        if isinstance(prompt, bytes):
            embeds = await safe_load_prompt_embeds_async(self.model_config, prompt)
            return EmbedsPrompt(prompt_embeds=embeds)
        return prompt

    async def render_prompts_async(
        self, prompts: Sequence[dict[str, object] | bytes]
    ) -> list[dict[str, object]]:
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")
        return [await self._render_prompt_async(prompt) for prompt in prompts]

    def render_messages(
        self, messages: list[dict[str, object]], params: ChatParams
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        rendered = self.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            **params.get_apply_chat_template_kwargs(),
        )
        return list(messages), TextPrompt(prompt=rendered)

    async def render_messages_async(
        self, messages: list[dict[str, object]], params: ChatParams
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        return self.render_messages(messages, params)

    def _can_produce_offsets(self) -> bool:
        return False

    @staticmethod
    def _build_tokens_prompt(
        token_ids: list[int] | tuple[int, ...],
        prompt: dict[str, object],
        *,
        offset_mapping: list[tuple[int, int]] | None = None,
    ) -> dict[str, object]:
        if offset_mapping is None:
            return TokensPrompt(prompt_token_ids=list(token_ids), **prompt)
        return TokensPrompt(
            prompt_token_ids=list(token_ids),
            prompt_token_offsets=[
                (int(start), int(end)) for start, end in offset_mapping
            ],
            **prompt,
        )

    def _tokenize_prompt(
        self, prompt: dict[str, object], params: TokenizeParams
    ) -> dict[str, object]:
        tokenizer = self.get_tokenizer()
        kwargs = params.get_encode_kwargs()
        want_offsets = (
            params.return_token_offsets
            and self._can_produce_offsets()
            and not prompt.get("multi_modal_data")
            and not prompt.get("multi_modal_uuids")
        )
        if want_offsets:
            kwargs = {**kwargs, "return_offsets_mapping": True}
        encoding = tokenizer(prompt["prompt"], **kwargs)
        offset_mapping = encoding["offset_mapping"] if want_offsets else None
        return self._build_tokens_prompt(
            encoding["input_ids"], prompt, offset_mapping=offset_mapping
        )

    def _detokenize_prompt(self, prompt: dict[str, object]) -> dict[str, object]:
        prompt["prompt"] = self.get_tokenizer().decode(prompt["prompt_token_ids"])
        return prompt

    async def _detokenize_prompt_async(
        self, prompt: dict[str, object]
    ) -> dict[str, object]:
        prompt["prompt"] = self.get_tokenizer().decode(prompt["prompt_token_ids"])
        return prompt

    def _tokenize_singleton_prompt(
        self, prompt: dict[str, object], params: TokenizeParams
    ) -> dict[str, object]:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            if not isinstance(prompt.get("prompt"), str):
                raise TypeError(
                    "Expected prompt['prompt'] to be a string before tokenization; use 'prompt_token_ids' for token ID inputs"
                )
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)
            prompt = self._tokenize_prompt(prompt, params)
        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")
            prompt = self._detokenize_prompt(prompt)
        return params.apply_post_tokenization(self.tokenizer, prompt)

    async def _tokenize_singleton_prompt_async(
        self, prompt: dict[str, object], params: TokenizeParams
    ) -> dict[str, object]:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            if not isinstance(prompt.get("prompt"), str):
                raise TypeError(
                    "Expected prompt['prompt'] to be a string before tokenization; use 'prompt_token_ids' for token ID inputs"
                )
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)
            prompt = self._tokenize_prompt(prompt, params)
        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")
            prompt = await self._detokenize_prompt_async(prompt)
        return params.apply_post_tokenization(self.tokenizer, prompt)

    def _tokenize_enc_dec_prompt(
        self, prompt: EncoderDecoderDictPrompt, params: TokenizeParams
    ) -> EncoderDecoderTokPrompt:
        return {
            "encoder_prompt": self._tokenize_singleton_prompt(
                prompt["encoder_prompt"], params
            ),
            "decoder_prompt": None
            if prompt["decoder_prompt"] is None
            else self._tokenize_singleton_prompt(prompt["decoder_prompt"], params),
        }

    async def _tokenize_enc_dec_prompt_async(
        self, prompt: EncoderDecoderDictPrompt, params: TokenizeParams
    ) -> EncoderDecoderTokPrompt:
        return {
            "encoder_prompt": await self._tokenize_singleton_prompt_async(
                prompt["encoder_prompt"], params
            ),
            "decoder_prompt": None
            if prompt["decoder_prompt"] is None
            else await self._tokenize_singleton_prompt_async(
                prompt["decoder_prompt"], params
            ),
        }

    def tokenize_prompt(
        self, prompt: dict[str, object], params: TokenizeParams
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            return self._tokenize_enc_dec_prompt(prompt, params)
        return self._tokenize_singleton_prompt(prompt, params)

    def tokenize_prompts(
        self, prompts: Sequence[dict[str, object]], params: TokenizeParams
    ) -> list[TokPrompt]:
        return [self.tokenize_prompt(prompt, params) for prompt in prompts]

    async def tokenize_prompt_async(
        self, prompt: dict[str, object], params: TokenizeParams
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            return await self._tokenize_enc_dec_prompt_async(prompt, params)
        return await self._tokenize_singleton_prompt_async(prompt, params)

    async def tokenize_prompts_async(
        self, prompts: Sequence[dict[str, object]], params: TokenizeParams
    ) -> list[TokPrompt]:
        return [await self.tokenize_prompt_async(prompt, params) for prompt in prompts]

    def _process_embeds(self, prompt: EmbedsPrompt) -> EmbedsInput:
        if not self.model_config.enable_prompt_embeds:
            raise ValueError(
                "You must set `--enable-prompt-embeds` to input `prompt_embeds`."
            )

        prompt_embeds = prompt["prompt_embeds"]

        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.squeeze(dim=0)

        if prompt_embeds.ndim != 2:
            raise ValueError("prompt_embeds must be of shape (seq_len, hidden_size).")

        return embeds_input(
            prompt_embeds=prompt_embeds.cpu(),
            prompt=prompt.get("prompt"),
            cache_salt=prompt.get("cache_salt"),
            prompt_token_ids=prompt.get("prompt_token_ids"),
            is_token_ids=prompt.get("prompt_is_token_ids"),
        )

    def _process_multimodal(
        self,
        prompt: list[int] | str,
        mm_data: dict[str, object],
        mm_uuids: dict[str, object] | None,
        mm_processor_kwargs: dict[str, object] | None,
        tokenization_kwargs: dict[str, object] | None,
        *,
        skip_mm_cache: bool = False,
    ) -> EngineInput:
        if self.mm_processor is None:
            raise ValueError(f"{self.model_config.model} is not a multimodal model")
        return self.mm_processor.apply(
            prompt,
            mm_data,
            mm_processor_kwargs or {},
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def _process_tokens(
        self, prompt: TokensPrompt, *, skip_mm_cache: bool = False
    ) -> TokensInput:
        return tokens_input(
            prompt["prompt_token_ids"],
            prompt=prompt.get("prompt"),
            cache_salt=prompt.get("cache_salt"),
        )

    def _process_enc_dec(
        self, prompt: EncoderDecoderTokPrompt, *, skip_mm_cache: bool = False
    ) -> EncoderDecoderInput:
        return build_enc_dec_input(
            self._process_tokens(prompt["encoder_prompt"], skip_mm_cache=skip_mm_cache),
            None
            if prompt["decoder_prompt"] is None
            else self._process_tokens(
                prompt["decoder_prompt"], skip_mm_cache=skip_mm_cache
            ),
            self.get_dec_start_token_id(),
        )
