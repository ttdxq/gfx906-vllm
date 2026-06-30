# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast, overload

from typing_extensions import assert_never

from vllm.config import VllmConfig
from vllm.inputs import build_enc_dec_input
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import is_list_of

if TYPE_CHECKING:
    from vllm.v1.metrics.stats import MultiModalCacheStats

from .engine import (
    DecoderEngineInput,
    DecoderOnlyEngineInput,
    EmbedsInput,
    EncoderDecoderInput,
    EncoderInput,
    EngineInput,
    MultiModalInput,
    SingletonInput,
    TokensInput,
    tokens_input,
)
from .llm import (
    DecoderOnlyPrompt,
    DecoderPrompt,
    EmbedsPrompt,
    ExplicitEncoderDecoderPrompt,
    EncoderPrompt,
    MultiModalDataDict,
    MultiModalUUIDDict,
    PromptType,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
)

logger = init_logger(__name__)


def _validate_prompt_dict(prompt: Mapping[str, object]) -> None:
    if (
        "prompt" not in prompt
        or "prompt_token_ids" in prompt
        or "prompt_embeds" in prompt
    ):
        return
    if not isinstance(prompt["prompt"], str):
        raise TypeError("Prompt text should be a string")


def _parse_enc_prompt(prompt: PromptType | object) -> EncoderPrompt:
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)
    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")
        return TokensPrompt(prompt_token_ids=prompt)
    if isinstance(prompt, dict):
        _validate_prompt_dict(prompt)
        if "prompt_embeds" in prompt:
            raise TypeError(
                "Cannot pass embeddings prompt to encoder-decoder models"
            )
        if "prompt" in prompt or "prompt_token_ids" in prompt:
            return cast(EncoderPrompt, prompt)
        raise TypeError("Prompt dictionary must contain text or tokens")
    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def _parse_dec_prompt(prompt: PromptType | object) -> DecoderPrompt:
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)
    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")
        return TokensPrompt(prompt_token_ids=prompt)
    if isinstance(prompt, dict):
        _validate_prompt_dict(prompt)
        if "prompt_embeds" in prompt:
            raise TypeError(
                "Cannot pass embeddings prompt to encoder-decoder models"
            )
        if (
            "multi_modal_data" in prompt
            or "mm_processor_kwargs" in prompt
            or "multi_modal_uuids" in prompt
        ):
            raise TypeError("Cannot pass multi-modal inputs to decoder prompt")
        if "prompt" in prompt or "prompt_token_ids" in prompt:
            return cast(DecoderPrompt, prompt)
        raise TypeError("Prompt dictionary must contain text or tokens")
    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def parse_dec_only_prompt(prompt: PromptType | object) -> DecoderOnlyPrompt:
    if isinstance(prompt, dict) and "encoder_prompt" in prompt:
        raise TypeError("Cannot pass encoder-decoder prompt to decoder-only models")

    if isinstance(prompt, dict) and "prompt_embeds" in prompt:
        return cast(DecoderOnlyPrompt, prompt)

    return cast(DecoderOnlyPrompt, _parse_dec_prompt(prompt))


def parse_enc_dec_prompt(prompt: PromptType | object) -> ExplicitEncoderDecoderPrompt:
    if isinstance(prompt, dict) and "encoder_prompt" in prompt:
        enc_prompt = prompt["encoder_prompt"]
        dec_prompt = prompt["decoder_prompt"]
    else:
        enc_prompt = prompt
        dec_prompt = None

    return ExplicitEncoderDecoderPrompt(
        encoder_prompt=_parse_enc_prompt(enc_prompt),
        decoder_prompt=None if dec_prompt is None else _parse_dec_prompt(dec_prompt),
    )


class InputPreprocessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        super().__init__()

        self.model_config = vllm_config.model_config
        self.renderer = renderer or renderer_from_config(vllm_config)
        self.mm_registry = mm_registry

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def stat_mm_cache(self) -> "MultiModalCacheStats | None":
        return self.renderer.stat_mm_cache()

    def clear_mm_cache(self) -> None:
        self.renderer.clear_mm_cache()

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(
            TextPrompt(prompt=prompt),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    def _process_multimodal(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInput:
        """
        Apply the model's multi-modal processor to a multi-modal prompt,
        returning the corresponding token IDs and metadata.
        """
        return self.renderer._process_multimodal(
            prompt,
            mm_data,
            mm_uuids=mm_uuids,
            mm_processor_kwargs=mm_processor_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

    def _process_embeds(
        self,
        parsed_content: EmbedsPrompt,
    ) -> EmbedsInput:
        return self.renderer._process_embeds(parsed_content)

    def _truncate_inputs(
        self, inputs: list[int], tokenization_kwargs: dict[str, Any] | None = None
    ) -> list[int]:
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(
            TokensPrompt(prompt_token_ids=inputs),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    def _process_tokens(
        self,
        parsed_content: TokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> TokensInput | MultiModalInput:
        prompt_token_ids = self._truncate_inputs(
            parsed_content["prompt_token_ids"], tokenization_kwargs
        )

        inputs: TokensInput | MultiModalInput
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs"),
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=(
                    mm_uuids
                    if mm_uuids is not None
                    else parsed_content.get("multi_modal_uuids")
                ),
            )
        else:
            inputs = tokens_input(prompt_token_ids)

        if prompt_text := parsed_content.get("prompt"):
            inputs["prompt"] = prompt_text
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_text(
        self,
        parsed_content: TextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> TokensInput | MultiModalInput:
        prompt_text = parsed_content["prompt"]

        inputs: TokensInput | MultiModalInput
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=(
                    mm_uuids
                    if mm_uuids is not None
                    else parsed_content.get("multi_modal_uuids")
                ),
            )
        else:
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            inputs = tokens_input(prompt_token_ids)

        inputs["prompt"] = prompt_text

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    @overload
    def _prompt_to_llm_inputs(
        self,
        prompt: EncoderPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> EncoderInput: ...

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: DecoderPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> DecoderEngineInput: ...

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: DecoderOnlyPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> DecoderOnlyEngineInput: ...

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> SingletonInput:
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        if "prompt_token_ids" in prompt:
            return self._process_tokens(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )  # type: ignore[arg-type]

        if "prompt" in prompt:
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        assert_never(prompt)  # type: ignore[arg-type]

    def _process_encoder_decoder_prompt(
        self,
        prompt: ExplicitEncoderDecoderPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> EncoderDecoderInput:
        encoder_prompt = prompt["encoder_prompt"]
        decoder_prompt = prompt["decoder_prompt"]

        skip_decoder_start_token = False
        if self.renderer.mm_processor is not None:
            from vllm.multimodal.processing import EncDecMultiModalProcessor

            if isinstance(self.renderer.mm_processor, EncDecMultiModalProcessor):
                skip_decoder_start_token = (
                    self.renderer.mm_processor.skip_decoder_start_token
                )

        return build_enc_dec_input(
            encoder_input=self._prompt_to_llm_inputs(
                encoder_prompt,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            ),
            decoder_input=(
                None
                if decoder_prompt is None
                else self._prompt_to_llm_inputs(
                    decoder_prompt,
                    tokenization_kwargs=tokenization_kwargs,
                    mm_uuids=mm_uuids,
                )
            ),
            decoder_start_token_id=self.renderer.get_dec_start_token_id(),
            skip_decoder_start_token=skip_decoder_start_token,
        )

    def _process_decoder_only_prompt(
        self,
        prompt: DecoderOnlyPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> DecoderOnlyEngineInput:
        return self._prompt_to_llm_inputs(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> EngineInput:
        """Preprocess the input prompt."""
        if self.model_config.is_encoder_decoder:
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder.
            return self._process_encoder_decoder_prompt(
                parse_enc_dec_prompt(prompt),
                tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        return self._process_decoder_only_prompt(
            parse_dec_only_prompt(prompt),
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
