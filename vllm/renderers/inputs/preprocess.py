from collections.abc import Mapping, Sequence
from typing import NamedTuple, TypeAlias, TypedDict, overload

from vllm.inputs import EmbedsPrompt, EngineInput, ExplicitEncoderDecoderPrompt, PromptType, SingletonPrompt, TextPrompt, TokensPrompt
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.collection_utils import is_list_of


@overload
def prompt_to_seq(prompt_or_prompts: SingletonPrompt | bytes | Sequence[SingletonPrompt | bytes]) -> Sequence[SingletonPrompt]: ...


@overload
def prompt_to_seq(prompt_or_prompts: ExplicitEncoderDecoderPrompt | Sequence[ExplicitEncoderDecoderPrompt]) -> Sequence[ExplicitEncoderDecoderPrompt]: ...


@overload
def prompt_to_seq(prompt_or_prompts: PromptType | Sequence[PromptType]) -> Sequence[PromptType]: ...


def prompt_to_seq(prompt_or_prompts: PromptType | bytes | Sequence[PromptType | bytes]) -> Sequence[PromptType]:
    if isinstance(prompt_or_prompts, (dict, str, bytes)) or (len(prompt_or_prompts) > 0 and is_list_of(prompt_or_prompts, int)):
        return [prompt_or_prompts]
    return prompt_or_prompts


def conversation_to_seq(conversation_or_conversations: list[dict[str, object]] | Sequence[list[dict[str, object]]]) -> Sequence[list[dict[str, object]]]:
    if len(conversation_or_conversations) > 0 and is_list_of(conversation_or_conversations, dict):
        return [conversation_or_conversations]
    return conversation_or_conversations


DecoderOnlyDictPrompt: TypeAlias = TextPrompt | TokensPrompt | EmbedsPrompt
EncoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt
DecoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt


class EncoderDecoderDictPrompt(TypedDict):
    encoder_prompt: EncoderDictPrompt
    decoder_prompt: DecoderDictPrompt | None


SingletonDictPrompt: TypeAlias = DecoderOnlyDictPrompt | EncoderDictPrompt | DecoderDictPrompt
DictPrompt: TypeAlias = DecoderOnlyDictPrompt | EncoderDecoderDictPrompt


def _validate_prompt_dict(prompt: Mapping[str, object]) -> None:
    if "prompt" not in prompt or "prompt_token_ids" in prompt or "prompt_embeds" in prompt:
        return
    if not isinstance(prompt["prompt"], str):
        raise TypeError("Prompt text should be a string")


def parse_dec_only_prompt(prompt: PromptType | object) -> DecoderOnlyDictPrompt:
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)
    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")
        return TokensPrompt(prompt_token_ids=prompt)
    if isinstance(prompt, dict):
        if "encoder_prompt" in prompt:
            raise TypeError("Cannot pass encoder-decoder prompt to decoder-only models")
        _validate_prompt_dict(prompt)
        if "prompt" in prompt or "prompt_token_ids" in prompt or "prompt_embeds" in prompt:
            return prompt
        raise TypeError("Prompt dictionary must contain text, tokens, or embeddings")
    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def _parse_enc_prompt(prompt: PromptType | object) -> EncoderDictPrompt:
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)
    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")
        return TokensPrompt(prompt_token_ids=prompt)
    if isinstance(prompt, dict):
        _validate_prompt_dict(prompt)
        if "prompt_embeds" in prompt:
            raise TypeError("Cannot pass embeddings prompt to encoder-decoder models")
        if "prompt" in prompt or "prompt_token_ids" in prompt:
            return prompt
        raise TypeError("Prompt dictionary must contain text or tokens")
    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def _parse_dec_prompt(prompt: PromptType | object) -> DecoderDictPrompt:
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)
    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")
        return TokensPrompt(prompt_token_ids=prompt)
    if isinstance(prompt, dict):
        _validate_prompt_dict(prompt)
        if "prompt_embeds" in prompt:
            raise TypeError("Cannot pass embeddings prompt to encoder-decoder models")
        if "multi_modal_data" in prompt or "mm_processor_kwargs" in prompt or "multi_modal_uuids" in prompt:
            raise TypeError("Cannot pass multi-modal inputs to decoder prompt")
        if "prompt" in prompt or "prompt_token_ids" in prompt:
            return prompt
        raise TypeError("Prompt dictionary must contain text or tokens")
    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


def parse_enc_dec_prompt(prompt: PromptType | object) -> EncoderDecoderDictPrompt:
    if isinstance(prompt, dict) and "encoder_prompt" in prompt:
        enc_prompt = prompt["encoder_prompt"]
        dec_prompt = prompt["decoder_prompt"]
    else:
        enc_prompt = prompt
        dec_prompt = None
    return EncoderDecoderDictPrompt(encoder_prompt=_parse_enc_prompt(enc_prompt), decoder_prompt=None if dec_prompt is None else _parse_dec_prompt(dec_prompt))


def parse_model_prompt(model_config: object, prompt: object):
    if getattr(model_config, "is_encoder_decoder", False):
        return parse_enc_dec_prompt(prompt)
    return parse_dec_only_prompt(prompt)


class PromptComponents(NamedTuple):
    text: str | None = None
    token_ids: list[int] | None = None
    embeds: object | None = None


def extract_target_prompt(model_config: object, prompt: object):
    return parse_enc_dec_prompt(prompt)["encoder_prompt"] if getattr(model_config, "is_encoder_decoder", False) else parse_dec_only_prompt(prompt)


def extract_prompt_components(model_config: object, prompt: PromptType | EngineInput) -> PromptComponents:
    target_prompt = extract_target_prompt(model_config, prompt)
    return PromptComponents(text=target_prompt.get("prompt"), token_ids=target_prompt.get("prompt_token_ids"), embeds=target_prompt.get("prompt_embeds"))


def extract_prompt_len(model_config: object, prompt: PromptType | EngineInput):
    target_prompt = extract_target_prompt(model_config, prompt)
    return length_from_prompt_token_ids_or_embeds(target_prompt.get("prompt_token_ids"), target_prompt.get("prompt_embeds"))
