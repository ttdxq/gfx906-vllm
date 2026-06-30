from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.utils.import_utils import resolve_obj_by_qualname

from .base import BaseRenderer

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_UNSET_TOKENIZER = object()

_VLLM_RENDERERS = {
    "hf": ("hf", "HfRenderer"),
    "kimi_audio": ("hf", "HfRenderer"),
    "mistral": ("mistral", "MistralRenderer"),
}


@dataclass
class RendererRegistry:
    renderers: dict[str, tuple[str, str]] = field(default_factory=dict)

    def register(self, renderer_mode: str, module: str, class_name: str) -> None:
        if renderer_mode in self.renderers:
            logger.warning(
                "%s.%s is already registered for renderer_mode=%r. It is overwritten by the new one.",
                module,
                class_name,
                renderer_mode,
            )
        self.renderers[renderer_mode] = (module, class_name)

    def load_renderer_cls(self, renderer_mode: str) -> type[BaseRenderer]:
        if renderer_mode not in self.renderers:
            raise ValueError(f"No renderer registered for {renderer_mode=!r}.")
        module, class_name = self.renderers[renderer_mode]
        return resolve_obj_by_qualname(f"vllm.renderers.{module}.{class_name}")

    def load_renderer(self, renderer_mode: str, config: "VllmConfig", tokenizer: TokenizerLike | None) -> BaseRenderer:
        return self.load_renderer_cls(renderer_mode)(config, tokenizer)


RENDERER_REGISTRY = RendererRegistry(dict(_VLLM_RENDERERS))


def renderer_from_config(
    config: "VllmConfig",
    *,
    tokenizer: TokenizerLike | None | object = _UNSET_TOKENIZER,
    **kwargs: object,
) -> BaseRenderer:
    model_config = config.model_config
    resolved_tokenizer: TokenizerLike | None
    if tokenizer is _UNSET_TOKENIZER:
        resolved_tokenizer = cached_tokenizer_from_config(model_config, **kwargs)
    else:
        resolved_tokenizer = tokenizer
    renderer_mode = getattr(model_config, "tokenizer_mode", "auto")
    if renderer_mode != "mistral":
        renderer_mode = "hf"
    return RENDERER_REGISTRY.load_renderer(renderer_mode, config, resolved_tokenizer)
