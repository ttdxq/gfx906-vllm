from __future__ import annotations

from vllm.tokenizers import MistralTokenizer

from .base import BaseRenderer


class MistralRenderer(BaseRenderer[MistralTokenizer]):
    pass
