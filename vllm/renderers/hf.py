from __future__ import annotations

from vllm.tokenizers import TokenizerLike

from .base import BaseRenderer


class HfRenderer(BaseRenderer[TokenizerLike]):
    def _can_produce_offsets(self) -> bool:
        return bool(getattr(self.get_tokenizer(), "is_fast", False))
