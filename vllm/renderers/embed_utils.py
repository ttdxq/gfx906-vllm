from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pybase64
import torch

from vllm.exceptions import VLLMValidationError
from vllm.utils.async_utils import make_async

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def safe_load_prompt_embeds(model_config: "ModelConfig", embed: bytes) -> torch.Tensor:
    if not model_config.enable_prompt_embeds:
        raise VLLMValidationError("You must set `--enable-prompt-embeds` to input `prompt_embeds`.", parameter="prompt_embeds")
    with torch.sparse.check_sparse_tensor_invariants():
        tensor = torch.load(BytesIO(pybase64.b64decode(embed, validate=True)), weights_only=True, map_location=torch.device("cpu"))
        if not isinstance(tensor, torch.Tensor):
            raise VLLMValidationError("`prompt_embeds` payload did not deserialize to a torch.Tensor.", parameter="prompt_embeds")
        tensor = tensor.to_dense()
    if tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise VLLMValidationError(f"`prompt_embeds` must be a 2D tensor of shape (num_tokens, hidden_size); got shape {tuple(tensor.shape)}.", parameter="prompt_embeds")
    expected_hidden_size = model_config.get_hidden_size()
    if tensor.shape[1] != expected_hidden_size:
        raise VLLMValidationError(f"`prompt_embeds` hidden_size {tensor.shape[1]} does not match the model's hidden_size {expected_hidden_size}.", parameter="prompt_embeds")
    expected_dtype = model_config.dtype
    if tensor.dtype != expected_dtype:
        if not tensor.is_floating_point():
            raise VLLMValidationError(f"`prompt_embeds` dtype {tensor.dtype} is not a floating-point type, cannot safely cast to the model's dtype {expected_dtype}.", parameter="prompt_embeds")
        tensor = tensor.to(expected_dtype)
    return tensor


safe_load_prompt_embeds_async = make_async(safe_load_prompt_embeds)
