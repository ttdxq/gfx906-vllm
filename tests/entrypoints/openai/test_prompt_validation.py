# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from types import SimpleNamespace
from unittest.mock import Mock

# imports for structured outputs tests
import openai
import pybase64
import pytest
import regex as re
import torch

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.entrypoints.openai.protocol import CompletionRequest
from vllm.entrypoints.renderer import CompletionRenderer
from vllm.exceptions import VLLMValidationError
from vllm.renderers import TokenizeParams
from vllm.renderers.hf import HfRenderer

from ...utils import RemoteOpenAIServer


@pytest.mark.asyncio
async def test_empty_prompt():
    model_name = "gpt2"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(
            openai.BadRequestError,
            match="Either prompt or prompt_embeds must be provided and non-empty.",
        ):
            await client.completions.create(
                model=model_name,
                prompt="",
                max_tokens=5,
                temperature=0.0,
                extra_body={"prompt_embeds": []},
            )


@pytest.mark.asyncio
async def test_out_of_vocab_token_ids():
    model_name = "gpt2"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        with pytest.raises(
            openai.BadRequestError, match=re.compile(".*out of vocabulary.*").pattern
        ):
            await client.completions.create(
                model=model_name, prompt=[999999], max_tokens=5, temperature=0.0
            )


@pytest.mark.parametrize(
    "prompt",
    [
        ["a", "b", "c", "d"],
        [[1], [2], [3], [4]],
    ],
)
def test_completion_prompt_list_limit(monkeypatch: pytest.MonkeyPatch, prompt: list):
    monkeypatch.setattr(envs, "VLLM_MAX_COMPLETION_PROMPTS", 3)

    with pytest.raises(ValueError, match="prompt list length 4 exceeds"):
        CompletionRequest(model="test", prompt=prompt, max_tokens=1)


def test_completion_prompt_list_exact_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_MAX_COMPLETION_PROMPTS", 3)

    request = CompletionRequest(model="test", prompt=["a", "b", "c"], max_tokens=1)
    assert request.prompt == ["a", "b", "c"]


def test_completion_flat_token_prompt_is_single_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_MAX_COMPLETION_PROMPTS", 1)

    request = CompletionRequest(model="test", prompt=[1, 2, 3], max_tokens=1)
    assert request.prompt == [1, 2, 3]


def test_completion_prompt_embeds_list_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_MAX_COMPLETION_PROMPTS", 2)

    with pytest.raises(ValueError, match="prompt_embeds list length 3 exceeds"):
        CompletionRequest(
            model="test",
            prompt_embeds=[b"a", b"b", b"c"],
            max_tokens=1,
        )


class _BoundedTokenizer:
    max_chars_per_token = 1
    truncation_side = "left"
    pad_token_id = 0

    def __init__(self):
        self.texts: list[str] = []
        self.kwargs: dict[str, object] = {}

    def __call__(self, text: str, **kwargs):
        self.texts.append(text)
        self.kwargs = kwargs
        return {"input_ids": [ord(char) for char in text]}


def _bounded_renderer(tokenizer: _BoundedTokenizer) -> HfRenderer:
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))
    return HfRenderer(config, tokenizer)


def test_tokenizer_rejects_unbounded_prompt_before_tokenization():
    tokenizer = _BoundedTokenizer()
    renderer = _bounded_renderer(tokenizer)

    with pytest.raises(VLLMValidationError, match="maximum context length"):
        renderer.tokenize_prompts(
            [{"prompt": "x" * 101}], TokenizeParams(max_total_tokens=100)
        )

    assert tokenizer.texts == []


def test_explicit_truncation_bounds_tokenizer_input():
    tokenizer = _BoundedTokenizer()
    renderer = _bounded_renderer(tokenizer)

    result = renderer.tokenize_prompts(
        [{"prompt": "x" * 500}],
        TokenizeParams(
            max_total_tokens=100,
            truncate_prompt_tokens=4,
            truncation_side="left",
        ),
    )[0]

    assert len(tokenizer.texts[0]) == 100
    assert tokenizer.kwargs["truncation"] is False
    assert len(result["prompt_token_ids"]) == 4


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        ("left", [ord(char) for char in "6789"]),
        ("right", [ord(char) for char in "0123"]),
    ],
)
def test_explicit_truncation_side_is_applied_after_tokenization(side, expected):
    tokenizer = _BoundedTokenizer()
    renderer = _bounded_renderer(tokenizer)

    result = renderer.tokenize_prompts(
        [{"prompt": "0123456789"}],
        TokenizeParams(
            max_total_tokens=100,
            truncate_prompt_tokens=4,
            truncation_side=side,
        ),
    )[0]

    assert result["prompt_token_ids"] == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "layout", [torch.strided, torch.sparse_coo, torch.sparse_csc, torch.sparse_csr]
)
@pytest.mark.parametrize("seq_len", [2, 10])
@pytest.mark.parametrize("hidden_size", [2, 10])
def test_load_prompt_embeds(
    dtype: torch.dtype, layout: torch.layout, seq_len: int, hidden_size: int
):
    model_config = Mock(spec=ModelConfig)
    model_config.enable_prompt_embeds = True
    renderer = CompletionRenderer(model_config, tokenizer=None)

    # construct arbitrary tensors of various dtypes, layouts, and sizes.
    # We need to check against different layouts to make sure that if a user
    # uses sparse tensors to reduce the transmission size of prompt embeddings,
    # we must cast them to dense/strided before passing them into the engine.
    # We don't use non-CPU tensors in this test to avoid preemptively
    # initializing cuda and break other tests in the suite that fork processes.
    # We also need to make sure that we only use devices that are actually
    # available in the environment the test is running on. For simplicity,
    # we just test against CPU.
    tensor = torch.randn((seq_len, hidden_size), dtype=dtype)
    if layout == torch.strided:
        tensor = tensor.contiguous()
    elif layout == torch.sparse_coo:
        tensor = tensor.to_sparse_coo()
    elif layout == torch.sparse_csc:
        tensor = tensor.to_sparse_csc()
    elif layout == torch.sparse_csr:
        tensor = tensor.to_sparse_csr()

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    encoded_tensor = pybase64.b64encode(buffer.getvalue())

    loaded_prompt_embeds = renderer.load_prompt_embeds(encoded_tensor)
    assert len(loaded_prompt_embeds) == 1
    loaded_tensor = loaded_prompt_embeds[0]["prompt_embeds"]
    assert loaded_tensor.device.type == "cpu"
    assert loaded_tensor.layout == torch.strided
    torch.testing.assert_close(
        loaded_tensor, tensor.to("cpu").to_dense(), equal_nan=True
    )


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("seq_len", [2])
@pytest.mark.parametrize("hidden_size", [2])
def test_disable_prompt_embeds(dtype: torch.dtype, seq_len: int, hidden_size: int):
    model_config = Mock(spec=ModelConfig)
    model_config.enable_prompt_embeds = False
    renderer = CompletionRenderer(model_config, tokenizer=None)

    tensor = torch.randn((seq_len, hidden_size), dtype=dtype)

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    encoded_tensor = pybase64.b64encode(buffer.getvalue())

    with pytest.raises(ValueError, match="--enable-prompt-embeds"):
        renderer.load_prompt_embeds(encoded_tensor)
