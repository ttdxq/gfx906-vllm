# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.tokenizers import MistralTokenizer


@pytest.fixture()
def serving() -> OpenAIServing:
    """Create a minimal OpenAIServing instance for testing."""

    # Create minimal mocks
    engine_client = Mock()
    model_config = Mock(spec=ModelConfig)
    model_config.max_model_len = 32768
    models = Mock(spec=OpenAIServingModels)
    models.model_config = model_config
    models.input_processor = Mock()
    models.io_processor = Mock()

    serving = OpenAIServing(
        engine_client=engine_client,
        models=models,
        request_logger=None,
    )
    return serving


@pytest.mark.asyncio
async def test_async_mistral_tokenizer_does_not_block_event_loop(
    serving: OpenAIServing,
):
    expected_tokens = [1, 2, 3]

    # Mock the blocking version to sleep
    def mocked_apply_chat_template(*_args, **_kwargs):
        time.sleep(2)
        return expected_tokens

    mock_tokenizer = Mock(spec=MistralTokenizer)
    mock_tokenizer.apply_chat_template.side_effect = mocked_apply_chat_template

    task = serving._apply_mistral_chat_template_async(
        tokenizer=mock_tokenizer, messages=[], chat_template=None, tools=[]
    )

    # Ensure the event loop is not blocked
    blocked_count = 0
    for _i in range(20):  # Check over ~2 seconds
        start = time.perf_counter()
        await asyncio.sleep(0)
        elapsed = time.perf_counter() - start

        # an overly generous elapsed time for slow machines
        if elapsed >= 0.5:
            blocked_count += 1

        await asyncio.sleep(0.1)

    # Ensure task completes
    tokens = await task
    assert tokens == expected_tokens, "Mocked blocking tokenizer was not called"
    assert blocked_count == 0, "Event loop blocked during tokenization"


def test_reasoning_effort_forwarded_to_chat_template_kwargs(
    serving: OpenAIServing,
):
    request = SimpleNamespace(reasoning_effort="low")
    assert serving._get_effective_chat_template_kwargs(None, request) == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }

    request = SimpleNamespace(reasoning_effort="none")
    assert serving._get_effective_chat_template_kwargs(None, request) == {
        "enable_thinking": False,
        "reasoning_effort": "none",
    }

    # Unset effort must not inject any kwargs.
    request = SimpleNamespace(reasoning_effort=None)
    assert (
        serving._get_effective_chat_template_kwargs({"a": 1}, request) == {"a": 1}
    )

    # Explicit chat_template_kwargs take precedence over the request field
    # (enable_thinking is still derived from the request field).
    request = SimpleNamespace(reasoning_effort="low")
    assert serving._get_effective_chat_template_kwargs(
        {"reasoning_effort": "xhigh"}, request
    ) == {"enable_thinking": True, "reasoning_effort": "xhigh"}


def test_reasoning_effort_from_responses_request(serving: OpenAIServing):
    request = SimpleNamespace(reasoning=SimpleNamespace(effort="medium"))
    assert serving._get_effective_chat_template_kwargs(None, request) == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
