import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

PARSER_NAME = "qwen3_5"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _build_parser(
    qwen3_tokenizer: AutoTokenizer,
    *,
    enable_thinking: bool,
) -> ReasoningParser:
    parser_cls = ReasoningParserManager.get_reasoning_parser(PARSER_NAME)
    return parser_cls(
        qwen3_tokenizer,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )


def test_qwen3_5_no_boundary_defaults_to_reasoning(qwen3_tokenizer):
    output = qwen3_tokenizer.tokenize("This is a reasoning section")
    output_tokens = [qwen3_tokenizer.convert_tokens_to_string([token]) for token in output]
    reasoning, content = run_reasoning_extraction(
        _build_parser(qwen3_tokenizer, enable_thinking=True),
        output_tokens,
        streaming=False,
    )

    assert reasoning == "This is a reasoning section"
    assert content is None


def test_qwen3_5_streaming_thinking_disabled_emits_content(qwen3_tokenizer):
    output = qwen3_tokenizer.tokenize("This is the rest")
    output_tokens = [qwen3_tokenizer.convert_tokens_to_string([token]) for token in output]
    reasoning, content = run_reasoning_extraction(
        _build_parser(qwen3_tokenizer, enable_thinking=False),
        output_tokens,
        streaming=True,
    )

    assert reasoning is None
    assert content == "This is the rest"
