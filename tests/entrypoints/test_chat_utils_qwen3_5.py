from types import SimpleNamespace

from vllm.entrypoints.chat_utils import _detect_content_format, apply_hf_chat_template


def test_qwen3_5_content_loop_is_detected_as_openai_format():
    template = """
    {% for message in messages %}
      {% set content = message['content'] %}
      {% for item in content %}{{ item['text'] }}{% endfor %}
    {% endfor %}
    """

    assert _detect_content_format(template, default="string") == "openai"


def test_apply_hf_chat_template_strips_empty_qwen3_5_thinking_scaffold():
    class MockQwen3_5Tokenizer:
        chat_template = "{% if enable_thinking is defined and enable_thinking is false %}dummy{% endif %}"
        name_or_path = "/tmp/qwen3_5_tokenizer"

        def get_chat_template(self, chat_template=None, tools=None):
            return self.chat_template

        def apply_chat_template(
            self,
            conversation,
            tools=None,
            chat_template=None,
            tokenize=False,
            **kwargs,
        ):
            assert kwargs["enable_thinking"] is False
            return (
                "<|im_start|>user\nquestion<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )

    tokenizer = MockQwen3_5Tokenizer()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen3_5_text",
            architectures=["Qwen3_5ForCausalLM"],
        ),
        trust_remote_code=False,
        tokenizer="/tmp/qwen3_5_tokenizer",
    )

    rendered = apply_hf_chat_template(
        tokenizer,
        conversation=[{"role": "user", "content": "question"}],
        chat_template=None,
        tools=None,
        model_config=model_config,
        enable_thinking=False,
    )

    assert rendered == "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n"


def test_apply_hf_chat_template_keeps_empty_scaffold_for_non_qwen3_5():
    class MockOtherTokenizer:
        chat_template = "{% if enable_thinking is defined and enable_thinking is false %}dummy{% endif %}"
        name_or_path = "/tmp/other_tokenizer"

        def get_chat_template(self, chat_template=None, tools=None):
            return self.chat_template

        def apply_chat_template(
            self,
            conversation,
            tools=None,
            chat_template=None,
            tokenize=False,
            **kwargs,
        ):
            assert kwargs["enable_thinking"] is False
            return (
                "<|im_start|>user\nquestion<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )

    tokenizer = MockOtherTokenizer()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="not_qwen3_5",
            architectures=["OtherArchitecture"],
        ),
        trust_remote_code=False,
        tokenizer="/tmp/other_tokenizer",
    )

    rendered = apply_hf_chat_template(
        tokenizer,
        conversation=[{"role": "user", "content": "question"}],
        chat_template=None,
        tools=None,
        model_config=model_config,
        enable_thinking=False,
    )

    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
