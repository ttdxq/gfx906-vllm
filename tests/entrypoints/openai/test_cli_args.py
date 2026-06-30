# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from argparse import Namespace

import pytest

import vllm.entrypoints.openai.api_server as api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.utils.argparse_utils import FlexibleArgumentParser

from ...utils import VLLM_PATH

LORA_MODULE = {
    "name": "module2",
    "path": "/path/to/module2",
    "base_model_name": "llama",
}
CHATML_JINJA_PATH = VLLM_PATH / "examples/template_chatml.jinja"
assert CHATML_JINJA_PATH.exists()


@pytest.fixture
def serve_parser():
    parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
    return make_arg_parser(parser)


### Test config parsing
def test_config_arg_parsing(serve_parser, cli_config_file):
    args = serve_parser.parse_args([])
    assert args.port == 8000
    args = serve_parser.parse_args(["--config", cli_config_file])
    assert args.port == 12312
    args = serve_parser.parse_args(
        [
            "--config",
            cli_config_file,
            "--port",
            "9000",
        ]
    )
    assert args.port == 9000
    args = serve_parser.parse_args(
        [
            "--port",
            "9000",
            "--config",
            cli_config_file,
        ]
    )
    assert args.port == 9000


### Tests for LoRA module parsing
def test_valid_key_value_format(serve_parser):
    # Test old format: name=path
    args = serve_parser.parse_args(
        [
            "--lora-modules",
            "module1=/path/to/module1",
        ]
    )
    expected = [LoRAModulePath(name="module1", path="/path/to/module1")]
    assert args.lora_modules == expected


def test_valid_json_format(serve_parser):
    # Test valid JSON format input
    args = serve_parser.parse_args(
        [
            "--lora-modules",
            json.dumps(LORA_MODULE),
        ]
    )
    expected = [
        LoRAModulePath(name="module2", path="/path/to/module2", base_model_name="llama")
    ]
    assert args.lora_modules == expected


def test_invalid_json_format(serve_parser):
    # Test invalid JSON format input, missing closing brace
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            ["--lora-modules", '{"name": "module3", "path": "/path/to/module3"']
        )


def test_invalid_type_error(serve_parser):
    # Test type error when values are not JSON or key=value
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            [
                "--lora-modules",
                "invalid_format",  # This is not JSON or key=value format
            ]
        )


def test_invalid_json_field(serve_parser):
    # Test valid JSON format but missing required fields
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            [
                "--lora-modules",
                '{"name": "module4"}',  # Missing required 'path' field
            ]
        )


def test_api_server_main_copies_model_tag(monkeypatch):
    captured = {}

    def fake_parse_args(self, args=None, namespace=None):
        return Namespace(
            model_tag="/root/model/Qwen3.5-4B-UD-Q4_K_XL.gguf",
            model="Qwen/Qwen3-0.6B",
        )

    def fake_validate_parsed_serve_args(args):
        captured["model"] = args.model

    def fake_run_server(args):
        captured["run_server_model"] = args.model

    monkeypatch.setattr(api_server, "cli_env_setup", lambda: None)
    monkeypatch.setattr(api_server, "make_arg_parser", lambda parser: parser)
    monkeypatch.setattr(api_server.FlexibleArgumentParser, "parse_args", fake_parse_args)
    monkeypatch.setattr(
        api_server,
        "validate_parsed_serve_args",
        fake_validate_parsed_serve_args,
    )
    monkeypatch.setattr(api_server, "run_server", fake_run_server)
    monkeypatch.setattr(api_server.uvloop, "run", lambda coro: coro)

    api_server.main()

    assert captured["model"] == "/root/model/Qwen3.5-4B-UD-Q4_K_XL.gguf"
    assert captured["run_server_model"] == "/root/model/Qwen3.5-4B-UD-Q4_K_XL.gguf"


def test_empty_values(serve_parser):
    # Test when no LoRA modules are provided
    args = serve_parser.parse_args(["--lora-modules", ""])
    assert args.lora_modules == []


def test_multiple_valid_inputs(serve_parser):
    # Test multiple valid inputs (both old and JSON format)
    args = serve_parser.parse_args(
        [
            "--lora-modules",
            "module1=/path/to/module1",
            json.dumps(LORA_MODULE),
        ]
    )
    expected = [
        LoRAModulePath(name="module1", path="/path/to/module1"),
        LoRAModulePath(
            name="module2", path="/path/to/module2", base_model_name="llama"
        ),
    ]
    assert args.lora_modules == expected


### Tests for serve argument validation that run prior to loading
def test_enable_auto_choice_passes_without_tool_call_parser(serve_parser):
    """Ensure validation fails if tool choice is enabled with no call parser"""
    # If we enable-auto-tool-choice, explode with no tool-call-parser
    args = serve_parser.parse_args(args=["--enable-auto-tool-choice"])
    with pytest.raises(TypeError):
        validate_parsed_serve_args(args)


def test_enable_auto_choice_passes_with_tool_call_parser(serve_parser):
    """Ensure validation passes with tool choice enabled with a call parser"""
    args = serve_parser.parse_args(
        args=[
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "mistral",
        ]
    )
    validate_parsed_serve_args(args)


def test_enable_auto_choice_fails_with_enable_reasoning(serve_parser):
    """Ensure validation fails if reasoning is enabled with auto tool choice"""
    args = serve_parser.parse_args(
        args=[
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "deepseek_r1",
        ]
    )
    with pytest.raises(TypeError):
        validate_parsed_serve_args(args)


def test_passes_with_reasoning_parser(serve_parser):
    """Ensure validation passes if reasoning is enabled
    with a reasoning parser"""
    args = serve_parser.parse_args(
        args=[
            "--reasoning-parser",
            "deepseek_r1",
        ]
    )
    validate_parsed_serve_args(args)


def test_chat_template_validation_for_happy_paths(serve_parser):
    """Ensure validation passes if the chat template exists"""
    args = serve_parser.parse_args(
        args=["--chat-template", CHATML_JINJA_PATH.absolute().as_posix()]
    )
    validate_parsed_serve_args(args)


def test_chat_template_validation_for_sad_paths(serve_parser):
    """Ensure validation fails if the chat template doesn't exist"""
    args = serve_parser.parse_args(args=["--chat-template", "does/not/exist"])
    with pytest.raises(ValueError):
        validate_parsed_serve_args(args)


@pytest.mark.parametrize(
    "cli_args, expected_middleware",
    [
        (
            ["--middleware", "middleware1", "--middleware", "middleware2"],
            ["middleware1", "middleware2"],
        ),
        ([], []),
    ],
)
def test_middleware(serve_parser, cli_args, expected_middleware):
    """Ensure multiple middleware args are parsed properly"""
    args = serve_parser.parse_args(args=cli_args)
    assert args.middleware == expected_middleware
