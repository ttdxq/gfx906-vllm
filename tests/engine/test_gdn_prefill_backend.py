# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import LlamaConfig

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_gdn_prefill_backend_cli_reaches_vllm_config(tmp_path):
    LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=4,
        vocab_size=128,
    ).save_pretrained(tmp_path)

    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    cli_args = parser.parse_args(
        [
            "--model",
            str(tmp_path),
            "--additional-config",
            '{"existing": true, "gdn_prefill_backend": "flashinfer"}',
            "--gdn-prefill-backend",
            "triton",
        ]
    )

    engine_args = EngineArgs.from_cli_args(cli_args)
    vllm_config = engine_args.create_engine_config()

    assert vllm_config.additional_config == {
        "existing": True,
        "gdn_prefill_backend": "triton",
    }
