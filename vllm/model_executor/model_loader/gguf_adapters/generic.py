# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic GGUF weights adapter backed by gguf-py tensor-name tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseGGUFWeightsAdapter, build_name_map_via_gguf_py

if TYPE_CHECKING:
    from vllm.config import ModelConfig


class GenericGGUFWeightsAdapter(BaseGGUFWeightsAdapter):
    """Map names via gguf-py for any architecture it covers.

    Handles the HF→GGUF architecture renames and pre-seeds the packed
    expert (``*_exps``) entries for MoE families that gguf-py's tables do
    not express directly.
    """

    @classmethod
    def matches(cls, model_config: ModelConfig) -> bool:
        del model_config
        return True

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        gguf_to_hf_name_map: dict[str, str] = {}
        model_type = config.model_type
        # hack: ggufs have a different name than transformers
        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            # Gemma3 models use "gemma3_text" in HuggingFace but
            # "gemma3" in GGUF architecture naming
            model_type = "gemma3"
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):
            model_type = model_type.replace("_", "")
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )

        return build_name_map_via_gguf_py(
            model_config,
            gguf_model_type=model_type,
            name_map=gguf_to_hf_name_map,
        )
