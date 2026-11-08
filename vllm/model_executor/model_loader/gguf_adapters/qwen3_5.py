# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 family GGUF weights adapters (dense, MoE, MTP draft, multimodal)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import gguf
import torch

from .base import BaseGGUFWeightsAdapter, build_name_map_via_gguf_py

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_QWEN35_MODEL_TYPES = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")
_QWEN35_MOE_MODEL_TYPES = ("qwen3_5_moe", "qwen3_5_moe_text")
_QWEN35_MULTIMODAL_ARCHITECTURES = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)
# Backbone architectures whose layers include gated-delta-net blocks.
_QWEN35_SERVING_ARCHITECTURES = (
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
)


class Qwen35MtpGGUFAdapter(BaseGGUFWeightsAdapter):
    """Name map for Qwen3.5 MTP draft weights embedded in the GGUF."""

    @classmethod
    def matches(cls, model_config: ModelConfig) -> bool:
        return model_config.architecture in ("Qwen3_5MTP", "Qwen3_5MoeMTP")

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        text_config = model_config.hf_config.get_text_config()
        num_mtp_layers = getattr(text_config, "mtp_num_hidden_layers", 1)
        first_mtp_block = text_config.num_hidden_layers
        is_moe = text_config.model_type == "qwen3_5_moe_text"

        weights_map = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output.weight": "lm_head.weight",
        }
        for mtp_idx in range(num_mtp_layers):
            block_idx = first_mtp_block + mtp_idx
            gguf_prefix = f"blk.{block_idx}"
            hf_prefix = f"mtp.layers.{mtp_idx}"
            weights_map.update(
                {
                    f"{gguf_prefix}.nextn.eh_proj.weight": "mtp.fc.weight",
                    f"{gguf_prefix}.nextn.enorm.weight": (
                        "mtp.pre_fc_norm_embedding.weight"
                    ),
                    f"{gguf_prefix}.nextn.hnorm.weight": (
                        "mtp.pre_fc_norm_hidden.weight"
                    ),
                    f"{gguf_prefix}.nextn.shared_head_norm.weight": ("mtp.norm.weight"),
                    f"{gguf_prefix}.attn_q.weight": (
                        f"{hf_prefix}.self_attn.q_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_k.weight": (
                        f"{hf_prefix}.self_attn.k_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_v.weight": (
                        f"{hf_prefix}.self_attn.v_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_output.weight": (
                        f"{hf_prefix}.self_attn.o_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_q_norm.weight": (
                        f"{hf_prefix}.self_attn.q_norm.weight"
                    ),
                    f"{gguf_prefix}.attn_k_norm.weight": (
                        f"{hf_prefix}.self_attn.k_norm.weight"
                    ),
                    f"{gguf_prefix}.attn_norm.weight": (
                        f"{hf_prefix}.input_layernorm.weight"
                    ),
                    f"{gguf_prefix}.post_attention_norm.weight": (
                        f"{hf_prefix}.post_attention_layernorm.weight"
                    ),
                }
            )
            if is_moe:
                weights_map.update(
                    {
                        f"{gguf_prefix}.ffn_gate_inp.weight": (
                            f"{hf_prefix}.mlp.gate.weight"
                        ),
                        f"{gguf_prefix}.ffn_gate_exps.weight": (
                            f"{hf_prefix}.mlp.experts.0.gate_proj.weight"
                        ),
                        f"{gguf_prefix}.ffn_up_exps.weight": (
                            f"{hf_prefix}.mlp.experts.0.up_proj.weight"
                        ),
                        f"{gguf_prefix}.ffn_down_exps.weight": (
                            f"{hf_prefix}.mlp.experts.0.down_proj.weight"
                        ),
                        f"{gguf_prefix}.ffn_gate_shexp.weight": (
                            f"{hf_prefix}.mlp.shared_expert.gate_proj.weight"
                        ),
                        f"{gguf_prefix}.ffn_up_shexp.weight": (
                            f"{hf_prefix}.mlp.shared_expert.up_proj.weight"
                        ),
                        f"{gguf_prefix}.ffn_down_shexp.weight": (
                            f"{hf_prefix}.mlp.shared_expert.down_proj.weight"
                        ),
                        f"{gguf_prefix}.ffn_gate_inp_shexp.weight": (
                            f"{hf_prefix}.mlp.shared_expert_gate.weight"
                        ),
                    }
                )
            else:
                weights_map.update(
                    {
                        f"{gguf_prefix}.ffn_gate.weight": (
                            f"{hf_prefix}.mlp.gate_proj.weight"
                        ),
                        f"{gguf_prefix}.ffn_up.weight": (
                            f"{hf_prefix}.mlp.up_proj.weight"
                        ),
                        f"{gguf_prefix}.ffn_down.weight": (
                            f"{hf_prefix}.mlp.down_proj.weight"
                        ),
                    }
                )
        return weights_map


class Qwen35GGUFAdapter(BaseGGUFWeightsAdapter):
    """Qwen3.5 dense/MoE backbone, including mmproj-backed multimodal.

    The gated-delta-net (``linear_attn``) projections have no gguf-py
    names, so they are resolved through explicit patterns instead.
    """

    @classmethod
    def matches(cls, model_config: ModelConfig) -> bool:
        if model_config.architecture in ("Qwen3_5MTP", "Qwen3_5MoeMTP"):
            return False
        model_type = getattr(model_config.hf_config, "model_type", "")
        return model_type in _QWEN35_MODEL_TYPES

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        text_config = config.get_text_config()
        vllm_arch = model_config.architecture
        is_qwen35_multimodal = vllm_arch in _QWEN35_MULTIMODAL_ARCHITECTURES

        def qwen35_text_hf_name(name: str) -> str:
            if not is_qwen35_multimodal:
                return name
            if name.startswith("model."):
                return f"model.language_model.{name.removeprefix('model.')}"
            return name

        model_type = config.model_type
        is_moe = model_type in _QWEN35_MOE_MODEL_TYPES
        gguf_to_hf_name_map: dict[str, str] = {}
        if is_moe:
            for idx in range(text_config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    qwen35_text_hf_name(
                        f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                    )
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    qwen35_text_hf_name(
                        f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                    )
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    qwen35_text_hf_name(
                        f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                    )
                )

        def hf_name_preprocess(hf_name: str) -> str:
            # gguf-py expects flattened names: the language model keeps its
            # 'model.' prefix while vision params drop the 'model.' level.
            if is_qwen35_multimodal and hf_name.startswith("model.language_model."):
                return "model." + hf_name.removeprefix("model.language_model.")
            if is_qwen35_multimodal and hf_name.startswith("model.visual."):
                return hf_name.removeprefix("model.")
            return hf_name

        # Only reachable when the mmproj companion file exists.
        base_name_overrides = {
            "visual.merger.norm": "visual.merger.ln_q",
            "visual.merger.linear_fc1": "visual.merger.mlp.0",
            "visual.merger.linear_fc2": "visual.merger.mlp.2",
        }

        def resolve_missing(
            hf_name: str, base_name: str, suffix: str
        ) -> str | None:
            # Gated-delta-net projections have no gguf-py tensor names.
            if m := re.fullmatch(
                r"model\.layers\.(?P<bid>\d+)\.linear_attn\.dt_bias", hf_name
            ):
                return f"blk.{m['bid']}.ssm_dt.bias"
            if m := re.fullmatch(
                r"model\.layers\.(?P<bid>\d+)\.linear_attn\.in_proj_qkv",
                base_name,
            ):
                return f"blk.{m['bid']}.attn_qkv.{suffix}"
            if m := re.fullmatch(
                r"model\.layers\.(?P<bid>\d+)\.linear_attn\.in_proj_z",
                base_name,
            ):
                return f"blk.{m['bid']}.attn_gate.{suffix}"
            if m := re.fullmatch(
                r"model\.layers\.(?P<bid>\d+)\.linear_attn\.in_proj_b",
                base_name,
            ):
                return f"blk.{m['bid']}.ssm_beta.{suffix}"
            if m := re.fullmatch(
                r"model\.layers\.(?P<bid>\d+)\.linear_attn\.in_proj_a",
                base_name,
            ):
                return f"blk.{m['bid']}.ssm_alpha.{suffix}"
            return None

        def dummy_config_override(cfg, is_multimodal: bool):
            # Text-only checkpoints of multimodal configs instantiate the
            # language model alone.
            if (
                not is_multimodal
                and cfg.model_type in ("qwen3_5", "qwen3_5_moe")
            ):
                return cfg.get_text_config()
            return cfg

        gguf_to_hf_name_map = build_name_map_via_gguf_py(
            model_config,
            gguf_model_type="qwen35moe" if is_moe else "qwen35",
            name_map=gguf_to_hf_name_map,
            hf_name_preprocess=hf_name_preprocess,
            base_name_overrides=base_name_overrides,
            resolve_missing=resolve_missing,
            dummy_config_override=dummy_config_override,
        )

        if vllm_arch in _QWEN35_SERVING_ARCHITECTURES:
            for idx in range(text_config.num_hidden_layers):
                if text_config.layer_types[idx] != "linear_attention":
                    continue
                gguf_to_hf_name_map[f"blk.{idx}.attn_qkv.weight"] = (
                    qwen35_text_hf_name(
                        f"model.layers.{idx}.linear_attn.in_proj_qkv.weight"
                    )
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_gate.weight"] = (
                    qwen35_text_hf_name(
                        f"model.layers.{idx}.linear_attn.in_proj_z.weight"
                    )
                )
                gguf_to_hf_name_map[f"blk.{idx}.ssm_beta.weight"] = (
                    qwen35_text_hf_name(
                        f"model.layers.{idx}.linear_attn.in_proj_b.weight"
                    )
                )
                gguf_to_hf_name_map[f"blk.{idx}.ssm_alpha.weight"] = (
                    qwen35_text_hf_name(
                        f"model.layers.{idx}.linear_attn.in_proj_a.weight"
                    )
                )
        return gguf_to_hf_name_map

    def transform_mmproj_weights(
        self,
        weights,
        model_config: ModelConfig,
        gguf_to_hf_name_map: dict[str, str],
        mmproj_file,
    ):
        del model_config
        # The temporal half of the patch embedding ships as a separate
        # 'v.patch_embd.weight.1' tensor; stack it with the spatial kernel.
        reader = gguf.GGUFReader(str(mmproj_file))
        temporal_patch = next(
            (
                torch.tensor(tensor.data)
                for tensor in reader.tensors
                if tensor.name == "v.patch_embd.weight.1"
            ),
            None,
        )
        patch_embed_name = gguf_to_hf_name_map.get("v.patch_embd.weight")
        for name, weight in weights:
            if (
                temporal_patch is not None
                and name == patch_embed_name
                and weight.ndim == 4
            ):
                weight = torch.stack((weight, temporal_patch), dim=2)
            yield name, weight
