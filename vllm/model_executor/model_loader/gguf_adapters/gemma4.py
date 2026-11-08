# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 family GGUF weights adapters (text, multimodal, MTP draft)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseGGUFWeightsAdapter

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch

    from vllm.config import ModelConfig


class Gemma4GGUFAdapter(BaseGGUFWeightsAdapter):
    @classmethod
    def matches(cls, model_config: ModelConfig) -> bool:
        return model_config.architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
        )

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        text_config = model_config.hf_config.get_text_config()
        architecture = model_config.architecture
        prefix = (
            "model.language_model."
            if architecture == "Gemma4ForConditionalGeneration"
            else "model."
        )
        weights_map = {
            "token_embd.weight": f"{prefix}embed_tokens.weight",
            "output_norm.weight": f"{prefix}norm.weight",
        }
        for layer_idx, layer_type in enumerate(text_config.layer_types):
            gguf_prefix = f"blk.{layer_idx}"
            hf_prefix = f"{prefix}layers.{layer_idx}"
            weights_map.update(
                {
                    f"{gguf_prefix}.layer_output_scale.weight": (
                        f"{hf_prefix}.layer_scalar"
                    ),
                    f"{gguf_prefix}.attn_norm.weight": (
                        f"{hf_prefix}.input_layernorm.weight"
                    ),
                    f"{gguf_prefix}.attn_q.weight": (
                        f"{hf_prefix}.self_attn.q_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_q_norm.weight": (
                        f"{hf_prefix}.self_attn.q_norm.weight"
                    ),
                    f"{gguf_prefix}.attn_k.weight": (
                        f"{hf_prefix}.self_attn.k_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_k_norm.weight": (
                        f"{hf_prefix}.self_attn.k_norm.weight"
                    ),
                    f"{gguf_prefix}.attn_output.weight": (
                        f"{hf_prefix}.self_attn.o_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_gate.weight": (
                        f"{hf_prefix}.mlp.gate_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_up.weight": (
                        f"{hf_prefix}.mlp.up_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_down.weight": (
                        f"{hf_prefix}.mlp.down_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_norm.weight": (
                        f"{hf_prefix}.pre_feedforward_layernorm.weight"
                    ),
                    f"{gguf_prefix}.post_attention_norm.weight": (
                        f"{hf_prefix}.post_attention_layernorm.weight"
                    ),
                    f"{gguf_prefix}.post_ffw_norm.weight": (
                        f"{hf_prefix}.post_feedforward_layernorm.weight"
                    ),
                    f"{gguf_prefix}.post_ffw_norm_1.weight": (
                        f"{hf_prefix}.post_feedforward_layernorm_1.weight"
                    ),
                    f"{gguf_prefix}.post_ffw_norm_2.weight": (
                        f"{hf_prefix}.post_feedforward_layernorm_2.weight"
                    ),
                    f"{gguf_prefix}.pre_ffw_norm_2.weight": (
                        f"{hf_prefix}.pre_feedforward_layernorm_2.weight"
                    ),
                    # MoE: the fused gate_up expert tensor is split into
                    # gate/up halves by transform_weights before it reaches
                    # the MoE weight loader.
                    f"{gguf_prefix}.ffn_gate_up_exps.weight": (
                        f"{hf_prefix}.experts.0.gate_up_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_down_exps.weight": (
                        f"{hf_prefix}.experts.0.down_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_gate_inp.weight": (
                        f"{hf_prefix}.router.proj.weight"
                    ),
                    # Router input scale and per-expert output scale. The
                    # model folds these into its routing math; skipping them
                    # silently leaves both at their ones init.
                    f"{gguf_prefix}.ffn_gate_inp.scale": (
                        f"{hf_prefix}.router.scale"
                    ),
                    f"{gguf_prefix}.ffn_down_exps.scale": (
                        f"{hf_prefix}.router.per_expert_scale"
                    ),
                }
            )
            if layer_type == "sliding_attention":
                weights_map[f"{gguf_prefix}.attn_v.weight"] = (
                    f"{hf_prefix}.self_attn.v_proj.weight"
                )
        if architecture == "Gemma4ForConditionalGeneration":
            vision_prefix = "model.vision_tower."
            weights_map["mm.input_projection.weight"] = (
                "model.embed_vision.embedding_projection.weight"
            )
            # The tower registers these as buffers with torch.empty init;
            # skipping them feeds uninitialized values into the
            # post-pooler standardization.
            weights_map["v.std_bias"] = f"{vision_prefix}std_bias"
            weights_map["v.std_scale"] = f"{vision_prefix}std_scale"
            weights_map["v.patch_embd.weight"] = (
                f"{vision_prefix}patch_embedder.input_proj.weight"
            )
            weights_map["v.position_embd.weight"] = (
                f"{vision_prefix}patch_embedder.position_embedding_table"
            )
            vision_layers = model_config.hf_config.vision_config.num_hidden_layers
            for layer_idx in range(vision_layers):
                gguf_prefix = f"v.blk.{layer_idx}"
                hf_prefix = f"{vision_prefix}encoder.layers.{layer_idx}"
                weights_map.update(
                    {
                        f"{gguf_prefix}.ln1.weight": (
                            f"{hf_prefix}.input_layernorm.weight"
                        ),
                        f"{gguf_prefix}.attn_q.weight": (
                            f"{hf_prefix}.self_attn.q_proj.linear.weight"
                        ),
                        f"{gguf_prefix}.attn_k.weight": (
                            f"{hf_prefix}.self_attn.k_proj.linear.weight"
                        ),
                        f"{gguf_prefix}.attn_v.weight": (
                            f"{hf_prefix}.self_attn.v_proj.linear.weight"
                        ),
                        f"{gguf_prefix}.attn_out.weight": (
                            f"{hf_prefix}.self_attn.o_proj.linear.weight"
                        ),
                        f"{gguf_prefix}.attn_q_norm.weight": (
                            f"{hf_prefix}.self_attn.q_norm.weight"
                        ),
                        f"{gguf_prefix}.attn_k_norm.weight": (
                            f"{hf_prefix}.self_attn.k_norm.weight"
                        ),
                        f"{gguf_prefix}.ffn_gate.weight": (
                            f"{hf_prefix}.mlp.gate_proj.linear.weight"
                        ),
                        f"{gguf_prefix}.ffn_up.weight": (
                            f"{hf_prefix}.mlp.up_proj.linear.weight"
                        ),
                        f"{gguf_prefix}.ffn_down.weight": (
                            f"{hf_prefix}.mlp.down_proj.linear.weight"
                        ),
                        f"{gguf_prefix}.attn_post_norm.weight": (
                            f"{hf_prefix}.post_attention_layernorm.weight"
                        ),
                        f"{gguf_prefix}.ffn_post_norm.weight": (
                            f"{hf_prefix}.post_feedforward_layernorm.weight"
                        ),
                        f"{gguf_prefix}.ln2.weight": (
                            f"{hf_prefix}.pre_feedforward_layernorm.weight"
                        ),
                    }
                )
        return weights_map

    def transform_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        model_config: ModelConfig,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Split fused GGUF MoE gate_up expert tensors into gate/up pairs.

        Gemma4 GGUF stores one ``ffn_gate_up_exps`` tensor per layer holding
        both projections of every expert.  The GGUF reader exposes it as an
        expert-major ``[experts, 2 * intermediate, packed]`` block-packed
        array; the MoE weight loader expects separate gate_proj/up_proj
        shards, so re-emit each fused tensor as its two dim-1 halves (gate
        first, up second) under both shard names.
        """
        del model_config
        for name, weight in weights:
            if ".gate_up_proj." not in name:
                yield name, weight
                continue
            stem, _, suffix = name.rpartition(".gate_up_proj.")
            if suffix == "qweight_type":
                yield f"{stem}.gate_proj.qweight_type", weight
                yield f"{stem}.up_proj.qweight_type", weight
                continue
            if weight.dim() != 3 or weight.shape[1] % 2 != 0:
                raise ValueError(
                    f"Cannot split fused GGUF gate_up tensor {name!r} "
                    f"with shape {tuple(weight.shape)}"
                )
            gate, up = weight.chunk(2, dim=1)
            target = "qweight" if suffix == "qweight" else "weight"
            yield f"{stem}.gate_proj.{target}", gate.contiguous()
            yield f"{stem}.up_proj.{target}", up.contiguous()

    def transform_mmproj_weights(
        self,
        weights,
        model_config: ModelConfig,
        gguf_to_hf_name_map,
        mmproj_file,
    ):
        del model_config, gguf_to_hf_name_map, mmproj_file
        # The vision patch projection ships as a conv kernel in ggml order
        # [kw, kh, channels, out], i.e. [out, channels, kh, kw] once read as
        # a torch tensor. HF patchifies each image as (patch_h, patch_w,
        # channel) flattened with the channel fastest, so the Linear weight
        # must be permuted to [out, kh, kw, channels] before flattening;
        # flattening the raw layout scrambles the patch pixels.
        for name, weight in weights:
            if (
                name.endswith("patch_embedder.input_proj.weight")
                and weight.ndim == 4
            ):
                weight = weight.permute(0, 2, 3, 1).flatten(1)
            yield name, weight


class Gemma4MtpGGUFAdapter(BaseGGUFWeightsAdapter):
    @classmethod
    def matches(cls, model_config: ModelConfig) -> bool:
        return model_config.architecture == "Gemma4MTPModel"

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        text_config = model_config.hf_config.get_text_config()
        weights_map = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output_norm.weight": "model.norm.weight",
            "nextn.pre_projection.weight": "pre_projection.weight",
            "nextn.post_projection.weight": "post_projection.weight",
        }
        for layer_idx in range(text_config.num_hidden_layers):
            gguf_prefix = f"blk.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"
            weights_map.update(
                {
                    f"{gguf_prefix}.attn_norm.weight": (
                        f"{hf_prefix}.input_layernorm.weight"
                    ),
                    f"{gguf_prefix}.layer_output_scale.weight": (
                        f"{hf_prefix}.layer_scalar"
                    ),
                    f"{gguf_prefix}.attn_q.weight": (
                        f"{hf_prefix}.self_attn.q_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_q_norm.weight": (
                        f"{hf_prefix}.self_attn.q_norm.weight"
                    ),
                    f"{gguf_prefix}.attn_output.weight": (
                        f"{hf_prefix}.self_attn.o_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_gate.weight": (
                        f"{hf_prefix}.mlp.gate_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_up.weight": (
                        f"{hf_prefix}.mlp.up_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_down.weight": (
                        f"{hf_prefix}.mlp.down_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_norm.weight": (
                        f"{hf_prefix}.pre_feedforward_layernorm.weight"
                    ),
                    f"{gguf_prefix}.post_attention_norm.weight": (
                        f"{hf_prefix}.post_attention_layernorm.weight"
                    ),
                    f"{gguf_prefix}.post_ffw_norm.weight": (
                        f"{hf_prefix}.post_feedforward_layernorm.weight"
                    ),
                }
            )
        return weights_map
