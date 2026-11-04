# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp GGUF weights adapter.

Hybrid GDN/full-attention MoE checkpoints ("Qwen3.8 Flash Next").  The
GGUF architecture has no gguf-py tensor-name table, so the map is built
explicitly from the config. The PLE n-gram table stays quantized and
supports host storage with selected-row dequantization.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import gguf
import torch

from .base import BaseGGUFWeightsAdapter

if TYPE_CHECKING:
    from collections.abc import Generator

    from vllm.config import ModelConfig

_QWEN4_EXP_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")

# Hyper-connection projections are tiny (low-rank 320/10240-wide) plain
# modules, but the checkpoint quantizes them (Q4_K/Q5_0); they are
# dequantized on the host instead of building quantized modules.
_HC_PROJ_SUFFIXES = (
    ".input_mix_weight_down",
    ".input_mix_weight_up",
)


def _dequantize_hc_projection(name: str) -> bool:
    if (
        name.endswith(".input_mix_weight_up")
        and os.getenv("VLLM_QWEN4EXP_GGUF_HC_UP_QUANT", "0") == "1"
    ):
        return False
    return name.endswith(_HC_PROJ_SUFFIXES)


class Qwen4ExpGGUFAdapter(BaseGGUFWeightsAdapter):
    @classmethod
    def matches(cls, model_config: ModelConfig) -> bool:
        model_type = getattr(model_config.hf_config, "model_type", "")
        if model_type in _QWEN4_EXP_MODEL_TYPES:
            return True
        text_config = model_config.hf_config.get_text_config()
        return getattr(text_config, "model_type", "") in _QWEN4_EXP_MODEL_TYPES

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        text_config = model_config.hf_config.get_text_config()
        num_layers = text_config.num_hidden_layers
        layer_types = list(text_config.layer_types)

        name_map = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output.weight": "lm_head.weight",
            "output_hc_down.weight": (
                "model.hyper_connection_mixer.input_mix_weight_down.weight"
            ),
            "output_hc_up.weight": (
                "model.hyper_connection_mixer.input_mix_weight_up.weight"
            ),
            "output_hc_norm.weight": "model.hyper_connection_mixer.hc_norm.weight",
        }
        ple_layer_ids = list(text_config.ple_layer_ids)
        if len(ple_layer_ids) > 1:
            raise NotImplementedError("Qwen4Exp GGUF supports a single PLE table")
        if ple_layer_ids:
            ple_layer = int(ple_layer_ids[0]) - 1
            hf_ple_prefix = f"model.layers.{ple_layer}.ple"
            name_map["per_layer_token_embd.weight"] = (
                f"{hf_ple_prefix}.ple_embedding.ngram_embedding.weight"
            )
            for gguf_suffix, hf_suffix in {
                "ple_key.weight": "key_proj.weight",
                "ple_value.weight": "value_proj.weight",
                "ple_conv1d.weight": "conv1d.weight",
                "ple_norm_key.weight": "norm_key.weight",
                "ple_norm_query.weight": "norm_query.weight",
                "ple_norm_conv.weight": "norm_conv.weight",
            }.items():
                name_map[f"blk.{ple_layer}.{gguf_suffix}"] = (
                    f"{hf_ple_prefix}.{hf_suffix}"
                )

        # Hyper-connections replace the per-layer RMS norms: hc_norm is the
        # branch norm, the fused input-mix down-projection and the block
        # injection join the stacked checkpoint name the model expects.
        hc_maps = {
            "hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down.weight",
            "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
            "hc_attn_inject.weight": "attn_hyper_connection.block_inject_weight.weight",
            "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
            "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down.weight",
            "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
            "hc_ffn_inject.weight": "mlp_hyper_connection.block_inject_weight.weight",
            "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
            # Routed experts use the stacked-across-experts naming the MoE
            # loader splits per shard; shared experts stay whole.
            "ffn_gate_inp.weight": "mlp.gate.weight",
            "ffn_gate_exps.weight": "mlp.experts.0.gate_proj.weight",
            "ffn_up_exps.weight": "mlp.experts.0.up_proj.weight",
            "ffn_down_exps.weight": "mlp.experts.0.down_proj.weight",
            "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
            "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
            "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
            "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
        }
        # Gated-delta-net blocks (the majority of layers).
        linear_attn_maps = {
            "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
            "attn_gate.weight": "linear_attn.in_proj_z.weight",
            "ssm_beta.weight": "linear_attn.in_proj_b.weight",
            "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
            # GGUF stores -exp(A_log); transform_weights restores the log.
            "ssm_a": "linear_attn.A_log",
            "ssm_dt.bias": "linear_attn.dt_bias",
            "ssm_conv1d.weight": "linear_attn.conv1d.weight",
            "ssm_norm.weight": "linear_attn.norm.weight",
            "ssm_out.weight": "linear_attn.out_proj.weight",
        }
        # QSA full-attention blocks: Q packs the sigmoid output gate next to
        # each head; the indexer query/key projections are merged on load.
        # The k half maps to the real module name so the weight-type map
        # reports the merged module's checkpoint type (BF16 here) and the
        # loader builds it unquantized; the q half stages behind a
        # transform-only name that transform_weights folds into the k part.
        # (When a checkpoint quantizes both halves they must share one ggml
        # type: the merged module carries a single qweight_type.)
        full_attn_maps = {
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "attn_q_norm.weight": "self_attn.q_norm.weight",
            "attn_k_norm.weight": "self_attn.k_norm.weight",
            "indexer.q_proj.weight": "self_attn.indexer.index_qk_proj_q.weight",
            "indexer.k_proj.weight": "self_attn.indexer.index_qk_proj.weight",
            "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
            "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
        }

        for idx in range(num_layers):
            gguf_prefix = f"blk.{idx}"
            hf_prefix = f"model.layers.{idx}"
            family_maps = [hc_maps]
            if layer_types[idx] == "linear_attention":
                family_maps.append(linear_attn_maps)
            else:
                family_maps.append(full_attn_maps)
            for maps in family_maps:
                for gguf_suffix, hf_suffix in maps.items():
                    name_map[f"{gguf_prefix}.{gguf_suffix}"] = (
                        f"{hf_prefix}.{hf_suffix}"
                    )
        return name_map

    def transform_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        model_config: ModelConfig,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        index_parts: dict[str, dict[str, tuple[torch.Tensor, bool]]] = {}
        hc_qtypes: dict[str, torch.Tensor] = {}
        gdn_out_qtypes: dict[str, int] = {}
        ple_buffers: list[tuple[str, torch.Tensor]] = []
        text_config = model_config.hf_config.get_text_config()
        key_heads = text_config.linear_num_key_heads
        value_heads = text_config.linear_num_value_heads
        value_head_dim = text_config.linear_value_head_dim
        key_width = key_heads * text_config.linear_key_head_dim
        value_heads_per_key = value_heads // key_heads

        def restore_value_heads(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
            # GGUF tiles value heads across keys; the GDN core and TP
            # partitions expect adjacent value heads belonging to one key.
            return (
                weight.unflatten(0, (value_heads_per_key, key_heads, head_dim))
                .transpose(0, 1)
                .flatten(0, 2)
                .contiguous()
            )

        for layer_id in text_config.ple_layer_ids:
            ple_prefix = f"model.layers.{int(layer_id) - 1}.ple.ple_embedding"
            for config_name, buffer_name in {
                "ple_layer_multipliers": "layer_multipliers",
                "ple_head_offsets": "ngram_heads_offsets",
                "ple_head_vocab_sizes": "ngram_heads_vocab_sizes",
            }.items():
                values = getattr(text_config, config_name, None)
                if values:
                    ple_buffers.append(
                        (
                            f"{ple_prefix}.{buffer_name}",
                            torch.tensor(values, dtype=torch.int64),
                        )
                    )
        for name, weight in weights:
            if ple_buffers and not name.endswith("qweight_type"):
                yield from ple_buffers
                ple_buffers.clear()
            if value_heads_per_key > 1:
                if name.endswith(".linear_attn.out_proj.qweight_type"):
                    stem = name.removesuffix(".qweight_type")
                    gdn_out_qtypes[stem] = int(weight)
                    dtype_to_ggml = {
                        torch.float16: gguf.GGMLQuantizationType.F16,
                        torch.bfloat16: gguf.GGMLQuantizationType.BF16,
                        torch.float32: gguf.GGMLQuantizationType.F32,
                    }
                    yield name, torch.tensor(dtype_to_ggml[model_config.dtype])
                    continue
                if name.endswith(".linear_attn.out_proj.qweight"):
                    stem = name.removesuffix(".qweight")
                    qtype = gguf.GGMLQuantizationType(gdn_out_qtypes[stem])
                    # A value head can be narrower than a GGML quant block.
                    # Decode this small projection before permuting columns.
                    weight = torch.from_numpy(gguf.dequantize(weight.numpy(), qtype))
                    weight = restore_value_heads(weight.T, value_head_dim).T
                    yield name, weight.contiguous().to(model_config.dtype)
                    continue
                if name.endswith(".linear_attn.out_proj.weight"):
                    weight = restore_value_heads(
                        weight.T, value_head_dim
                    ).T.contiguous()
                elif name.endswith(
                    (
                        ".linear_attn.in_proj_qkv.weight",
                        ".linear_attn.in_proj_qkv.qweight",
                        ".linear_attn.conv1d.weight",
                    )
                ):
                    qk, value = weight.split(
                        [2 * key_width, value_heads * value_head_dim], dim=0
                    )
                    weight = torch.cat(
                        [qk, restore_value_heads(value, value_head_dim)], dim=0
                    )
                elif name.endswith(
                    (".linear_attn.in_proj_z.weight", ".linear_attn.in_proj_z.qweight")
                ):
                    weight = restore_value_heads(weight, value_head_dim)
                elif name.endswith(
                    (
                        ".linear_attn.in_proj_a.weight",
                        ".linear_attn.in_proj_a.qweight",
                        ".linear_attn.in_proj_b.weight",
                        ".linear_attn.in_proj_b.qweight",
                        ".linear_attn.A_log",
                        ".linear_attn.dt_bias",
                    )
                ):
                    weight = restore_value_heads(weight, 1)
            if name.endswith(".ngram_embedding.qweight_type"):
                yield name.removesuffix(".qweight_type") + ".weight_type", weight
                continue
            if name.endswith(".ngram_embedding.qweight"):
                yield name.removesuffix(".qweight") + ".weight", weight
                continue
            if name.endswith(".linear_attn.A_log"):
                if not bool((weight < 0).all()):
                    raise ValueError("Qwen4Exp GGUF ssm_a must contain negative decays")
                weight = torch.log(-weight.float())
            elif name.endswith(
                (
                    ".hc_norm.weight",
                    ".self_attn.q_norm.weight",
                    ".self_attn.k_norm.weight",
                    ".indexer.q_layernorm.weight",
                    ".indexer.k_layernorm.weight",
                    ".ple.norm_key.weight",
                    ".ple.norm_query.weight",
                    ".ple.norm_conv.weight",
                )
            ):
                weight = weight.float() - 1.0
            elif name.endswith(".ple.conv1d.weight") and weight.ndim == 2:
                weight = weight.unsqueeze(1)
            # The indexer ships q/k as separate GGUF tensors but the model
            # packs them into one projection (query heads first, then the
            # single key head).  Unquantized GGUF tensors arrive in HF
            # [out, in] orientation, so pack along dim 0; quantized halves
            # (same ggml type) pack as their raw packed rows.
            if name.endswith(
                (".indexer.index_qk_proj_q.weight", ".indexer.index_qk_proj_q.qweight")
            ):
                stem = name.rsplit(".index_qk_proj_q.", 1)[0] + "."
                packed = name.endswith(".qweight")
                index_parts.setdefault(stem, {})["q"] = (weight, packed)
                continue
            if name.endswith(".index_qk_proj_q.qweight_type"):
                # staging-module type entry: the merged module takes its
                # type from the k half.
                continue
            if (
                name.endswith(
                    (".indexer.index_qk_proj.weight", ".indexer.index_qk_proj.qweight")
                )
                and ".index_qk_proj_q." not in name
            ):
                stem = name.rsplit(".index_qk_proj.", 1)[0] + "."
                packed = name.endswith(".qweight")
                index_parts.setdefault(stem, {})["k"] = (weight, packed)
                continue
            if name.endswith(".qweight_type"):
                base = name[: -len(".qweight_type")]
                if _dequantize_hc_projection(base):
                    # Consumed by the dequant path below (types stream
                    # before weights).
                    hc_qtypes[base] = weight
                    continue
            elif name.endswith(".qweight"):
                base = name[: -len(".qweight")]
                if _dequantize_hc_projection(base):
                    # The hyper-connection mixers are plain unquantized
                    # modules (their down-projection also merges the block
                    # injection via the model's stacked weight mapper), so
                    # dequantize the packed checkpoint tensors on the host
                    # and stream them under the stacked checkpoint names.
                    qtype = gguf.GGMLQuantizationType(int(hc_qtypes[base]))
                    dequant = gguf.dequantize(weight.numpy(), qtype)
                    yield (
                        f"{base}.weight",
                        torch.from_numpy(dequant).to(model_config.dtype),
                    )
                    continue
            if name.endswith(".mlp.shared_expert_gate.weight") and weight.ndim == 1:
                # The router gate is a single-logit linear; the GGUF stores
                # its weight as a flat vector.
                weight = weight.view(1, -1)
            yield name, weight
        yield from ple_buffers
        for stem, parts in index_parts.items():
            if "q" not in parts or "k" not in parts:
                missing = "q_proj" if "q" not in parts else "k_proj"
                raise ValueError(
                    f"Qwen4Exp indexer tensor {stem}{missing}.weight is "
                    "missing its companion projection"
                )
            q_weight, q_packed = parts["q"]
            k_weight, k_packed = parts["k"]
            if q_packed != k_packed:
                raise ValueError(
                    f"Qwen4Exp indexer projections at {stem} mix quantized "
                    "and unquantized halves; both must share one layout"
                )
            suffix = ".qweight" if q_packed else ".weight"
            yield (
                f"{stem}index_qk_proj{suffix}",
                torch.cat([q_weight, k_weight], dim=0),
            )
