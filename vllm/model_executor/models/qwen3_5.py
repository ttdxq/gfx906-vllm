# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

import os
import typing
from collections.abc import Callable, Iterable
from inspect import signature

import torch
from einops import rearrange
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    VllmConfig,
)
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gguf import (
    try_gguf_rms_norm_gated_out_proj_mmvq,
    try_grouped_gguf_linear_mmvq,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _qwen35_env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _qwen35_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _qwen35_env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _append_qwen35_load_debug(message: str) -> None:
    debug_file = os.environ.get("VLLM_QWEN35_LOAD_DEBUG_FILE")
    if not debug_file:
        return
    logger.warning(message)
    try:
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


def _is_qwen35_gguf_projection_aux_param(name: str) -> bool:
    if ".linear_attn.in_proj_" not in name:
        return False
    if not name.endswith((".qweight", ".qweight_type")):
        return False
    return any(
        f".{proj}." in name
        for proj in (
            "in_proj_qkvz",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_ba",
            "in_proj_b",
            "in_proj_a",
        )
    )


def _maybe_unsqueeze_shared_expert_gate(
    name: str, param: torch.Tensor, loaded_weight: torch.Tensor
) -> torch.Tensor:
    if (
        name.endswith("shared_expert_gate.weight")
        and loaded_weight.ndim == 1
        and param.ndim == 2
        and param.shape[0] == 1
        and param.shape[1] == loaded_weight.shape[0]
    ):
        return loaded_weight.unsqueeze(0)
    return loaded_weight


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
    def __init__(
        self,
        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig,
        model_config=None,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        speculative_config=None,
        split_projections: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            speculative_config=speculative_config,
            prefix=prefix,
        )
        def _env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.lower() in {"1", "true", "yes", "on"}

        self.split_projections = split_projections
        capability = current_platform.get_device_capability()
        self._qwen35_is_gfx906_rocm = (
            current_platform.is_rocm()
            and capability is not None
            and capability.major == 9
            and capability.minor == 0
        )
        self.expand_qk_heads_for_gdn = _env_bool(
            "VLLM_QWEN35_EXPAND_QK", True
        )
        # The chunk path preserves the packed-decode state layout across
        # scheduler chunks. Keep the recurrent path as a debugging fallback.
        default_recurrent_prefill = False
        self.use_recurrent_prefill_for_gdn = _env_bool(
            "VLLM_QWEN35_REC_PREFILL", default_recurrent_prefill
        )
        self.use_local_recurrent_decode_for_gdn = _env_bool(
            "VLLM_QWEN35_LOCAL_REC_DECODE", False
        )
        self.use_qk_l2norm_in_kernel_for_gdn = _env_bool(
            "VLLM_QWEN35_QK_L2NORM", True
        )
        self.use_grouped_gguf_mmvq = _env_bool("VLLM_QWEN35_GROUPED_GGUF_MMVQ", True)
        self.use_grouped_gguf_ba_mmvq = _env_bool(
            "VLLM_QWEN35_GROUPED_GGUF_BA_MMVQ", True
        )
        self.enable_packed_recurrent_decode = self._use_gfx906_packed_decode_path()
        self.enable_combined_packed_decode = (
            self.enable_packed_recurrent_decode
            and self._qwen35_is_gfx906_rocm
            and _env_bool("VLLM_QWEN35_COMBINED_PACKED_DECODE", True)
        )
        self.use_transposed_state_for_packed_decode = self.split_projections
        self.use_empty_core_attn_out_for_single_token = (
            self.split_projections
            and self._qwen35_is_gfx906_rocm
            and _env_bool("VLLM_QWEN35_EMPTY_CORE_ATTN_SINGLE_TOKEN", True)
        )
        self.use_tiled_qk_expand = _env_bool(
            "VLLM_QWEN35_TILED_QK_EXPAND", self.split_projections
        )
        self.call_b_first = os.getenv("VLLM_QWEN35_CALL_ORDER", "ba").lower() != "ab"
        if self.split_projections:
            del self.in_proj_qkvz
            del self.in_proj_ba
            self.in_proj_qkv = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[self.key_dim, self.key_dim, self.value_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkv",
            )
            self.in_proj_z = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.value_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_z",
            )
            self.in_proj_b = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.num_v_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_b",
            )
            self.in_proj_a = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.num_v_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_a",
            )
        else:
            self.in_proj_qkvz.output_sizes = [
                self.key_dim,
                self.key_dim,
                self.value_dim,
                self.value_dim,
            ]
            self.in_proj_ba.output_sizes = [self.num_v_heads, self.num_v_heads]
        self._qwen35_linear_attn_profile_count = 0
        self._qwen35_linear_attn_profile_enabled = _qwen35_env_flag(
            "VLLM_QWEN35_LINEAR_ATTN_PROFILE"
        )
        self._qwen35_linear_attn_profile_max_tokens = _qwen35_env_optional_int(
            "VLLM_QWEN35_LINEAR_ATTN_PROFILE_MAX_TOKENS"
        )
        self._qwen35_linear_attn_profile_limit = _qwen35_env_int(
            "VLLM_QWEN35_LINEAR_ATTN_PROFILE_LIMIT", 4
        )
        self._qwen35_gguf_debug_enabled = _env_bool(
            "VLLM_QWEN35_GGUF_DEBUG", False
        )
        self._qwen35_gguf_debug_target_prefix = os.getenv(
            "VLLM_QWEN35_GGUF_DEBUG_LAYER", "model.layers.0.linear_attn"
        )
        self._qwen35_gguf_debug_file = os.getenv("VLLM_QWEN35_GGUF_DEBUG_FILE")

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        capability = current_platform.get_device_capability()
        if (
            current_platform.is_rocm()
            and capability is not None
            and capability.major == 9
            and capability.minor == 0
        ):
            if self.cache_config.mamba_cache_dtype != "auto":
                return super().get_state_dtype()
            if self.cache_config.mamba_ssm_cache_dtype != "auto":
                _, temporal_dtype = super().get_state_dtype()
                return (torch.float32, temporal_dtype)
            return (torch.float32, torch.float32)
        return super().get_state_dtype()

    def _expand_qk_heads_for_gdn(
        self,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if query is None or key is None:
            return query, key

        head_ratio = self.num_v_heads // self.num_k_heads
        if head_ratio <= 1:
            return query.contiguous(), key.contiguous()

        if self.use_tiled_qk_expand:
            query = query.repeat(1, 1, head_ratio, 1)
            key = key.repeat(1, 1, head_ratio, 1)
        else:
            query = query.repeat_interleave(head_ratio, dim=2)
            key = key.repeat_interleave(head_ratio, dim=2)
        return query.contiguous(), key.contiguous()

    def _use_gfx906_packed_decode_path(self) -> bool:
        raw = os.getenv(
            "VLLM_QWEN35_PACKED_DECODE",
            "1" if self.split_projections else "0",
        )
        return raw.lower() in {"1", "true", "yes", "on"}

    def _use_tiled_qk_head_mapping_for_packed_decode(self) -> bool:
        raw = os.getenv(
            "VLLM_QWEN35_PACKED_DECODE_TILED_QK",
            "1" if self.split_projections else "0",
        )
        return raw.lower() in {"1", "true", "yes", "on"}

    def _maybe_log_debug_stats(self, **tensors: torch.Tensor) -> None:
        if not self._qwen35_gguf_debug_enabled:
            return
        target_prefix = self._qwen35_gguf_debug_target_prefix
        prefix_matches = self.prefix == target_prefix or self.prefix.endswith(
            ".layers.0.linear_attn"
        )
        if not prefix_matches:
            return

        def summarize(name: str, tensor: torch.Tensor) -> str:
            tensor_f32 = tensor.detach().float()
            finite = torch.isfinite(tensor_f32)
            finite_ratio = finite.float().mean().item() if tensor.numel() else 1.0
            return (
                f"{name}: shape={tuple(tensor.shape)} "
                f"mean={tensor_f32.mean().item():.6g} "
                f"std={tensor_f32.std(unbiased=False).item():.6g} "
                f"min={tensor_f32.min().item():.6g} "
                f"max={tensor_f32.max().item():.6g} "
                f"finite={finite_ratio:.6f}"
            )

        lines = [
            f"[qwen3.5 debug] prefix={self.prefix}",
            *[summarize(name, tensor) for name, tensor in tensors.items()],
        ]
        logger.warning("\n".join(lines))
        debug_file = self._qwen35_gguf_debug_file
        if debug_file:
            try:
                with open(debug_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            except Exception:
                pass

    def _use_qwen35_linear_attn_profile(self, hidden_states: torch.Tensor) -> bool:
        if not self._qwen35_linear_attn_profile_enabled:
            return False
        try:
            if torch._dynamo.is_compiling():
                return False
        except AttributeError:
            pass
        if not hidden_states.is_cuda:
            return False
        max_tokens = self._qwen35_linear_attn_profile_max_tokens
        if max_tokens is not None and hidden_states.shape[0] > max_tokens:
            return False
        return self._qwen35_linear_attn_profile_count < (
            self._qwen35_linear_attn_profile_limit
        )

    def _record_qwen35_linear_attn_profile(
        self,
        hidden_states: torch.Tensor,
        events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
    ) -> None:
        torch.cuda.synchronize(hidden_states.device)
        timings = {name: start.elapsed_time(end) for name, start, end in events}
        total = sum(timings.values())
        logger.warning(
            "QWEN35_LINEAR_ATTN_PROFILE prefix=%s tokens=%d proj=%.4fms "
            "core=%.4fms out=%.4fms total=%.4fms split=%s grouped=%s "
            "combined_decode=%s",
            self.prefix,
            hidden_states.shape[0],
            timings.get("proj", 0.0),
            timings.get("core", 0.0),
            timings.get("out", 0.0),
            total,
            self.split_projections,
            self.use_grouped_gguf_mmvq,
            self.enable_combined_packed_decode,
        )
        self._qwen35_linear_attn_profile_count += 1

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.num_v_heads // self.num_k_heads) * self.head_v_dim
                + (self.num_v_heads // self.num_k_heads) * self.head_v_dim
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads) * self.head_v_dim,
            (self.num_v_heads // self.num_k_heads) * self.head_v_dim,
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        num_tokens = hidden_states.size(0)
        profile_linear_attn = self._use_qwen35_linear_attn_profile(hidden_states)
        profile_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

        def timed(name: str, fn: Callable[[], object]) -> object:
            if not profile_linear_attn:
                return fn()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            profile_events.append((name, start, end))
            return result

        def run_projections() -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            if self.split_projections:
                grouped_mmvq = None
                if self.use_grouped_gguf_mmvq and self.use_grouped_gguf_ba_mmvq:
                    grouped_mmvq = try_grouped_gguf_linear_mmvq(
                        hidden_states,
                        [
                            self.in_proj_qkv,
                            self.in_proj_z,
                            self.in_proj_b,
                            self.in_proj_a,
                        ],
                    )
                if grouped_mmvq is None:
                    grouped_mmvq_qkvz = None
                    if self.use_grouped_gguf_mmvq:
                        grouped_mmvq_qkvz = try_grouped_gguf_linear_mmvq(
                            hidden_states,
                            [self.in_proj_qkv, self.in_proj_z],
                        )
                    if grouped_mmvq_qkvz is None:
                        projected_states_qkv, _ = self.in_proj_qkv(hidden_states)
                        z, _ = self.in_proj_z(hidden_states)
                    else:
                        projected_states_qkv, z = grouped_mmvq_qkvz
                    b, _ = self.in_proj_b(hidden_states)
                    a, _ = self.in_proj_a(hidden_states)
                else:
                    projected_states_qkv, z, b, a = grouped_mmvq

                q_size = self.key_dim // self.tp_size
                k_size = self.key_dim // self.tp_size
                v_size = self.value_dim // self.tp_size
                q, k, v = torch.split(
                    projected_states_qkv,
                    [q_size, k_size, v_size],
                    dim=-1,
                )

                mixed_qkv = projected_states_qkv
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b = b.contiguous()
                a = a.contiguous()
                return mixed_qkv, q, k, v, z, b, a

            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            q, k, v, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba
            )
            q, k, v = map(
                lambda x: rearrange(x, "l p d -> l (p d)"),
                (q, k, v),
            )
            mixed_qkv = torch.cat((q, k, v), dim=-1)
            return mixed_qkv, q, k, v, z, b, a

        mixed_qkv, q, k, v, z, b, a = timed("proj", run_projections)
        self._maybe_log_debug_stats(
            hidden_states=hidden_states,
            mixed_qkv=mixed_qkv,
            q=q,
            k=k,
            v=v,
            z=z,
            b=b,
            a=a,
        )

        b = b.contiguous()
        a = a.contiguous()
        core_attn_out_shape = (
            num_tokens,
            self.num_v_heads // self.tp_size,
            self.head_v_dim,
        )
        core_attn_out_factory = torch.zeros
        if (
            self.use_empty_core_attn_out_for_single_token
            and num_tokens == 1
        ) or self._packed_decode_writes_full_core_output(num_tokens):
            core_attn_out_factory = torch.empty
        core_attn_out = core_attn_out_factory(
            core_attn_out_shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        timed(
            "core",
            lambda: torch.ops.vllm.gdn_attention_core(
                mixed_qkv,
                b if self.call_b_first else a,
                a if self.call_b_first else b,
                core_attn_out,
                self.prefix,
            ),
        )
        self._maybe_log_debug_stats(core_attn_out=core_attn_out)

        def run_out_proj() -> torch.Tensor:
            projected_out = try_gguf_rms_norm_gated_out_proj_mmvq(
                core_attn_out,
                self.norm.weight,
                z,
                self.out_proj,
                self.norm.eps,
                self.norm.norm_before_gate,
            )
            if projected_out is not None:
                return projected_out
            z_shape_og = z.shape
            local_core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            local_z = z.reshape(-1, z.shape[-1])
            local_core_attn_out = self.norm(local_core_attn_out, local_z)
            self._maybe_log_debug_stats(post_norm=local_core_attn_out)
            local_core_attn_out = local_core_attn_out.reshape(z_shape_og)
            local_core_attn_out = rearrange(
                local_core_attn_out, "... h d -> ... (h d)"
            )
            projected_out, _ = self.out_proj(local_core_attn_out)
            return projected_out

        projected_out = timed("out", run_out_proj)
        output[:num_tokens] = projected_out
        if profile_linear_attn:
            self._record_qwen35_linear_attn_profile(hidden_states, profile_events)

    def _use_gfx906_chunk_decode_path(self) -> bool:
        raw = os.getenv("VLLM_QWEN35_CHUNK_DECODE")
        if raw is not None:
            return raw.lower() in {"1", "true", "yes", "on"}
        capability = current_platform.get_device_capability()
        return (
            self.split_projections
            and current_platform.is_rocm()
            and capability is not None
            and capability.major == 9
            and capability.minor == 0
        )

    def _packed_decode_writes_full_core_output(self, num_tokens: int) -> bool:
        try:
            if torch._dynamo.is_compiling():
                return False
        except AttributeError:
            pass
        if not (
            self.enable_packed_recurrent_decode and self._qwen35_is_gfx906_rocm
        ):
            return False
        try:
            attn_metadata = get_forward_context().attn_metadata
        except Exception:
            return False
        if not isinstance(attn_metadata, dict) or self.prefix not in attn_metadata:
            return False
        gdn_metadata = attn_metadata[self.prefix]
        if not isinstance(gdn_metadata, GDNAttentionMetadata):
            return False
        return (
            gdn_metadata.spec_sequence_masks is None
            and gdn_metadata.num_prefills == 0
            and gdn_metadata.num_decodes > 0
            and gdn_metadata.num_actual_tokens == num_tokens
        )


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            if vllm_config.model_config.quantization in {
                "gguf",
                "compressed-tensors",
            }:
                self.linear_attn = Qwen3_5GatedDeltaNet(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    speculative_config=speculative_config,
                    split_projections=True,
                    prefix=f"{prefix}.linear_attn",
                )
            else:
                self.linear_attn = GatedDeltaNetAttention(
                    config=config,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.linear_attn",
                    gqa_interleaved_layout=False,
                    create_in_proj_qkvz=vllm_config.lora_config is None,
                )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

        self._qwen35_profile_count = 0
        self._qwen35_layer_profile_enabled = _qwen35_env_flag(
            "VLLM_QWEN35_LAYER_PROFILE"
        )
        self._qwen35_layer_profile_max_tokens = _qwen35_env_optional_int(
            "VLLM_QWEN35_LAYER_PROFILE_MAX_TOKENS"
        )
        self._qwen35_layer_profile_limit = _qwen35_env_int(
            "VLLM_QWEN35_LAYER_PROFILE_LIMIT", 2
        )

    def _use_qwen35_layer_profile(self, hidden_states: torch.Tensor) -> bool:
        if not self._qwen35_layer_profile_enabled:
            return False
        try:
            if torch._dynamo.is_compiling():
                return False
        except AttributeError:
            pass
        if not hidden_states.is_cuda:
            return False
        max_tokens = self._qwen35_layer_profile_max_tokens
        if max_tokens is not None and hidden_states.shape[0] > max_tokens:
            return False
        return self._qwen35_profile_count < self._qwen35_layer_profile_limit

    def _record_qwen35_profile(
        self,
        hidden_states: torch.Tensor,
        events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
    ) -> None:
        torch.cuda.synchronize(hidden_states.device)
        timings = {name: start.elapsed_time(end) for name, start, end in events}
        total = sum(timings.values())
        logger.warning(
            "QWEN35_LAYER_PROFILE layer=%s type=%s tokens=%d norm1=%.4fms "
            "attn=%.4fms norm2=%.4fms mlp=%.4fms total=%.4fms",
            self.layer_idx,
            self.layer_type,
            hidden_states.shape[0],
            timings.get("norm1", 0.0),
            timings.get("attn", 0.0),
            timings.get("norm2", 0.0),
            timings.get("mlp", 0.0),
            total,
        )
        self._qwen35_profile_count += 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if not self._use_qwen35_layer_profile(hidden_states):
            return super().forward(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                **kwargs,
            )

        events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

        def timed(name: str, fn: Callable[[], object]) -> object:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            events.append((name, start, end))
            return result

        def run_norm1():
            nonlocal hidden_states, residual
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual
                )

        timed("norm1", run_norm1)

        self_attention_output = torch.empty_like(hidden_states)

        def run_attn():
            if self.layer_type == "linear_attention":
                self.linear_attn(
                    hidden_states=hidden_states,
                    output=self_attention_output,
                )
            elif self.layer_type == "full_attention":
                self.self_attn(
                    hidden_states=hidden_states,
                    output=self_attention_output,
                    positions=positions,
                )
            else:
                raise ValueError("Invalid layer_type")

        timed("attn", run_attn)
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        def run_norm2():
            nonlocal hidden_states, residual
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        timed("norm2", run_norm2)

        def run_mlp():
            nonlocal hidden_states
            hidden_states = self.mlp(hidden_states)

        timed("mlp", run_mlp)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        self._record_qwen35_profile(hidden_states, events)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.enable_lora = vllm_config.lora_config is not None
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        first_linear_attn = None
        for module_name, module in self.named_modules():
            if module_name.endswith("layers.0.linear_attn"):
                first_linear_attn = module
                break

        if first_linear_attn is not None:
            _append_qwen35_load_debug(
                "RUNTIME "
                f"linear_attn_type={type(first_linear_attn).__name__} "
                f"has_in_proj_b={hasattr(first_linear_attn, 'in_proj_b')} "
                f"has_in_proj_a={hasattr(first_linear_attn, 'in_proj_a')} "
                f"has_in_proj_ba={hasattr(first_linear_attn, 'in_proj_ba')} "
                f"split_attr={getattr(first_linear_attn, 'split_projections', None)}"
            )

        force_split = bool(
            getattr(first_linear_attn, "split_projections", False)
            if first_linear_attn is not None
            else False
        )

        has_split_qkv = bool(
            first_linear_attn is not None
            and hasattr(first_linear_attn, "in_proj_qkv")
        )
        has_split_ba = bool(
            first_linear_attn is not None
            and hasattr(first_linear_attn, "in_proj_b")
            and hasattr(first_linear_attn, "in_proj_a")
        )
        has_fused_ba = bool(
            first_linear_attn is not None
            and hasattr(first_linear_attn, "in_proj_ba")
            and first_linear_attn.in_proj_ba is not None
        )
        force_split_ba = os.getenv("VLLM_QWEN35_FORCE_BA_SPLIT", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if force_split_ba:
            has_split_ba = True
            has_fused_ba = False
        use_split_gdn_proj = has_split_qkv

        _append_qwen35_load_debug(
            f"MODE force_split={force_split} has_split_qkv={has_split_qkv} has_split_ba={has_split_ba} has_fused_ba={has_fused_ba}"
        )

        if has_fused_ba and not has_split_ba and not force_split:
            reverse_ba = os.getenv("VLLM_QWEN35_REVERSE_BA_SHARDS", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if reverse_ba:
                stacked_params_mapping.extend(
                    [
                        ("in_proj_ba", "in_proj_a", 0),
                        ("in_proj_ba", "in_proj_b", 1),
                    ]
                )
            else:
                stacked_params_mapping.extend(
                    [
                        ("in_proj_ba", "in_proj_b", 0),
                        ("in_proj_ba", "in_proj_a", 1),
                    ]
                )

        if not has_split_qkv and not force_split:
            stacked_params_mapping.extend(
                [
                    ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
                    ("in_proj_qkvz", "in_proj_z", 3),
                ]
            )

        modules_dict = dict(self.named_modules())
        loaded_params: set[str] = set()
        # Track partition sizes for merged parameters to calculate correct offsets
        partition_sizes: dict[str, dict[int, int]] = {}
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )

        def get_split_sizes_for_param(
            param_name: str, shard_ids: tuple[int, ...]
        ) -> list[int] | None:
            module_name, _, _ = param_name.rpartition(".")
            if not module_name:
                return None

            module = modules_dict.get(module_name)
            output_sizes = getattr(module, "output_sizes", None)
            if output_sizes is None:
                return None

            max_shard_id = max(shard_ids)
            if max_shard_id >= len(output_sizes):
                return None

            return [output_sizes[int(shard_id)] for shard_id in shard_ids]

        def load_split_gguf_shards(
            param_name,
            param,
            loaded_weight,
            shard_ids,
            weight_loader,
        ):
            split_sizes = get_split_sizes_for_param(param_name, shard_ids)

            if getattr(param, "is_gguf_weight_type", False):
                for sid in shard_ids:
                    param.shard_weight_type[sid] = loaded_weight.item()
                return

            if getattr(param, "is_gguf_weight", False):
                output_dim = getattr(param, "output_dim", 0)
                if split_sizes is None:
                    shard_size = loaded_weight.size(output_dim) // len(shard_ids)
                    split_sizes = [shard_size] * len(shard_ids)
                assert sum(split_sizes) == loaded_weight.size(output_dim), (
                    f"Cannot split GGUF shard for {param_name}: expected logical sizes "
                    f"{split_sizes}, got shape {tuple(loaded_weight.shape)} over dim "
                    f"{output_dim}"
                )
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                split_offset = 0

                for sid, shard_size in zip(shard_ids, split_sizes, strict=True):
                    assert shard_size % tp_size == 0, (
                        f"GGUF shard size {shard_size} for {param_name} is not divisible "
                        f"by tensor parallel size {tp_size}"
                    )
                    local_shard = shard_size // tp_size
                    start_idx = tp_rank * local_shard
                    shard = loaded_weight.narrow(output_dim, split_offset, shard_size)
                    shard = shard.narrow(output_dim, start_idx, local_shard)
                    shard = shard.to(device=param.device)
                    param.shard_id.append(sid)
                    param.shard_id_map[sid] = len(param.data_container)
                    param.data_container.append(shard)
                    split_offset += shard_size
                return

            if loaded_weight.ndim == 0 or loaded_weight.numel() == 1:
                for sid in shard_ids:
                    weight_loader(param, loaded_weight, sid)
                return

            output_dim = getattr(param, "output_dim", 0)
            if split_sizes is None:
                shard_size = loaded_weight.size(output_dim) // len(shard_ids)
                split_sizes = [shard_size] * len(shard_ids)
            assert sum(split_sizes) == loaded_weight.size(output_dim), (
                f"Cannot split GGUF shard for {param_name}: expected logical sizes "
                f"{split_sizes}, got shape {tuple(loaded_weight.shape)} over dim "
                f"{output_dim}"
            )
            split_offset = 0
            tp_size = get_tensor_model_parallel_world_size()
            for sid, shard_size in zip(shard_ids, split_sizes, strict=True):
                shard = loaded_weight.narrow(output_dim, split_offset, shard_size)
                if hasattr(param, "load_merged_column_weight"):
                    param.load_merged_column_weight(
                        loaded_weight=shard,
                        shard_id=sid,
                        shard_offset=split_offset // tp_size,
                        shard_size=shard_size // tp_size,
                    )
                elif "loaded_shard_id" in signature(weight_loader).parameters:
                    weight_loader(param, shard, loaded_shard_id=sid)
                else:
                    weight_loader(param, shard, sid)
                split_offset += shard_size

        def load_single_gguf_shard(param, loaded_weight, shard_id):
            if getattr(param, "is_gguf_weight_type", False):
                param.shard_weight_type[shard_id] = loaded_weight.item()
                return

            if getattr(param, "is_gguf_weight", False):
                output_dim = getattr(param, "output_dim", 0)
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                local_shard = loaded_weight.size(output_dim) // tp_size
                start_idx = tp_rank * local_shard
                shard = loaded_weight.narrow(output_dim, start_idx, local_shard)
                shard = shard.to(device=param.device)
                param.shard_id.append(shard_id)
                param.shard_id_map[shard_id] = len(param.data_container)
                param.data_container.append(shard)
                return

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            if self.quant_config is not None and self.quant_config.get_name() == "gguf":
                module_name, _, param_leaf = name.rpartition(".")
                module = modules_dict.get(module_name)
                if param_leaf == "weight" and isinstance(module, Qwen3_5RMSNorm):
                    loaded_weight = loaded_weight - 1.0

                if name.endswith("A_log"):
                    loaded_weight = torch.log(-loaded_weight)

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if f".{weight_name}." not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                if "linear_attn.in_proj" in name and any(
                    token in name
                    for token in ("in_proj_qkv", "in_proj_z", "in_proj_ba", "in_proj_b", "in_proj_a")
                ):
                    _append_qwen35_load_debug(
                        "STACKED "
                        f"name={name} shard_id={shard_id} "
                        f"is_gguf_weight={getattr(param, 'is_gguf_weight', False)} "
                        f"is_gguf_weight_type={getattr(param, 'is_gguf_weight_type', False)} "
                        f"loaded_shape={tuple(loaded_weight.shape) if hasattr(loaded_weight, 'shape') else 'NA'}"
                    )
                if isinstance(shard_id, tuple):
                    if getattr(param, "is_gguf_weight", False) or getattr(
                        param, "is_gguf_weight_type", False
                    ):
                        load_split_gguf_shards(
                            name, param, loaded_weight, shard_id, weight_loader
                        )
                    else:
                        weight_loader(param, loaded_weight, shard_id)
                    break
                if shard_id is not None and (
                    getattr(param, "is_gguf_weight", False)
                    or getattr(param, "is_gguf_weight_type", False)
                ):
                    load_single_gguf_shard(param, loaded_weight, shard_id)
                    break
                # Check if weight_loader accepts shard_id parameter
                import inspect

                sig = inspect.signature(weight_loader)
                if len(sig.parameters) >= 3:
                    weight_loader(param, loaded_weight, shard_id)
                elif (
                    hasattr(param, "load_merged_column_weight") and shard_id is not None
                ):
                    # Handle merged column parallel parameters (e.g., gate_up_proj)
                    # Calculate shard_offset by accumulating previous partition sizes
                    if isinstance(shard_id, int):
                        # For gate_up_proj: shard_id 0 = gate, shard_id 1 = up
                        # Use the sharded size from checkpoint, accounting for TP
                        tp_size = get_tensor_model_parallel_world_size()
                        shard_size = loaded_weight.size(param.output_dim) // tp_size

                        # Track this partition's size for offset calculation
                        if name not in partition_sizes:
                            partition_sizes[name] = {}
                        partition_sizes[name][shard_id] = shard_size

                        # Calculate offset as sum of all previous partition sizes
                        shard_offset = sum(
                            size
                            for sid, size in partition_sizes[name].items()
                            if sid < shard_id
                        )

                        param.load_merged_column_weight(
                            loaded_weight,
                            shard_offset=shard_offset,
                            shard_size=shard_size,
                        )
                    elif hasattr(param, "load_qkv_weight"):
                        # For QKV parameters with string shard_id
                        num_heads = getattr(self.config, "num_attention_heads", None)
                        if num_heads:
                            # Calculate shard_size and offset for QKV
                            # Use the sharded size from checkpoint, accounting for TP
                            tp_size = get_tensor_model_parallel_world_size()
                            shard_size = loaded_weight.size(param.output_dim) // tp_size

                            # Track this partition's size for offset calculation
                            if name not in partition_sizes:
                                partition_sizes[name] = {}
                            # Convert string shard_id to integer for tracking
                            qkv_shard_map = {"q": 0, "k": 1, "v": 2}
                            shard_id_int = qkv_shard_map.get(shard_id, 0)
                            partition_sizes[name][shard_id_int] = shard_size

                            # Calculate offset as sum of all previous partition sizes
                            shard_offset = sum(
                                size
                                for sid, size in partition_sizes[name].items()
                                if sid < shard_id_int
                            )

                            param.load_qkv_weight(
                                loaded_weight,
                                shard_offset=shard_offset,
                                shard_size=shard_size,
                                shard_id=shard_id,
                                num_heads=num_heads,
                            )
                        else:
                            weight_loader(param, loaded_weight)
                    else:
                        try:
                            weight_loader(param, loaded_weight)
                        except AssertionError as exc:
                            raise AssertionError(
                                f"Failed loading {name}: param shape {tuple(param.shape)} "
                                f"vs weight shape {tuple(loaded_weight.shape)}"
                            ) from exc
                else:
                    try:
                        loaded_weight = _maybe_unsqueeze_shared_expert_gate(
                            name, param, loaded_weight
                        )
                        weight_loader(param, loaded_weight)
                    except AssertionError as exc:
                        raise AssertionError(
                            f"Failed loading {name}: param shape {tuple(param.shape)} "
                            f"vs weight shape {tuple(loaded_weight.shape)}"
                        ) from exc
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        if _is_qwen35_gguf_projection_aux_param(name):
                            _append_qwen35_load_debug(f"MISSING name={name}")
                            continue
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    if "linear_attn.in_proj" in name and any(
                        token in name
                        for token in ("in_proj_qkv", "in_proj_z", "in_proj_ba", "in_proj_b", "in_proj_a")
                    ):
                        _append_qwen35_load_debug(
                            "DIRECT "
                            f"name={name} "
                            f"is_gguf_weight={getattr(param, 'is_gguf_weight', False)} "
                            f"is_gguf_weight_type={getattr(param, 'is_gguf_weight_type', False)} "
                            f"loaded_shape={tuple(loaded_weight.shape) if hasattr(loaded_weight, 'shape') else 'NA'}"
                        )
                    if getattr(param, "is_gguf_weight", False) or getattr(
                        param, "is_gguf_weight_type", False
                    ):
                        if getattr(param, "is_gguf_weight_type", False):
                            param.weight_type = loaded_weight.item()
                        else:
                            output_dim = getattr(param, "output_dim", 0)
                            tp_size = get_tensor_model_parallel_world_size()
                            tp_rank = get_tensor_model_parallel_rank()
                            local_shard = loaded_weight.size(output_dim) // tp_size
                            start_idx = tp_rank * local_shard
                            loaded_weight = loaded_weight.narrow(
                                output_dim, start_idx, local_shard
                            )
                            loaded_weight = loaded_weight.to(device=param.device)
                            loaded_param = torch.nn.Parameter(
                                loaded_weight.contiguous(), requires_grad=False
                            )
                            for attr_name, attr_value in vars(param).items():
                                setattr(loaded_param, attr_name, attr_value)
                            module_name, _, param_leaf = name.rpartition(".")
                            module = modules_dict[module_name]
                            module.register_parameter(param_leaf, loaded_param)
                            params_dict[name] = loaded_param
                        loaded_params.add(name)
                        continue
                    try:
                        loaded_weight = _maybe_unsqueeze_shared_expert_gate(
                            name, param, loaded_weight
                        )
                        weight_loader(param, loaded_weight)
                    except AssertionError as exc:
                        raise AssertionError(
                            f"Failed loading {name}: param shape {tuple(param.shape)} "
                            f"vs weight shape {tuple(loaded_weight.shape)}"
                        ) from exc
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        capability = current_platform.get_device_capability()
        self.cast_logits_input_to_fp32 = (
            self.quant_config is not None
            and self.quant_config.get_name() == "gguf"
            and current_platform.is_rocm()
            and capability is not None
            and capability.major == 9
            and capability.minor == 0
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.cast_logits_input_to_fp32:
            hidden_states = hidden_states.float()
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    # Qwen3.5 does not support multimodal pruning (EVS).
    supports_multimodal_pruning = False

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        enable_multimodal = (
            multimodal_config is not None
            and not multimodal_config.language_model_only
            and any(
                multimodal_config.get_limit_per_prompt(modality) > 0
                for modality in ("image", "video")
            )
        )

        self.config = config
        self.multimodal_config = multimodal_config if enable_multimodal else None
        self.use_data_parallel = (
            enable_multimodal
            and multimodal_config.mm_encoder_tp_mode == "data"
        )
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        if enable_multimodal:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )
        else:
            self.visual = PPMissingLayer()

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def recompute_mrope_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3.5 does not support multimodal pruning (EVS). "
            "recompute_mrope_positions should never be called."
        )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list["MultiModalFeatureSpec"],
    ) -> tuple[torch.Tensor, int]:
        if self.multimodal_config is None:
            text_len = len(input_tokens)
            positions = torch.arange(text_len, dtype=torch.long)
            return positions.unsqueeze(0).expand(3, -1), 0
        return super().get_mrope_input_positions(input_tokens, mm_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        conv_dtype, temporal_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
        capability = current_platform.get_device_capability()
        if (
            current_platform.is_rocm()
            and capability is not None
            and capability.major == 9
            and capability.minor == 0
        ):
            if vllm_config.cache_config.mamba_cache_dtype != "auto":
                return (conv_dtype, temporal_dtype)
            if vllm_config.cache_config.mamba_ssm_cache_dtype != "auto":
                return (torch.float32, temporal_dtype)
            return (torch.float32, torch.float32)
        return (conv_dtype, temporal_dtype)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        mm_limits = (
            getattr(multimodal_config, "limit_per_prompt", {})
            if multimodal_config is not None
            else {}
        )
        text_only_by_limits = all(
            getattr(mm_limits.get(modality), "count", None) == 0
            for modality in ("image", "video")
        )
        enable_multimodal = (
            multimodal_config is not None
            and not multimodal_config.language_model_only
            and not text_only_by_limits
            and any(
                multimodal_config.get_limit_per_prompt(modality) > 0
                for modality in ("image", "video")
            )
        )

        self.config = config
        self.multimodal_config = multimodal_config if enable_multimodal else None
        self.use_data_parallel = (
            enable_multimodal
            and multimodal_config.mm_encoder_tp_mode == "data"
        )
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        if enable_multimodal:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )
        else:
            self.visual = PPMissingLayer()

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
