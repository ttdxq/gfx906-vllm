# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import os
from functools import lru_cache

import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from typing import Any

from vllm import envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import (
    causal_conv1d_recurrent_gated_delta_rule_packed_decode,
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

logger = init_logger(__name__)
ENABLE_QWEN35_FORCE_Z_ONES = os.getenv("VLLM_QWEN35_FORCE_Z_ONES", "0") == "1"
ENABLE_QWEN35_EMPTY_CORE_ATTN_OUT = os.getenv(
    "VLLM_QWEN35_EMPTY_CORE_ATTN_OUT", "0"
).lower() in {"1", "true", "yes", "on"}


def _gdn_runtime_debug_enabled() -> bool:
    return bool(os.environ.get("VLLM_QWEN35_RUNTIME_DEBUG_FILE"))


def _append_gdn_runtime_debug(message: str) -> None:
    debug_file = os.environ.get("VLLM_QWEN35_RUNTIME_DEBUG_FILE")
    if not debug_file:
        return
    logger.warning(message)
    try:
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


@lru_cache(maxsize=1)
def _is_gfx906_rocm() -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_rocm()
        and capability is not None
        and capability.major == 9
        and capability.minor == 0
    )


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()
    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        backend_cfg = get_current_vllm_config().additional_config.get(
            "gdn_prefill_backend", "auto"
        )
        backend = str(backend_cfg).strip().lower()

        supports_flashinfer = (
            current_platform.is_cuda() and current_platform.is_device_capability(90)
        )

        if backend == "flashinfer":
            use_flashinfer = supports_flashinfer
            if not use_flashinfer:
                logger.warning_once(
                    "GDN prefill backend 'flashinfer' is selected but "
                    "cannot use this kernel on the current platform. "
                    "Falling back to Triton/FLA."
                )
        elif backend == "triton":
            use_flashinfer = False
        else:
            use_flashinfer = supports_flashinfer

        if use_flashinfer:
            logger.info_once("Using FlashInfer GDN prefill kernel", scope="local")
            logger.info_once(
                "FlashInfer GDN prefill kernel is JIT-compiled; first run may "
                "take a while to compile. Set `--gdn-prefill-backend triton` to "
                "avoid JIT compile time.",
                scope="local",
            )
        else:
            logger.info_once("Using Triton/FLA GDN prefill kernel", scope="local")

        self._forward_method = (
            self.forward_cuda if use_flashinfer else self.forward_native
        )

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        head_first: bool = False,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        head_first: bool = False,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )


class GatedDeltaNetAttention(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        create_in_proj_qkvz: bool = True,
        gqa_interleaved_layout=False,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix
        self.config = config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )
        self.gqa_interleaved_layout = gqa_interleaved_layout

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        if create_in_proj_qkvz:
            self.in_proj_qkvz = self.create_qkvz_proj(
                hidden_size=self.hidden_size,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
            )
        else:
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
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [query_key_settings, query_key_settings, value_settings],
            self.tp_size,
            self.tp_rank,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads // self.tp_size))
        self.A_log = nn.Parameter(
            torch.empty(divide(self.num_v_heads, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        packed_decode_env = os.environ.get("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE")
        if packed_decode_env is None:
            self.enable_packed_recurrent_decode = getattr(
                envs, "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", _is_gfx906_rocm()
            )
        else:
            self.enable_packed_recurrent_decode = packed_decode_env.lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        combined_decode_env = os.environ.get("VLLM_QWEN35_COMBINED_PACKED_DECODE")
        if combined_decode_env is None:
            self.enable_combined_packed_decode = (
                self.enable_packed_recurrent_decode and _is_gfx906_rocm()
            )
        else:
            self.enable_combined_packed_decode = combined_decode_env.lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

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
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
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
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def _can_use_empty_core_attn_out(self, num_tokens: int) -> bool:
        if not ENABLE_QWEN35_EMPTY_CORE_ATTN_OUT:
            return False
        try:
            attn_metadata = get_forward_context().attn_metadata
        except Exception:
            return False
        if not isinstance(attn_metadata, dict):
            return False
        layer_metadata = attn_metadata.get(self.prefix)
        if not isinstance(layer_metadata, GDNAttentionMetadata):
            return False
        return layer_metadata.num_actual_tokens == num_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        num_tokens = hidden_states.size(0)
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            if self.gqa_interleaved_layout:
                mixed_qkv_grouped = mixed_qkv.view(
                    mixed_qkv.size(0),
                    self.num_k_heads // self.tp_size,
                    self.head_k_dim
                    + self.head_k_dim
                    + (self.num_v_heads // self.num_k_heads) * self.head_v_dim,
                )
                z_grouped = z.view(
                    z.size(0),
                    self.num_k_heads // self.tp_size,
                    (self.num_v_heads // self.num_k_heads) * self.head_v_dim,
                )
                mixed_qkvz_grouped = torch.cat([mixed_qkv_grouped, z_grouped], dim=-1)
                ba_grouped = ba.view(
                    ba.size(0),
                    self.num_k_heads // self.tp_size,
                    2 * self.num_v_heads // self.num_k_heads,
                )
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz_grouped.reshape(mixed_qkvz_grouped.size(0), -1),
                    ba_grouped.reshape(ba_grouped.size(0), -1),
                )
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
                )
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba
                )
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
                )
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        core_attn_shape = (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim)
        if self._can_use_empty_core_attn_out(num_tokens):
            core_attn_out = torch.empty(
                core_attn_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            core_attn_out = torch.zeros(
                core_attn_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )

        if ENABLE_QWEN35_FORCE_Z_ONES:
            z = torch.ones_like(z)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
        if hasattr(self, "_prefill_kernels_warmed_up"):
            return
        self._prefill_kernels_warmed_up = True

        device = mixed_qkv.device
        dtype = mixed_qkv.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        for T in (16, 32, 64):
            q = torch.randn(1, T, num_k_heads, self.head_k_dim, device=device, dtype=dtype)
            k = torch.randn(1, T, num_k_heads, self.head_k_dim, device=device, dtype=dtype)
            v = torch.randn(1, T, num_v_heads, self.head_v_dim, device=device, dtype=dtype)
            dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
            dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
            g, beta = fused_gdn_gating(self.A_log, dummy_a, dummy_b, self.dt_bias)
            state = torch.zeros(
                1,
                num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=device,
                dtype=state_dtype,
            )
            cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

            try:
                self.chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )
            except Exception:
                logger.warning(
                    "GDN prefill kernel warmup (T=%d) failed for layer %s. First inference may OOM due to autotuner.",
                    T,
                    self.prefix,
                    exc_info=True,
                )
            finally:
                del q, k, v, dummy_a, dummy_b, g, beta, state, cu_seqlens

        torch.accelerator.empty_cache()

    def _log_projection_debug_once(self) -> None:
        if getattr(self, "_logged_projection_debug", False):
            return
        self._logged_projection_debug = True

        def summarize_proj(name: str, module: object) -> str:
            qweight = getattr(module, "qweight", None)
            qtype = getattr(module, "qweight_type", None)
            return (
                f"{name}: "
                f"has_qweight={qweight is not None} "
                f"shard_id={getattr(qweight, 'shard_id', None)} "
                f"has_offset_map={hasattr(qweight, 'shard_offset_map')} "
                f"container_len={len(getattr(qweight, 'data_container', [])) if qweight is not None else 'NA'} "
                f"qtype_weight={getattr(qtype, 'weight_type', None)} "
                f"qtype_shards={getattr(qtype, 'shard_weight_type', None)}"
            )

        if _gdn_runtime_debug_enabled():
            _append_gdn_runtime_debug(
                "PROJMETA "
                + " | ".join(
                    [
                        f"prefix={self.prefix}",
                        summarize_proj("in_proj_qkv", getattr(self, "in_proj_qkv", None)),
                        summarize_proj("in_proj_z", getattr(self, "in_proj_z", None)),
                        summarize_proj("in_proj_ba", getattr(self, "in_proj_ba", None)),
                    ]
                )
            )

    def _forward_core(self, mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor, core_attn_out: torch.Tensor):
        self._log_projection_debug_once()
        forward_context = get_forward_context()
        attn_metadata: Any = forward_context.attn_metadata

        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv)
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if _gdn_runtime_debug_enabled():
            _append_gdn_runtime_debug(
                "COREMETA "
                f"prefix={self.prefix} packed={self.enable_packed_recurrent_decode} "
                f"combined={self.enable_combined_packed_decode} "
                f"spec_is_none={attn_metadata.spec_sequence_masks is None} "
                f"num_prefills={attn_metadata.num_prefills} num_decodes={attn_metadata.num_decodes}"
            )

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        self_kv_cache = self.kv_cache
        if isinstance(self_kv_cache, list):
            self_kv_cache = self_kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                a_spec = a
                b_spec = b
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][: attn_metadata.num_spec_decodes],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens
        if split_non_spec:
            assert mixed_qkv_non_spec is not None
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec[:num_decode_tokens]
            )
            query_non_spec = key_non_spec = value_non_spec = None
        else:
            query_decode = key_decode = value_decode = None
            if attn_metadata.num_prefills == 0:
                query_non_spec, key_non_spec, value_non_spec = (
                    self.rearrange_mixed_qkv(mixed_qkv_non_spec)
                )
            else:
                query_non_spec = key_non_spec = value_non_spec = None

        g_non_spec: torch.Tensor | None = None
        beta_non_spec: torch.Tensor | None = None

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            if spec_sequence_masks is not None:
                mixed_qkv_prefill = mixed_qkv_non_spec
                a_prefill = a.index_select(0, non_spec_token_indx)
                b_prefill = b.index_select(0, non_spec_token_indx)
            elif split_non_spec:
                mixed_qkv_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                a_prefill = a[num_decode_tokens:]
                b_prefill = b[num_decode_tokens:]
            else:
                mixed_qkv_prefill = mixed_qkv_non_spec
                a_prefill = a
                b_prefill = b

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = fused_post_conv_prep(
                conv_output=mixed_qkv_prefill,
                a=a_prefill,
                b=b_prefill,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)

        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a_spec,
                b=b_spec,
                dt_bias=self.dt_bias,
                q=query_spec,
                k=key_spec,
                v=value_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        if split_non_spec:
            assert query_decode is not None
            assert key_decode is not None
            assert value_decode is not None
            core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a[:num_decode_tokens],
                b=b[:num_decode_tokens],
                dt_bias=self.dt_bias,
                q=query_decode,
                k=key_decode,
                v=value_decode,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_decodes
                ],
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_decode = None

        if attn_metadata.num_prefills > 0:
            prefill_query_start_loc = attn_metadata.prefill_query_start_loc
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_query_start_loc is not None
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            if _is_gfx906_rocm():
                if spec_sequence_masks is not None:
                    a_non_spec = a.index_select(0, non_spec_token_indx)
                    b_non_spec = b.index_select(0, non_spec_token_indx)
                elif split_non_spec:
                    a_non_spec = a[num_decode_tokens:]
                    b_non_spec = b[num_decode_tokens:]
                else:
                    a_non_spec = a
                    b_non_spec = b
                initial_state = (
                    ssm_state[prefill_state_indices]
                    .transpose(-1, -2)
                    .contiguous()
                )
                initial_state[~prefill_has_initial_state, ...] = 0
                if _gdn_runtime_debug_enabled():
                    _append_gdn_runtime_debug(
                        "PREFILL "
                        f"q={tuple(query_non_spec.shape)} k={tuple(key_non_spec.shape)} "
                        f"v={tuple(value_non_spec.shape)} a={tuple(a_non_spec.shape)} b={tuple(b_non_spec.shape)}"
                    )
                core_attn_out_non_spec, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a_non_spec,
                    b=b_non_spec,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=initial_state,
                    inplace_final_state=True,
                    cu_seqlens=prefill_query_start_loc,
                    ssm_state_indices=None,
                    use_qk_l2norm_in_kernel=False,
                )
                ssm_state[prefill_state_indices] = last_recurrent_state.transpose(
                    -1, -2
                ).to(ssm_state.dtype)
            else:
                assert g_non_spec is not None
                assert beta_non_spec is not None
                initial_state = ssm_state[prefill_state_indices].contiguous()
                initial_state[~prefill_has_initial_state, ...] = 0
                if _gdn_runtime_debug_enabled():
                    _append_gdn_runtime_debug(
                        "PREFILL "
                        f"q={tuple(query_non_spec.shape)} k={tuple(key_non_spec.shape)} "
                        f"v={tuple(value_non_spec.shape)} g={tuple(g_non_spec.shape)} beta={tuple(beta_non_spec.shape)}"
                    )
                core_attn_out_non_spec, last_recurrent_state = self.chunk_gated_delta_rule(
                    q=query_non_spec.transpose(1, 2),
                    k=key_non_spec.transpose(1, 2),
                    v=value_non_spec.transpose(1, 2),
                    g=g_non_spec.transpose(1, 2),
                    beta=beta_non_spec.transpose(1, 2),
                    initial_state=initial_state.transpose(-1, -2).contiguous(),
                    output_final_state=True,
                    cu_seqlens=prefill_query_start_loc,
                    head_first=True,
                    use_qk_l2norm_in_kernel=False,
                )
                ssm_state[prefill_state_indices] = last_recurrent_state.to(
                    ssm_state.dtype
                )

            if core_attn_out_decode is not None:
                core_attn_out_non_spec = torch.cat(
                    (core_attn_out_decode, core_attn_out_non_spec), dim=1
                )
        elif attn_metadata.num_decodes > 0:
            if _is_gfx906_rocm():
                core_attn_out_non_spec, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_non_spec, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        forward_context = get_forward_context()
        self_kv_cache = self.kv_cache
        if isinstance(self_kv_cache, list):
            self_kv_cache = self_kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        state_indices = non_spec_state_indices_tensor[:num_actual_tokens]
        if self.enable_combined_packed_decode:
            try:
                if _gdn_runtime_debug_enabled():
                    _append_gdn_runtime_debug(
                        "DECODE_COMBINED_PACKED "
                        f"mixed_qkv={tuple(mixed_qkv.shape)} a={tuple(a.shape)} b={tuple(b.shape)} "
                        f"out={tuple(out_buf.shape)}"
                    )
                causal_conv1d_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=mixed_qkv,
                    conv_state=conv_state,
                    conv_weight=conv_weights,
                    conv_bias=self.conv1d.bias,
                    a=a,
                    b=b,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    scale=self.head_k_dim**-0.5,
                    initial_state=ssm_state,
                    out=out_buf,
                    ssm_state_indices=state_indices,
                    pad_slot_id=PAD_SLOT_ID,
                    silu_activation=self.activation in ("silu", "swish"),
                    use_qk_l2norm_in_kernel=True,
                )
                return
            except (AttributeError, RuntimeError) as exc:
                if _gdn_runtime_debug_enabled():
                    _append_gdn_runtime_debug(
                        "DECODE_COMBINED_PACKED_FALLBACK "
                        f"{type(exc).__name__}: {exc}"
                    )

        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices,
            validate_data=False,
        )

        if _gdn_runtime_debug_enabled():
            _append_gdn_runtime_debug(
                "DECODE_PACKED "
                f"mixed_qkv={tuple(mixed_qkv_non_spec.shape)} a={tuple(a.shape)} b={tuple(b.shape)} "
                f"out={tuple(out_buf.shape)}"
            )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    capability = current_platform.get_device_capability()
    if (
        current_platform.is_rocm()
        and capability is not None
        and capability.major == 9
        and capability.minor == 0
    ):
        x = a.float() + dt_bias.view(1, -1).float()
        g = -torch.exp(A_log.float()).view(1, -1) * torch.nn.functional.softplus(
            x, beta=beta, threshold=threshold
        )
        beta_output = torch.sigmoid(b.float()).to(b.dtype)
        return g.unsqueeze(0), beta_output.unsqueeze(0)

    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
