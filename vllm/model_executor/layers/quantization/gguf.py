# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable, Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Optional

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


ENABLE_GGUF_SHARED_EXPERT_FUSED_ACT_DOWN = _env_flag(
    "VLLM_GGUF_SHARED_EXPERT_FUSED_ACT_DOWN", True
)
ENABLE_GGUF_RMS_GATED_OUT_PROJ_MMVQ = _env_flag(
    "VLLM_GGUF_RMS_GATED_OUT_PROJ_MMVQ", True
)
ENABLE_GGUF_SIGMOID_GATED_OUT_PROJ_MMVQ = _env_flag(
    "VLLM_GGUF_SIGMOID_GATED_OUT_PROJ_MMVQ", True
)
ENABLE_GGUF_Q8_0_FAST_MMVQ = _env_flag("VLLM_GGUF_Q8_0_FAST_MMVQ", False)
ENABLE_GGUF_Q8_1_BUFFER_CACHE = _env_flag("VLLM_GGUF_Q8_1_BUFFER_CACHE", True)
ENABLE_GGUF_SINGLE_MMVQ_Q8_CACHE = _env_flag(
    "VLLM_GGUF_SINGLE_MMVQ_Q8_CACHE", False
)
ENABLE_GGUF_GROUPED_MMVQ_OUTPUT_CACHE = _env_flag(
    "VLLM_GGUF_GROUPED_MMVQ_OUTPUT_CACHE", False
)
ENABLE_GGUF_FUSED_MMVQ_OUTPUT_CACHE = _env_flag(
    "VLLM_GGUF_FUSED_MMVQ_OUTPUT_CACHE", True
)
ENABLE_GGUF_COALESCE_SAME_TYPE_SHARDS = _env_flag(
    "VLLM_GGUF_COALESCE_SAME_TYPE_SHARDS", False
)
ENABLE_GGUF_SHARDED_MMVQ_Q8_CACHE = _env_flag(
    "VLLM_GGUF_SHARDED_MMVQ_Q8_CACHE", True
)
ENABLE_GGUF_GROUPED_SAME_TYPE_MMVQ = _env_flag(
    "VLLM_GGUF_GROUPED_SAME_TYPE_MMVQ", True
)
ENABLE_GGUF_QKV3_MMVQ = _env_flag("VLLM_GGUF_QKV3_MMVQ", False)
ENABLE_GGUF_MOE_HOST_FUSED_ACT_W2 = _env_flag(
    "VLLM_GGUF_MOE_HOST_FUSED_ACT_W2", True
)
ENABLE_GGUF_GFX906_SMALL_BATCH_MMVQ = _env_flag(
    "VLLM_GGUF_GFX906_SMALL_BATCH_MMVQ", True
)
ENABLE_GGUF_LINEAR_PROFILE = _env_flag("VLLM_GGUF_LINEAR_PROFILE", False)
ENABLE_GGUF_FORCE_DEQUANT_MATMUL = _env_flag(
    "VLLM_GGUF_FORCE_DEQUANT_MATMUL", False
)
ENABLE_GGUF_REPACK_IQ4_XS_TO_Q8_0 = _env_flag(
    "VLLM_GGUF_REPACK_IQ4_XS_TO_Q8_0", True
)

_GGUF_Q8_1_BUFFER_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_GGUF_TENSOR_BUFFER_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_GGUF_MOE_PROFILE_COUNTS: dict[str, int] = {}
_GGUF_MOE_TIMING_COUNTS: dict[str, int] = {}
_GGUF_LINEAR_PROFILE_COUNTS: dict[tuple[Any, ...], int] = {}


@lru_cache(maxsize=1)
def _gguf_gfx906_small_batch_mmvq_enabled() -> bool:
    if not ENABLE_GGUF_GFX906_SMALL_BATCH_MMVQ:
        return False
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_gfx906

        return on_gfx906()
    except Exception:
        return False


@lru_cache(maxsize=1)
def _gguf_gfx906_mmvq_max_batch() -> int:
    return max(2, _env_int("VLLM_GGUF_GFX906_MMVQ_MAX_BATCH", 8))


@lru_cache(maxsize=1)
def _gguf_gfx906_q4_mmvq_max_batch() -> int:
    return max(2, _env_int("VLLM_GGUF_GFX906_Q4_MMVQ_MAX_BATCH", 16))


@lru_cache(maxsize=1)
def _gguf_gfx906_q6_mmvq_max_batch() -> int:
    return max(2, _env_int("VLLM_GGUF_GFX906_Q6_MMVQ_MAX_BATCH", 16))


@lru_cache(maxsize=1)
def _gguf_repack_iq4_xs_to_q8_0_enabled() -> bool:
    if not ENABLE_GGUF_REPACK_IQ4_XS_TO_Q8_0:
        return False
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_gfx906

        return on_gfx906()
    except Exception:
        return False


@lru_cache(maxsize=1)
def _gguf_moe_mmq_min_tokens() -> int:
    return max(1, _env_int("VLLM_GGUF_MOE_MMQ_MIN_TOKENS", 5))


@lru_cache(maxsize=1)
def _gguf_moe_runtime_config() -> tuple[bool, bool, int, bool, int]:
    use_weighted_sum = _env_flag("VLLM_GGUF_MOE_WEIGHTED_SUM", True)
    use_fused_act_quant = _env_flag("VLLM_GGUF_MOE_FUSED_ACT_QUANT", True)
    try:
        fused_act_quant_max_tokens = int(
            os.getenv("VLLM_GGUF_MOE_FUSED_ACT_QUANT_MAX_TOKENS", "4")
        )
    except ValueError:
        fused_act_quant_max_tokens = 4
    use_fused_w2_weighted_sum = _env_flag("VLLM_GGUF_MOE_FUSED_W2_WEIGHTED_SUM", True)
    try:
        fused_w2_weighted_sum_max_topk = int(
            os.getenv("VLLM_GGUF_MOE_FUSED_W2_WEIGHTED_SUM_MAX_TOPK", "8")
        )
    except ValueError:
        fused_w2_weighted_sum_max_topk = 8
    return (
        use_weighted_sum,
        use_fused_act_quant,
        fused_act_quant_max_tokens,
        use_fused_w2_weighted_sum,
        fused_w2_weighted_sum_max_topk,
    )


@lru_cache(maxsize=1)
def _gguf_moe_ops_available() -> tuple[bool, bool, bool]:
    return (
        hasattr(torch.ops._C, "ggml_silu_and_mul_quantize_row_q8_1")
        and hasattr(torch.ops._C, "ggml_moe_q8_vec"),
        hasattr(torch.ops._C, "ggml_moe_q8_vec_weighted_sum"),
        hasattr(torch.ops._C, "ggml_moe_a8_vec_silu_q8_weighted_sum_out"),
    )


@lru_cache(maxsize=1)
def _gguf_moe_profile_enabled() -> bool:
    return _env_flag("VLLM_GGUF_MOE_PROFILE", False)


def _gguf_moe_profile_hit(
    path: str,
    x: torch.Tensor,
    top_k: int,
    qweight_type: int,
    qweight_type2: int,
) -> None:
    if not _gguf_moe_profile_enabled():
        return
    count = _GGUF_MOE_PROFILE_COUNTS.get(path, 0) + 1
    _GGUF_MOE_PROFILE_COUNTS[path] = count
    if count <= 8 or count in {16, 32, 64, 128, 256, 512, 1024}:
        logger.warning(
            "GGUF_MOE_PROFILE path=%s count=%d tokens=%d hidden=%d top_k=%d "
            "w1_type=%s w2_type=%s dtype=%s",
            path,
            count,
            x.shape[0],
            x.shape[1],
            top_k,
            qweight_type,
            qweight_type2,
            x.dtype,
        )


@lru_cache(maxsize=1)
def _gguf_moe_timing_enabled() -> bool:
    return _env_flag("VLLM_GGUF_MOE_TIMING", False)


@lru_cache(maxsize=1)
def _gguf_linear_profile_enabled() -> bool:
    return ENABLE_GGUF_LINEAR_PROFILE


def _gguf_quant_name(qweight_type: int) -> str:
    try:
        return WeightType(qweight_type).name
    except ValueError:
        return str(qweight_type)


def _gguf_linear_profile_hit(
    path: str,
    x: torch.Tensor,
    qweights: list[torch.Tensor],
    qweight_types: list[int],
    *,
    grouped_same_type: bool | None = None,
) -> None:
    if not _gguf_linear_profile_enabled():
        return
    try:
        max_tokens = int(os.getenv("VLLM_GGUF_LINEAR_PROFILE_MAX_TOKENS", "4"))
    except ValueError:
        max_tokens = 4
    if x.shape[0] > max_tokens:
        return

    rows = tuple(int(qweight.shape[0]) for qweight in qweights)
    types = tuple(_gguf_quant_name(int(qweight_type)) for qweight_type in qweight_types)
    key = (
        path,
        types,
        rows,
        int(x.shape[0]),
        int(x.shape[1]),
        str(x.dtype),
        grouped_same_type,
    )
    count = _GGUF_LINEAR_PROFILE_COUNTS.get(key, 0) + 1
    _GGUF_LINEAR_PROFILE_COUNTS[key] = count
    try:
        limit = int(os.getenv("VLLM_GGUF_LINEAR_PROFILE_LIMIT", "24"))
    except ValueError:
        limit = 24
    if count <= limit or count in {32, 64, 128, 256, 512, 1024, 2048}:
        logger.warning(
            "GGUF_LINEAR_PROFILE path=%s count=%d tokens=%d hidden=%d "
            "types=%s rows=%s same_type=%s dtype=%s",
            path,
            count,
            x.shape[0],
            x.shape[1],
            ",".join(types),
            ",".join(str(row) for row in rows),
            grouped_same_type,
            x.dtype,
        )


class _GGUFMoeTimer:

    def __init__(
        self,
        path: str,
        x: torch.Tensor,
        top_k: int,
        qweight_type: int,
        qweight_type2: int,
    ) -> None:
        self.path = path
        self.x = x
        self.top_k = top_k
        self.qweight_type = qweight_type
        self.qweight_type2 = qweight_type2
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.enabled = self._should_enable(x)

    @staticmethod
    def _should_enable(x: torch.Tensor) -> bool:
        if not _gguf_moe_timing_enabled() or not x.is_cuda:
            return False
        try:
            if torch._dynamo.is_compiling():
                return False
        except AttributeError:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return False
        except RuntimeError:
            return False
        return True

    def record(self, name: str, fn: Callable[[], Any]) -> Any:
        if not self.enabled:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        self.events.append((name, start, end))
        return result

    def report(self) -> None:
        if not self.enabled or not self.events:
            return
        torch.cuda.synchronize(self.x.device)
        count = _GGUF_MOE_TIMING_COUNTS.get(self.path, 0) + 1
        _GGUF_MOE_TIMING_COUNTS[self.path] = count
        try:
            limit = int(os.getenv("VLLM_GGUF_MOE_TIMING_LIMIT", "16"))
        except ValueError:
            limit = 16
        if count > limit and count not in {32, 64, 128, 256, 512, 1024}:
            return
        timings = [(name, start.elapsed_time(end)) for name, start, end in self.events]
        total = sum(value for _, value in timings)
        timing_text = " ".join(f"{name}={value:.4f}ms" for name, value in timings)
        logger.warning(
            "GGUF_MOE_TIMING path=%s count=%d tokens=%d hidden=%d top_k=%d "
            "w1_type=%s w2_type=%s total=%.4fms %s",
            self.path,
            count,
            self.x.shape[0],
            self.x.shape[1],
            self.top_k,
            self.qweight_type,
            self.qweight_type2,
            total,
            timing_text,
        )


def _gguf_q8_1_packed_cols(cols: int) -> int:
    return ((cols + 511) // 512 * 512) // 32 * 9


def _get_gguf_q8_1_buffer(
    owner: Any,
    cache_name: Any,
    rows: int,
    cols: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not ENABLE_GGUF_Q8_1_BUFFER_CACHE:
        return None
    shape = (rows, _gguf_q8_1_packed_cols(cols))
    if owner is None:
        key = (cache_name, device.type, device.index, *shape)
        cached = _GGUF_Q8_1_BUFFER_CACHE.get(key)
        if cached is None or cached.device != device:
            cached = torch.empty(shape, dtype=torch.int32, device=device)
            _GGUF_Q8_1_BUFFER_CACHE[key] = cached
        return cached

    cached = getattr(owner, cache_name, None)
    if (
        cached is None
        or cached.shape != shape
        or cached.device != device
        or cached.dtype != torch.int32
    ):
        cached = torch.empty(shape, dtype=torch.int32, device=device)
        setattr(owner, cache_name, cached)
    return cached


def _get_gguf_tensor_buffer(
    cache_name: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if not ENABLE_GGUF_Q8_1_BUFFER_CACHE:
        return None
    key = (cache_name, device.type, device.index, dtype, *shape)
    cached = _GGUF_TENSOR_BUFFER_CACHE.get(key)
    if cached is None or cached.device != device or cached.dtype != dtype:
        cached = torch.empty(shape, dtype=dtype, device=device)
        _GGUF_TENSOR_BUFFER_CACHE[key] = cached
    return cached


def _quantize_row_q8_1_cached(
    x: torch.Tensor,
    owner: Any,
    cache_name: Any,
) -> torch.Tensor:
    if hasattr(torch.ops._C, "ggml_quantize_row_q8_1_out"):
        quant_x = _get_gguf_q8_1_buffer(
            owner, cache_name, x.shape[0], x.shape[1], x.device
        )
        if quant_x is not None:
            ops.ggml_quantize_row_q8_1_out(x, quant_x)
            return quant_x
    return ops.ggml_quantize_row_q8_1(x)


def _silu_and_mul_quantize_row_q8_1_cached(
    x: torch.Tensor,
    owner: Any,
    cache_name: Any,
) -> torch.Tensor:
    if hasattr(torch.ops._C, "ggml_silu_and_mul_quantize_row_q8_1_out"):
        quant_x = _get_gguf_q8_1_buffer(
            owner, cache_name, x.shape[0], x.shape[1] // 2, x.device
        )
        if quant_x is not None:
            ops.ggml_silu_and_mul_quantize_row_q8_1_out(x, quant_x)
            return quant_x
    return ops.ggml_silu_and_mul_quantize_row_q8_1(x)


def _sigmoid_and_mul_quantize_row_q8_1_cached(
    x: torch.Tensor,
    gate: torch.Tensor,
    owner: Any,
    cache_name: Any,
) -> torch.Tensor:
    if hasattr(torch.ops._C, "ggml_sigmoid_and_mul_quantize_row_q8_1_out"):
        quant_x = _get_gguf_q8_1_buffer(
            owner, cache_name, x.shape[0], x.shape[1], x.device
        )
        if quant_x is not None:
            ops.ggml_sigmoid_and_mul_quantize_row_q8_1_out(x, gate, quant_x)
            return quant_x
    return ops.ggml_sigmoid_and_mul_quantize_row_q8_1(x, gate)


def _rms_norm_gated_quantize_row_q8_1_cached(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    epsilon: float,
    norm_before_gate: bool,
    owner: Any,
    cache_name: Any,
) -> torch.Tensor:
    if hasattr(torch.ops._C, "ggml_rms_norm_gated_quantize_row_q8_1_out"):
        quant_x = _get_gguf_q8_1_buffer(
            owner, cache_name, x.shape[0], x.shape[1] * x.shape[2], x.device
        )
        if quant_x is not None:
            ops.ggml_rms_norm_gated_quantize_row_q8_1_out(
                x, weight, gate, quant_x, epsilon, norm_before_gate
            )
            return quant_x
    return ops.ggml_rms_norm_gated_quantize_row_q8_1(
        x, weight, gate, epsilon, norm_before_gate
    )


def _gguf_moe_q8_vec_weighted_sum_cached(
    quant_out: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    qweight_type2: int,
    row: int,
    num_tokens: int,
    col: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if hasattr(torch.ops._C, "ggml_moe_q8_vec_weighted_sum_out"):
        output = _get_gguf_tensor_buffer(
            ("_gguf_moe_w2_weighted_sum", w2.data_ptr(), row, num_tokens),
            (num_tokens, row),
            dtype,
            w2.device,
        )
        if output is not None:
            ops.ggml_moe_q8_vec_weighted_sum_out(
                quant_out,
                w2,
                topk_ids,
                topk_weights,
                output,
                top_k,
                qweight_type2,
                row,
                num_tokens,
                col,
            )
            return output
    return ops.ggml_moe_q8_vec_weighted_sum(
        quant_out,
        w2,
        topk_ids,
        topk_weights,
        top_k,
        qweight_type2,
        row,
        num_tokens,
        col,
        dtype,
    )


def _gguf_moe_a8_vec_cached(
    x: torch.Tensor,
    w1: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    qweight_type: int,
    row: int,
    num_tokens: int,
) -> torch.Tensor:
    if hasattr(torch.ops._C, "ggml_moe_a8_vec_out"):
        output = _get_gguf_tensor_buffer(
            ("_gguf_moe_w1_out", w1.data_ptr(), row, num_tokens, top_k),
            (num_tokens * top_k, row),
            x.dtype,
            x.device,
        )
        quant_x = _get_gguf_q8_1_buffer(
            None,
            ("_gguf_moe_w1_quant_x", w1.data_ptr(), num_tokens, x.shape[1]),
            num_tokens,
            x.shape[1],
            x.device,
        )
        if output is not None and quant_x is not None:
            ops.ggml_moe_a8_vec_out(
                x,
                w1,
                topk_ids,
                output,
                quant_x,
                top_k,
                qweight_type,
                row,
                num_tokens,
            )
            return output
    return ops.ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, row, num_tokens)


def _gguf_moe_decode_fused_cached(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    qweight_type: int,
    qweight_type2: int,
    w1_row: int,
    w2_row: int,
    num_tokens: int,
) -> torch.Tensor | None:
    if not hasattr(torch.ops._C, "ggml_moe_a8_vec_silu_q8_weighted_sum_out"):
        return None
    w1_output = _get_gguf_tensor_buffer(
        ("_gguf_moe_host_fused_w1_out", w1.data_ptr(), w1_row, num_tokens, top_k),
        (num_tokens * top_k, w1_row),
        x.dtype,
        x.device,
    )
    quant_x = _get_gguf_q8_1_buffer(
        None,
        ("_gguf_moe_host_fused_quant_x", w1.data_ptr(), num_tokens, x.shape[1]),
        num_tokens,
        x.shape[1],
        x.device,
    )
    quant_w1_output = _get_gguf_q8_1_buffer(
        None,
        (
            "_gguf_moe_host_fused_quant_w1_out",
            w2.data_ptr(),
            num_tokens,
            top_k,
            w1_row,
        ),
        num_tokens * top_k,
        w1_row // 2,
        x.device,
    )
    output = _get_gguf_tensor_buffer(
        ("_gguf_moe_host_fused_output", w2.data_ptr(), w2_row, num_tokens),
        (num_tokens, w2_row),
        x.dtype,
        x.device,
    )
    if (
        w1_output is None
        or quant_x is None
        or quant_w1_output is None
        or output is None
    ):
        return None
    ops.ggml_moe_a8_vec_silu_q8_weighted_sum_out(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        w1_output,
        quant_x,
        quant_w1_output,
        output,
        top_k,
        qweight_type,
        qweight_type2,
        w1_row,
        w2_row,
        num_tokens,
    )
    return output


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__()
        self.unquantized_modules = unquantized_modules or []

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_name(self) -> QuantizationMethods:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedEmbeddingMethod()
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            return GGUFMoEMethod(self, layer.moe_config)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the vllm model structure

        :param hf_to_vllm_mapper: maps from hf model structure (the assumed
            structure of the qconfig) to vllm model structure
        """
        if self.unquantized_modules is not None:
            self.unquantized_modules = hf_to_vllm_mapper.apply_list(
                self.unquantized_modules
            )


def is_layer_skipped_gguf(
    prefix: str,
    unquantized_modules: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
):
    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    proj_name = prefix.split(".")[-1]
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(
                shard_prefix in module_name for module_name in unquantized_modules
            )

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = any(module_name in prefix for module_name in unquantized_modules)

    assert is_skipped is not None
    return is_skipped


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def _fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    mmvq_safe = _mmvq_safe_batch(qweight, qweight_type)
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        if ENABLE_GGUF_LINEAR_PROFILE:
            _gguf_linear_profile_hit(
                "single_unquantized", x, [qweight], [qweight_type]
            )
        return x @ qweight.T
    if ENABLE_GGUF_FORCE_DEQUANT_MATMUL:
        if ENABLE_GGUF_LINEAR_PROFILE:
            _gguf_linear_profile_hit(
                "single_forced_dequant", x, [qweight], [qweight_type]
            )
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        return x @ weight.T
    # enable MMVQ in contiguous batching with batch_size=1
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        if ENABLE_GGUF_LINEAR_PROFILE:
            _gguf_linear_profile_hit("single_mmvq", x, [qweight], [qweight_type])
        if (
            ENABLE_GGUF_Q8_0_FAST_MMVQ
            and qweight_type == WeightType.Q8_0
            and hasattr(torch.ops._C, "ggml_mul_mat_vec_q8_0_fast")
        ):
            quant_x = ops.ggml_quantize_row_q8_1(x)
            y = ops.ggml_mul_mat_vec_q8_0_fast(
                qweight, quant_x, qweight.shape[0], x.shape[1], x.dtype
            )
        else:
            y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # Use MMQ Kernel if it's available (standard + k-quants)
    elif qweight_type in MMQ_QUANT_TYPES:
        if ENABLE_GGUF_LINEAR_PROFILE:
            _gguf_linear_profile_hit("single_mmq", x, [qweight], [qweight_type])
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # If there is no available MMQ kernel, fallback to dequantize
    elif qweight_type in DEQUANT_TYPES:
        if ENABLE_GGUF_LINEAR_PROFILE:
            _gguf_linear_profile_hit("single_dequant", x, [qweight], [qweight_type])
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def _gguf_mmvq_from_q8_activation(
    qweight: torch.Tensor,
    quant_x: torch.Tensor,
    qweight_type: int,
    row: int,
    col: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        ENABLE_GGUF_Q8_0_FAST_MMVQ
        and qweight_type == WeightType.Q8_0
        and hasattr(torch.ops._C, "ggml_mul_mat_vec_q8_0_fast")
    ):
        return ops.ggml_mul_mat_vec_q8_0_fast(qweight, quant_x, row, col, dtype)
    return ops.ggml_mul_mat_vec_q8(qweight, quant_x, qweight_type, row, col, dtype)


def _gguf_mmvq_from_q8_activation_cached_out(
    qweight: torch.Tensor,
    quant_x: torch.Tensor,
    qweight_type: int,
    row: int,
    col: int,
    dtype: torch.dtype,
    cache_name: Any,
) -> torch.Tensor:
    if (
        ENABLE_GGUF_FUSED_MMVQ_OUTPUT_CACHE
        and hasattr(torch.ops._C, "ggml_mul_mat_vec_q8_out")
        and not (
            ENABLE_GGUF_Q8_0_FAST_MMVQ
            and qweight_type == WeightType.Q8_0
            and hasattr(torch.ops._C, "ggml_mul_mat_vec_q8_0_fast")
        )
    ):
        output = _get_gguf_tensor_buffer(
            (cache_name, "mmvq_output", qweight.data_ptr(), quant_x.shape[0], row),
            (quant_x.shape[0], row),
            dtype,
            qweight.device,
        )
        if output is not None:
            ops.ggml_mul_mat_vec_q8_out(
                qweight, quant_x, output, qweight_type, row, col
            )
            return output
    return _gguf_mmvq_from_q8_activation(
        qweight, quant_x, qweight_type, row, col, dtype
    )


def _fused_mul_mat_gguf_cached_mmvq(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    owner: Any,
    cache_name: Any,
) -> torch.Tensor:
    if (
        not ENABLE_GGUF_SINGLE_MMVQ_Q8_CACHE
        or qweight_type not in MMVQ_QUANT_TYPES
        or qweight_type in UNQUANTIZED_TYPES
        or ENABLE_GGUF_FORCE_DEQUANT_MATMUL
        or x.shape[0] == 0
        or x.shape[0] > _mmvq_safe_batch(qweight, qweight_type)
    ):
        return fused_mul_mat_gguf(x, qweight, qweight_type)

    if ENABLE_GGUF_LINEAR_PROFILE:
        _gguf_linear_profile_hit("single_mmvq_q8_cache", x, [qweight], [qweight_type])
    quant_x = _quantize_row_q8_1_cached(x, owner, cache_name)
    if hasattr(torch.ops._C, "ggml_mul_mat_vec_q8_out"):
        output = _get_gguf_tensor_buffer(
            (cache_name, "output", qweight.data_ptr(), x.shape[0], qweight.shape[0]),
            (x.shape[0], qweight.shape[0]),
            x.dtype,
            qweight.device,
        )
        if output is not None:
            ops.ggml_mul_mat_vec_q8_out(
                qweight, quant_x, output, qweight_type, qweight.shape[0], x.shape[1]
            )
            return output
    return _gguf_mmvq_from_q8_activation(
        qweight, quant_x, qweight_type, qweight.shape[0], x.shape[1], x.dtype
    )


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf

except AttributeError as error:
    raise error


def _mmvq_safe_batch(qweight: torch.Tensor, qweight_type: int) -> int:
    if qweight_type in IMATRIX_QUANT_TYPES:
        return 8 if qweight.shape[0] > 5120 else 16
    if qweight.shape[0] > 5120:
        if (
            qweight_type in MMQ_QUANT_TYPES
            and _gguf_gfx906_small_batch_mmvq_enabled()
        ):
            if qweight_type in {WeightType.Q4_0, WeightType.Q4_1, WeightType.Q4_K}:
                return _gguf_gfx906_q4_mmvq_max_batch()
            if qweight_type == WeightType.Q6_K:
                return _gguf_gfx906_q6_mmvq_max_batch()
            return _gguf_gfx906_mmvq_max_batch()
        return 2
    return 6


def _can_share_mmvq_activation(
    x: torch.Tensor,
    qweights: list[torch.Tensor],
    qweight_types: list[int],
) -> bool:
    if x.shape[0] == 0:
        return False
    for qweight, qweight_type in zip(qweights, qweight_types):
        if qweight_type not in MMVQ_QUANT_TYPES:
            return False
        if x.shape[0] > _mmvq_safe_batch(qweight, qweight_type):
            return False
    return True


def _shared_mmvq_max_batch(
    qweights: list[torch.Tensor],
    qweight_types: list[int],
) -> int:
    max_batch = torch.iinfo(torch.int32).max
    for qweight, qweight_type in zip(qweights, qweight_types):
        if qweight_type not in MMVQ_QUANT_TYPES:
            return 0
        max_batch = min(max_batch, _mmvq_safe_batch(qweight, qweight_type))
    return max_batch


def _get_full_mmvq_shard_weight(
    layer: torch.nn.Module,
    qweight_types: list[int],
) -> tuple[torch.Tensor, int] | None:
    qweight = layer.qweight
    shard_id = getattr(qweight, "shard_id", [])
    shard_offset_map = getattr(qweight, "shard_offset_map", None)
    if not shard_id or not shard_offset_map:
        return None
    if len(set(qweight_types)) != 1:
        return None

    qweight_type = qweight_types[0]
    if qweight_type not in MMVQ_QUANT_TYPES:
        return None

    expected_start = 0
    packed_width = qweight.shape[1]
    for idx in _ordered_gguf_shard_ids(shard_id):
        start, end, offset = shard_offset_map[idx]
        if start != expected_start or offset != packed_width:
            return None
        expected_start = end

    if expected_start != qweight.shape[0]:
        return None
    return qweight, qweight_type


def _collect_gguf_linear_shards(
    layer: torch.nn.Module,
) -> tuple[list[torch.Tensor], list[int]] | None:
    cached = getattr(layer, "_gguf_collected_shards", None)
    if cached is not None:
        return cached

    if not hasattr(layer, "qweight") or not hasattr(layer, "qweight_type"):
        return None
    if getattr(layer, "use_dense_gguf_fallback", False):
        return None
    if not _has_loaded_gguf_weight(layer):
        return None

    shard_id = getattr(layer.qweight, "shard_id", [])
    if shard_id:
        if all(isinstance(idx, int) for idx in shard_id):
            shard_id = sorted(shard_id)
        elif {"q", "k", "v"}.issubset(set(shard_id)):
            shard_id = ["q", "k", "v"]
        qweight = layer.qweight
        qweights = []
        qweight_types = []
        for idx in shard_id:
            qweight_type = layer.qweight_type.shard_weight_type[idx]
            if hasattr(layer.qweight, "shard_offset_map"):
                start, end, offset = layer.qweight.shard_offset_map[idx]
                shard = getattr(layer, "_gguf_shard_cache", {}).get(idx)
                if shard is None:
                    shard = qweight[start:end, :offset]
                    if not shard.is_contiguous():
                        shard = shard.contiguous()
            else:
                shard = qweight.data_container[qweight.shard_id_map[idx]].contiguous()
            qweights.append(shard)
            qweight_types.append(qweight_type)
        return qweights, qweight_types

    return [layer.qweight], [layer.qweight_type.weight_type]


def _coalesce_adjacent_same_type_mmvq_shards(
    qweights: list[torch.Tensor],
    qweight_types: list[int],
) -> tuple[list[torch.Tensor], list[int]] | None:
    if not ENABLE_GGUF_COALESCE_SAME_TYPE_SHARDS or len(qweights) <= 1:
        return None
    if len(set(qweight_types)) <= 1:
        return None

    merged_qweights: list[torch.Tensor] = []
    merged_qweight_types: list[int] = []
    changed = False
    i = 0
    while i < len(qweights):
        qweight_type = qweight_types[i]
        group = [qweights[i]]
        j = i + 1
        while (
            j < len(qweights)
            and qweight_types[j] == qweight_type
            and qweight_type in MMVQ_QUANT_TYPES
            and qweights[j].shape[1] == group[0].shape[1]
            and qweights[j].dtype == group[0].dtype
            and qweights[j].device == group[0].device
        ):
            group.append(qweights[j])
            j += 1

        if len(group) > 1:
            merged_qweights.append(torch.cat(group, dim=0).contiguous())
            merged_qweight_types.append(qweight_type)
            changed = True
        else:
            merged_qweights.append(qweights[i])
            merged_qweight_types.append(qweight_type)
        i = j

    if not changed:
        return None
    return merged_qweights, merged_qweight_types


def _fused_mul_mat_gguf_sharded_mmvq(
    x: torch.Tensor,
    qweights: list[torch.Tensor],
    qweight_types: list[int],
) -> torch.Tensor:
    return ops.ggml_mul_mat_vec_a8_sharded(qweights, x, qweight_types)


def _fused_mul_mat_gguf_sharded_mmvq_q8(
    quant_x: torch.Tensor,
    qweights: list[torch.Tensor],
    qweight_types: list[int],
    col: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return ops.ggml_mul_mat_vec_q8_sharded(qweights, quant_x, qweight_types, col, dtype)


def _fused_mul_mat_gguf_sharded_cached_mmvq(
    x: torch.Tensor,
    qweights: list[torch.Tensor],
    qweight_types: list[int],
    owner: Any,
    cache_name: Any,
) -> torch.Tensor:
    if (
        not ENABLE_GGUF_SHARDED_MMVQ_Q8_CACHE
        or x.shape[0] == 0
        or ENABLE_GGUF_FORCE_DEQUANT_MATMUL
    ):
        return fused_mul_mat_gguf_sharded_mmvq(x, qweights, qweight_types)

    quant_x = _quantize_row_q8_1_cached(x, owner, cache_name)
    if (
        ENABLE_GGUF_QKV3_MMVQ
        and len(qweights) == 3
        and qweight_types[0] == qweight_types[1]
        and qweight_types[0] in {WeightType.Q4_K, WeightType.Q5_K}
        and qweight_types[2] == WeightType.Q6_K
        and hasattr(torch.ops._C, "ggml_mul_mat_vec_q8_qkv3")
    ):
        if ENABLE_GGUF_LINEAR_PROFILE:
            _gguf_linear_profile_hit("sharded_qkv3_mmvq", x, qweights, qweight_types)
        return ops.ggml_mul_mat_vec_q8_qkv3(
            qweights[0],
            qweights[1],
            qweights[2],
            quant_x,
            qweight_types[0],
            x.shape[1],
            x.dtype,
        )
    return _fused_mul_mat_gguf_sharded_mmvq_q8(
        quant_x, qweights, qweight_types, x.shape[1], x.dtype
    )


def _can_group_same_type_mmvq(
    qweights: list[torch.Tensor],
    qweight_types: list[int],
) -> bool:
    if not ENABLE_GGUF_GROUPED_SAME_TYPE_MMVQ:
        return False
    return (
        len(qweights) <= 8
        and len(set(qweight_types)) == 1
        and qweight_types[0] in MMVQ_QUANT_TYPES
    )


def _fused_mul_mat_gguf_grouped_same_type_mmvq_q8(
    quant_x: torch.Tensor,
    qweights: list[torch.Tensor],
    qweight_type: int,
    col: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return ops.ggml_mul_mat_vec_q8_grouped_same_type(
        qweights, quant_x, qweight_type, col, dtype
    )


def _get_grouped_mmvq_output_buffer(
    cache_name: str,
    qweights: list[torch.Tensor],
    qweight_types: list[int],
    tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not ENABLE_GGUF_GROUPED_MMVQ_OUTPUT_CACHE:
        return None
    total_rows = sum(qweight.shape[0] for qweight in qweights)
    key = (
        cache_name,
        tuple(qweight.data_ptr() for qweight in qweights),
        tuple(int(qweight_type) for qweight_type in qweight_types),
        tokens,
        total_rows,
    )
    return _get_gguf_tensor_buffer(
        key,
        (tokens, total_rows),
        dtype,
        qweights[0].device,
    )


def _fused_mul_mat_gguf_sharded_mmvq_fake(
    x: torch.Tensor,
    qweights: list[torch.Tensor],
    qweight_types: list[int],
) -> torch.Tensor:
    return torch.empty(
        x.shape[0],
        sum(qweight.shape[0] for qweight in qweights),
        dtype=x.dtype,
        device=x.device,
    )


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf_sharded_mmvq",
        op_func=_fused_mul_mat_gguf_sharded_mmvq,
        fake_impl=_fused_mul_mat_gguf_sharded_mmvq_fake,
    )
    fused_mul_mat_gguf_sharded_mmvq = torch.ops.vllm._fused_mul_mat_gguf_sharded_mmvq

except AttributeError as error:
    raise error


def _fused_mul_mat_gguf_sharded_mmvq_q8_fake(
    quant_x: torch.Tensor,
    qweights: list[torch.Tensor],
    qweight_types: list[int],
    col: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        quant_x.shape[0],
        sum(qweight.shape[0] for qweight in qweights),
        dtype=dtype,
        device=qweights[0].device,
    )


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf_sharded_mmvq_q8",
        op_func=_fused_mul_mat_gguf_sharded_mmvq_q8,
        fake_impl=_fused_mul_mat_gguf_sharded_mmvq_q8_fake,
    )
    fused_mul_mat_gguf_sharded_mmvq_q8 = (
        torch.ops.vllm._fused_mul_mat_gguf_sharded_mmvq_q8
    )

except AttributeError as error:
    raise error


def _fused_mul_mat_gguf_grouped_same_type_mmvq_q8_fake(
    quant_x: torch.Tensor,
    qweights: list[torch.Tensor],
    qweight_type: int,
    col: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        quant_x.shape[0],
        sum(qweight.shape[0] for qweight in qweights),
        dtype=dtype,
        device=qweights[0].device,
    )


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf_grouped_same_type_mmvq_q8",
        op_func=_fused_mul_mat_gguf_grouped_same_type_mmvq_q8,
        fake_impl=_fused_mul_mat_gguf_grouped_same_type_mmvq_q8_fake,
    )
    fused_mul_mat_gguf_grouped_same_type_mmvq_q8 = (
        torch.ops.vllm._fused_mul_mat_gguf_grouped_same_type_mmvq_q8
    )

except AttributeError as error:
    raise error


def _get_grouped_gguf_mmvq_plan(
    layers: list[torch.nn.Module],
) -> tuple[list[torch.Tensor], list[int], list[int], int, bool] | None:
    if not layers:
        return None

    layer_ids = tuple(id(layer) for layer in layers)
    cached = getattr(layers[0], "_gguf_grouped_mmvq_plan", None)
    if cached is not None and cached[0] == layer_ids:
        return cached[1]

    qweights = []
    qweight_types = []
    split_sizes = []
    for layer in layers:
        if getattr(layer, "bias", None) is not None:
            return None
        if getattr(layer, "gather_output", False) and getattr(layer, "tp_size", 1) > 1:
            return None

        collected = _collect_gguf_linear_shards(layer)
        if collected is None:
            return None

        layer_qweights, layer_qweight_types = collected
        layer_mmvq_max_batch = _shared_mmvq_max_batch(
            layer_qweights, layer_qweight_types
        )
        if layer_mmvq_max_batch <= 0:
            return None

        full_mmvq = _get_full_mmvq_shard_weight(layer, layer_qweight_types)
        if full_mmvq is not None:
            qweight, qweight_type = full_mmvq
            qweights.append(qweight)
            qweight_types.append(qweight_type)
            split_sizes.append(qweight.shape[0])
        else:
            qweights.extend(layer_qweights)
            qweight_types.extend(layer_qweight_types)
            split_sizes.append(sum(qweight.shape[0] for qweight in layer_qweights))

    mmvq_max_batch = _shared_mmvq_max_batch(qweights, qweight_types)
    if mmvq_max_batch <= 0:
        return None

    use_same_type_kernel = _can_group_same_type_mmvq(qweights, qweight_types)
    plan = (
        qweights,
        qweight_types,
        split_sizes,
        mmvq_max_batch,
        use_same_type_kernel,
    )
    layers[0]._gguf_grouped_mmvq_plan = (layer_ids, plan)
    return plan


def try_grouped_gguf_linear_mmvq(
    x: torch.Tensor,
    layers: list[torch.nn.Module],
) -> list[torch.Tensor] | None:
    if x.shape[0] == 0:
        return None

    plan = _get_grouped_gguf_mmvq_plan(layers)
    if plan is None:
        return None

    (
        qweights,
        qweight_types,
        split_sizes,
        mmvq_max_batch,
        use_same_type_kernel,
    ) = plan
    if x.shape[0] > mmvq_max_batch:
        return None

    quant_x = _quantize_row_q8_1_cached(
        x, layers[0], "_gguf_grouped_mmvq_quant_x"
    )
    out = _get_grouped_mmvq_output_buffer(
        "_gguf_grouped_mmvq_output",
        qweights,
        qweight_types,
        x.shape[0],
        x.dtype,
    )
    if use_same_type_kernel:
        if ENABLE_GGUF_LINEAR_PROFILE:
            _gguf_linear_profile_hit(
                "grouped_mmvq",
                x,
                qweights,
                qweight_types,
                grouped_same_type=True,
            )
        if out is not None and hasattr(
            torch.ops._C, "ggml_mul_mat_vec_q8_grouped_same_type_out"
        ):
            ops.ggml_mul_mat_vec_q8_grouped_same_type_out(
                qweights, quant_x, out, qweight_types[0], x.shape[1]
            )
        else:
            out = fused_mul_mat_gguf_grouped_same_type_mmvq_q8(
                quant_x, qweights, qweight_types[0], x.shape[1], x.dtype
            )
    else:
        if ENABLE_GGUF_LINEAR_PROFILE:
            _gguf_linear_profile_hit(
                "grouped_sharded_mmvq",
                x,
                qweights,
                qweight_types,
                grouped_same_type=False,
            )
        if out is not None and hasattr(
            torch.ops._C, "ggml_mul_mat_vec_q8_sharded_out"
        ):
            ops.ggml_mul_mat_vec_q8_sharded_out(
                qweights, quant_x, out, qweight_types, x.shape[1]
            )
        else:
            out = fused_mul_mat_gguf_sharded_mmvq_q8(
                quant_x, qweights, qweight_types, x.shape[1], x.dtype
            )
    return list(torch.split(out, split_sizes, dim=-1))


def try_gguf_silu_and_mul_down_mmvq(
    gate_up: torch.Tensor,
    down_layer: torch.nn.Module,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not ENABLE_GGUF_SHARED_EXPERT_FUSED_ACT_DOWN:
        return None
    if gate_up.dim() != 2 or gate_up.shape[1] % 2 != 0:
        return None
    if gate_up.shape[0] == 0:
        return None
    if not gate_up.is_cuda:
        return None
    if not (
        hasattr(torch.ops._C, "ggml_silu_and_mul_quantize_row_q8_1")
        and hasattr(torch.ops._C, "ggml_mul_mat_vec_q8")
    ):
        return None
    if getattr(down_layer, "use_dense_gguf_fallback", False):
        return None
    if getattr(down_layer, "input_is_parallel", True) is not True:
        return None
    if (
        getattr(down_layer, "reduce_results", False)
        and getattr(down_layer, "tp_size", 1) > 1
    ):
        return None
    if getattr(down_layer, "return_bias", True) is not True:
        return None
    if not hasattr(down_layer, "qweight") or not hasattr(down_layer, "qweight_type"):
        return None
    qweight = down_layer.qweight
    qweight_type_param = down_layer.qweight_type
    if qweight is None or qweight_type_param is None:
        return None
    if not _has_loaded_gguf_weight(down_layer):
        return None
    col = gate_up.shape[1] // 2

    if getattr(qweight, "shard_id", []):
        collected = _collect_gguf_linear_shards(down_layer)
        if collected is None:
            return None
        qweights, qweight_types = collected
        if not _can_share_mmvq_activation(gate_up[:, :col], qweights, qweight_types):
            return None
        if ENABLE_GGUF_LINEAR_PROFILE:
            _gguf_linear_profile_hit(
                "silu_down_sharded_mmvq", gate_up, qweights, qweight_types
            )
        quant_out = _silu_and_mul_quantize_row_q8_1_cached(
            gate_up, down_layer, "_gguf_silu_down_quant_x"
        )
        out = _get_gguf_tensor_buffer(
            ("_gguf_silu_down_sharded_output", id(down_layer), gate_up.shape[0]),
            (gate_up.shape[0], sum(qweight.shape[0] for qweight in qweights)),
            gate_up.dtype,
            gate_up.device,
        )
        if out is not None and hasattr(torch.ops._C, "ggml_mul_mat_vec_q8_sharded_out"):
            ops.ggml_mul_mat_vec_q8_sharded_out(qweights, quant_out, out, qweight_types, col)
        else:
            out = _fused_mul_mat_gguf_sharded_mmvq_q8(
                quant_out, qweights, qweight_types, col, gate_up.dtype
            )
        if bias is not None:
            out.add_(bias)
        return out

    qweight_type = qweight_type_param.weight_type
    if qweight_type not in MMVQ_QUANT_TYPES:
        return None
    if gate_up.shape[0] > _mmvq_safe_batch(qweight, qweight_type):
        return None

    if ENABLE_GGUF_LINEAR_PROFILE:
        _gguf_linear_profile_hit("silu_down_mmvq", gate_up, [qweight], [qweight_type])
    quant_out = _silu_and_mul_quantize_row_q8_1_cached(
        gate_up, down_layer, "_gguf_silu_down_quant_x"
    )
    out = _gguf_mmvq_from_q8_activation_cached_out(
        qweight,
        quant_out,
        qweight_type,
        qweight.shape[0],
        col,
        gate_up.dtype,
        "_gguf_silu_down_output",
    )
    if bias is not None:
        out.add_(bias)
    return out


def try_gguf_rms_norm_gated_out_proj_mmvq(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    out_layer: torch.nn.Module,
    epsilon: float,
    norm_before_gate: bool,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not ENABLE_GGUF_RMS_GATED_OUT_PROJ_MMVQ:
        return None
    if x.dim() != 3 or gate.shape != x.shape:
        return None
    if x.shape[0] == 0:
        return None
    if not (x.is_cuda and gate.is_cuda and weight.is_cuda):
        return None
    if not (
        hasattr(torch.ops._C, "ggml_rms_norm_gated_quantize_row_q8_1")
        and hasattr(torch.ops._C, "ggml_mul_mat_vec_q8")
    ):
        return None
    if x.dtype != gate.dtype or x.dtype != weight.dtype:
        return None
    if x.stride(-1) != 1 or gate.stride(-1) != 1 or not weight.is_contiguous():
        return None
    if getattr(out_layer, "use_dense_gguf_fallback", False):
        return None
    if getattr(out_layer, "input_is_parallel", True) is not True:
        return None
    if (
        getattr(out_layer, "reduce_results", False)
        and getattr(out_layer, "tp_size", 1) > 1
    ):
        return None
    if getattr(out_layer, "return_bias", True) is not True:
        return None
    if not hasattr(out_layer, "qweight") or not hasattr(out_layer, "qweight_type"):
        return None
    qweight = out_layer.qweight
    qweight_type_param = out_layer.qweight_type
    if qweight is None or qweight_type_param is None:
        return None
    if getattr(qweight, "shard_id", []):
        return None
    if not _has_loaded_gguf_weight(out_layer):
        return None
    qweight_type = qweight_type_param.weight_type
    if qweight_type not in MMVQ_QUANT_TYPES:
        return None

    col = x.shape[1] * x.shape[2]
    if col % 512 != 0 or x.shape[2] % 32 != 0:
        return None
    if weight.numel() != x.shape[2]:
        return None
    if x.shape[0] > _mmvq_safe_batch(qweight, qweight_type):
        return None

    if ENABLE_GGUF_LINEAR_PROFILE:
        _gguf_linear_profile_hit(
            "rms_gated_out_mmvq",
            x.reshape(x.shape[0], col),
            [qweight],
            [qweight_type],
        )
    quant_x = _rms_norm_gated_quantize_row_q8_1_cached(
        x,
        weight,
        gate,
        epsilon,
        norm_before_gate,
        out_layer,
        "_gguf_rms_gated_out_quant_x",
    )
    out = _gguf_mmvq_from_q8_activation_cached_out(
        qweight,
        quant_x,
        qweight_type,
        qweight.shape[0],
        col,
        x.dtype,
        "_gguf_rms_gated_out_output",
    )
    if bias is not None:
        out.add_(bias)
    return out


def try_gguf_sigmoid_gated_out_proj_mmvq(
    x: torch.Tensor,
    gate: torch.Tensor,
    out_layer: torch.nn.Module,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not ENABLE_GGUF_SIGMOID_GATED_OUT_PROJ_MMVQ:
        return None
    if x.dim() != 2 or gate.shape != x.shape:
        return None
    if x.shape[0] == 0:
        return None
    if not (x.is_cuda and gate.is_cuda):
        return None
    if not (
        hasattr(torch.ops._C, "ggml_sigmoid_and_mul_quantize_row_q8_1")
        and hasattr(torch.ops._C, "ggml_mul_mat_vec_q8")
    ):
        return None
    if x.dtype != gate.dtype:
        return None
    if x.stride(-1) != 1 or gate.stride(-1) != 1:
        return None
    if getattr(out_layer, "use_dense_gguf_fallback", False):
        return None
    if getattr(out_layer, "input_is_parallel", True) is not True:
        return None
    if (
        getattr(out_layer, "reduce_results", False)
        and getattr(out_layer, "tp_size", 1) > 1
    ):
        return None
    if getattr(out_layer, "return_bias", True) is not True:
        return None
    if not hasattr(out_layer, "qweight") or not hasattr(out_layer, "qweight_type"):
        return None
    qweight = out_layer.qweight
    qweight_type_param = out_layer.qweight_type
    if qweight is None or qweight_type_param is None:
        return None
    if getattr(qweight, "shard_id", []):
        return None
    if not _has_loaded_gguf_weight(out_layer):
        return None
    qweight_type = qweight_type_param.weight_type
    if qweight_type not in MMVQ_QUANT_TYPES:
        return None

    col = x.shape[1]
    if x.shape[0] > _mmvq_safe_batch(qweight, qweight_type):
        return None

    if ENABLE_GGUF_LINEAR_PROFILE:
        _gguf_linear_profile_hit(
            "sigmoid_gated_out_mmvq",
            x,
            [qweight],
            [qweight_type],
        )
    quant_x = _sigmoid_and_mul_quantize_row_q8_1_cached(
        x, gate, out_layer, "_gguf_sigmoid_gated_out_quant_x"
    )
    out = _gguf_mmvq_from_q8_activation_cached_out(
        qweight,
        quant_x,
        qweight_type,
        qweight.shape[0],
        col,
        x.dtype,
        "_gguf_sigmoid_gated_out_output",
    )
    if bias is not None:
        out.add_(bias)
    return out


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    (
        use_weighted_sum,
        use_fused_act_quant,
        fused_act_quant_max_tokens,
        use_fused_w2_weighted_sum,
        fused_w2_weighted_sum_max_topk,
    ) = _gguf_moe_runtime_config()

    def reduce_expert_outputs(out: torch.Tensor, output: torch.Tensor) -> None:
        if use_weighted_sum:
            ops.moe_weighted_sum(out, topk_weights, output)
        else:
            out.mul_(topk_weights.view(out.shape[0], out.shape[1], 1))
            ops.moe_sum(out, output)

    def act(x: torch.Tensor):
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        if activation == "silu":
            torch.ops._C.silu_and_mul(out, x)
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(out, x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        return out

    # unless we decent expert reuse we are better off running moe_vec kernel
    if (
        qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] >= _gguf_moe_mmq_min_tokens()
    ):
        # lazy import to avoid triggering triton import in CPU/backend decode paths
        from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        _gguf_moe_profile_hit(
            "mmq_prefill",
            x,
            top_k,
            qweight_type,
            qweight_type2,
        )
        timer = _GGUFMoeTimer(
            "mmq_prefill",
            x,
            top_k,
            qweight_type,
            qweight_type2,
        )
        BLOCK_SIZE = ops.ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = timer.record(
            "align",
            lambda: moe_align_block_size(topk_ids, BLOCK_SIZE, E),
        )
        out = timer.record(
            "w1",
            lambda: ops.ggml_moe_a8(
                x,
                w1,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                qweight_type,
                N,
                top_k,
                num_tokens,
            ),
        )
        out = timer.record("act", lambda: act(out))
        out = timer.record(
            "w2",
            lambda: ops.ggml_moe_a8(
                out,
                w2,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                qweight_type2,
                w2.shape[1],
                1,
                num_tokens * top_k,
            ),
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1])
        out_hidden_states = torch.empty_like(x)
        timer.record("reduce", lambda: reduce_expert_outputs(out, out_hidden_states))
        timer.report()
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        (
            fused_act_quant_available,
            fused_w2_weighted_sum_available,
            host_fused_act_w2_weighted_sum_available,
        ) = _gguf_moe_ops_available()

        if (
            ENABLE_GGUF_MOE_HOST_FUSED_ACT_W2
            and use_fused_act_quant
            and use_fused_w2_weighted_sum
            and activation == "silu"
            and top_k <= fused_w2_weighted_sum_max_topk
            and host_fused_act_w2_weighted_sum_available
        ):
            timer = _GGUFMoeTimer(
                "mmvq_decode_host_fused_act_w2_weighted_sum",
                x,
                top_k,
                qweight_type,
                qweight_type2,
            )
            out_hidden_states = timer.record(
                "host_fused",
                lambda: _gguf_moe_decode_fused_cached(
                    x,
                    w1,
                    w2,
                    topk_ids,
                    topk_weights,
                    top_k,
                    qweight_type,
                    qweight_type2,
                    N,
                    w2.shape[1],
                    num_tokens,
                ),
            )
            if out_hidden_states is not None:
                _gguf_moe_profile_hit(
                    "mmvq_decode_host_fused_act_w2_weighted_sum",
                    x,
                    top_k,
                    qweight_type,
                    qweight_type2,
                )
                timer.report()
                return out_hidden_states

        timer = _GGUFMoeTimer(
            "mmvq_decode",
            x,
            top_k,
            qweight_type,
            qweight_type2,
        )
        out = timer.record(
            "w1",
            lambda: _gguf_moe_a8_vec_cached(
                x, w1, topk_ids, top_k, qweight_type, N, num_tokens
            ),
        )
        if (
            use_fused_act_quant
            and num_tokens <= fused_act_quant_max_tokens
            and activation == "silu"
            and fused_act_quant_available
        ):
            quant_out = timer.record(
                "act_quant",
                lambda: _silu_and_mul_quantize_row_q8_1_cached(
                    out,
                    None,
                    (
                        "_gguf_moe_silu_quant_x",
                        w2.data_ptr(),
                        num_tokens,
                        top_k,
                        out.shape[-1],
                    ),
                ),
            )
            if (
                use_fused_w2_weighted_sum
                and top_k <= fused_w2_weighted_sum_max_topk
                and qweight_type2 in MMVQ_QUANT_TYPES
                and fused_w2_weighted_sum_available
            ):
                _gguf_moe_profile_hit(
                    "mmvq_decode_fused_act_w2_weighted_sum",
                    x,
                    top_k,
                    qweight_type,
                    qweight_type2,
                )
                out_hidden_states = timer.record(
                    "w2_weighted_sum",
                    lambda: _gguf_moe_q8_vec_weighted_sum_cached(
                        quant_out,
                        w2,
                        topk_ids,
                        topk_weights,
                        top_k,
                        qweight_type2,
                        w2.shape[1],
                        num_tokens,
                        out.shape[-1] // 2,
                        x.dtype,
                    ),
                )
                timer.report()
                return out_hidden_states
            else:
                _gguf_moe_profile_hit(
                    "mmvq_decode_fused_act_w2_vec",
                    x,
                    top_k,
                    qweight_type,
                    qweight_type2,
                )
                out = timer.record(
                    "w2_vec",
                    lambda: ops.ggml_moe_q8_vec(
                        quant_out,
                        w2,
                        topk_ids,
                        1,
                        qweight_type2,
                        w2.shape[1],
                        num_tokens * top_k,
                        out.shape[-1] // 2,
                        x.dtype,
                    ),
                )
        else:
            _gguf_moe_profile_hit(
                "mmvq_decode_unfused_act",
                x,
                top_k,
                qweight_type,
                qweight_type2,
            )
            out = timer.record("act", lambda: act(out))
            out = timer.record(
                "w2",
                lambda: ops.ggml_moe_a8_vec(
                    out,
                    w2,
                    topk_ids,
                    1,
                    qweight_type2,
                    w2.shape[1],
                    num_tokens * top_k,
                ),
            )
        out = out.reshape(num_tokens, top_k, w2.shape[1])
        out_hidden_states = torch.empty_like(x)
        timer.record("reduce", lambda: reduce_expert_outputs(out, out_hidden_states))
        timer.report()
    else:
        _gguf_moe_profile_hit(
            "slow_fallback",
            x,
            topk_ids.shape[1],
            qweight_type,
            qweight_type2,
        )
        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        out_hidden_states = torch.empty_like(x)
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                expert_up = w1[ii]

                out = fused_mul_mat_gguf(inp, expert_up, qweight_type)
                out = act(out)

                expert_down = w2[ii]
                current_state = fused_mul_mat_gguf(
                    out, expert_down, qweight_type2
                ).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf

except AttributeError as error:
    raise error


def _apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ops.ggml_dequantize(
            quant, qweight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


def _apply_gguf_embedding_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.empty(x.shape[0], hidden_size, dtype=dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_apply_gguf_embedding",
        op_func=_apply_gguf_embedding,
        fake_impl=_apply_gguf_embedding_fake,
    )
    apply_gguf_embedding = torch.ops.vllm._apply_gguf_embedding

except AttributeError as error:
    raise error


def _dequantize_gguf_weight(
    qweight: torch.Tensor,
    qweight_type: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return qweight.to(dtype=out_dtype).contiguous()
    if qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        return ops.ggml_dequantize(
            qweight, qweight_type, *shape, out_dtype
        ).contiguous()

    qweight_type = WeightType(qweight_type)
    raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


def _is_gfx906_qwen35_linear_attn_fallback(layer: torch.nn.Module) -> bool:
    return False


def _ordered_gguf_shard_ids(shard_id: list[int | str]) -> list[int | str]:
    if all(isinstance(idx, int) for idx in shard_id):
        return sorted(shard_id)
    if {"q", "k", "v"}.issubset(set(shard_id)):
        return ["q", "k", "v"]
    return list(shard_id)


def _materialize_dense_gguf_weight(
    layer: torch.nn.Module,
    out_dtype: torch.dtype,
) -> torch.nn.Parameter:
    shard_id = layer.qweight.shard_id
    if shard_id:
        qweight = layer.qweight
        parts = []
        for idx in _ordered_gguf_shard_ids(shard_id):
            qweight_type = layer.qweight_type.shard_weight_type[idx]
            if hasattr(qweight, "shard_offset_map"):
                start, end, offset = qweight.shard_offset_map[idx]
                shard = qweight[start:end, :offset].contiguous()
            else:
                shard = qweight.data_container[qweight.shard_id_map[idx]].contiguous()
            parts.append(_dequantize_gguf_weight(shard, qweight_type, out_dtype))
        weight = torch.cat(parts, dim=0).contiguous()
    else:
        weight = _dequantize_gguf_weight(
            layer.qweight,
            layer.qweight_type.weight_type,
            out_dtype,
        )
    return Parameter(weight, requires_grad=False)


def _has_loaded_gguf_weight(layer: torch.nn.Module) -> bool:
    qweight = layer.qweight
    if getattr(qweight, "shard_id", None):
        return True
    return not isinstance(qweight, UninitializedParameter)


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        if _is_gfx906_qwen35_linear_attn_fallback(layer) and _has_loaded_gguf_weight(
            layer
        ):
            layer.register_parameter(
                "weight",
                _materialize_dense_gguf_weight(layer, self.params_dtype),
            )
            layer.register_parameter("qweight", None)
            layer.register_parameter("qweight_type", None)
            layer.use_dense_gguf_fallback = True
            return
        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.
        self._create_padded_weight_param(layer)
        self._maybe_repack_iq4_xs_to_q8_0(layer)
        self._cache_gguf_shards(layer)
        self._cache_gguf_shard_metadata(layer)

    def _maybe_repack_iq4_xs_to_q8_0(self, layer: torch.nn.Module) -> None:
        if not _gguf_repack_iq4_xs_to_q8_0_enabled():
            return
        if not hasattr(torch.ops._C, "ggml_repack_iq4_xs_to_q8_0"):
            return
        qweight = layer.qweight
        qweight_type = layer.qweight_type
        if qweight is None or qweight_type is None:
            return
        if not qweight.is_cuda or not qweight.is_contiguous():
            return

        q8_type = int(WeightType.Q8_0)
        iq4_xs_type = int(WeightType.IQ4_XS)
        block_size, type_size = gguf.GGML_QUANT_SIZES[iq4_xs_type]
        shard_offset_map = getattr(qweight, "shard_offset_map", None)

        if shard_offset_map:
            shard_ids = list(getattr(qweight, "shard_id", []))
            if not shard_ids:
                return
            if any(
                int(qweight_type.shard_weight_type.get(idx, -1)) != iq4_xs_type
                for idx in shard_ids
            ):
                return
            offsets = [shard_offset_map[idx][2] for idx in shard_ids]
            if len(set(offsets)) != 1 or offsets[0] != qweight.shape[1]:
                return
            col = offsets[0] // type_size * block_size
            repacked = ops.ggml_repack_iq4_xs_to_q8_0(qweight, qweight.shape[0], col)
            new_offset = repacked.shape[1]
            new_map = {
                idx: (start, end, new_offset)
                for idx, (start, end, _offset) in shard_offset_map.items()
            }
            for idx in shard_ids:
                qweight_type.shard_weight_type[idx] = q8_type
        else:
            if int(qweight_type.weight_type) != iq4_xs_type:
                return
            col = qweight.shape[1] // type_size * block_size
            repacked = ops.ggml_repack_iq4_xs_to_q8_0(qweight, qweight.shape[0], col)
            new_map = None
            qweight_type.weight_type = q8_type
            if qweight_type.numel() > 0:
                qweight_type.data.fill_(q8_type)

        repacked_param = Parameter(repacked.contiguous(), requires_grad=False)
        attrs = vars(qweight).copy()
        if new_map is not None:
            attrs["shard_offset_map"] = new_map
        set_weight_attrs(repacked_param, attrs)
        layer.register_parameter("qweight", repacked_param)
        logger.info_once("Repacked GGUF IQ4_XS weights to Q8_0 for faster MMVQ.")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        data_container = qweight.data_container
        if len(data_container) == 1:
            data = data_container[0].to(device=qweight.device).contiguous()
            qweight.data_container.clear()
            padded_param = Parameter(data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(
                padded_param,
                {"shard_offset_map": {shard_id[0]: (0, data.size(0), data.size(1))}},
            )
            layer.register_parameter("qweight", padded_param)
        elif len(data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            # concat dim0 and pad dim1
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            # Pad the quantized weights to dense tensor, and create a map
            # with the location of each shard in the padded tensor.
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            # (dim0_start, dim0_end, dim1_size)
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def _cache_gguf_shards(self, layer: torch.nn.Module):
        qweight = layer.qweight
        shard_offset_map = getattr(qweight, "shard_offset_map", None)
        if not shard_offset_map:
            return

        cache = {}
        for idx, (start, end, offset) in shard_offset_map.items():
            shard = qweight[start:end, :offset]
            if not shard.is_contiguous():
                shard = shard.contiguous()
            cache[idx] = shard
        layer._gguf_shard_cache = cache

    def _cache_gguf_shard_metadata(self, layer: torch.nn.Module):
        if not getattr(layer.qweight, "shard_id", []):
            return
        collected = _collect_gguf_linear_shards(layer)
        if collected is None:
            return
        qweights, qweight_types = collected
        layer._gguf_collected_shards = collected
        layer._gguf_sharded_mmvq_max_batch = _shared_mmvq_max_batch(
            qweights, qweight_types
        )
        full_mmvq = _get_full_mmvq_shard_weight(layer, qweight_types)
        if full_mmvq is not None:
            qweight, qweight_type = full_mmvq
            layer._gguf_full_shard_mmvq = (
                qweight,
                qweight_type,
                _mmvq_safe_batch(qweight, qweight_type),
            )
            return

        coalesced = _coalesce_adjacent_same_type_mmvq_shards(
            qweights, qweight_types
        )
        if coalesced is not None:
            coalesced_qweights, coalesced_qweight_types = coalesced
            layer._gguf_coalesced_shards = (
                coalesced_qweights,
                coalesced_qweight_types,
                _shared_mmvq_max_batch(
                    coalesced_qweights, coalesced_qweight_types
                ),
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "use_dense_gguf_fallback", False):
            weight = layer.weight.to(dtype=x.dtype)
            out = x @ weight.T
            if bias is not None:
                out.add_(bias.to(dtype=out.dtype))
            return out

        shard_id = getattr(layer.qweight, "shard_id", [])

        if shard_id:
            collected = _collect_gguf_linear_shards(layer)
            assert collected is not None
            qweights, qweight_types = collected
            mmvq_max_batch = getattr(layer, "_gguf_sharded_mmvq_max_batch", None)
            can_share_mmvq = (
                x.shape[0] > 0
                and mmvq_max_batch is not None
                and x.shape[0] <= mmvq_max_batch
            )
            if mmvq_max_batch is None:
                can_share_mmvq = _can_share_mmvq_activation(x, qweights, qweight_types)
            if can_share_mmvq:
                full_mmvq = getattr(layer, "_gguf_full_shard_mmvq", None)
                if full_mmvq is not None and x.shape[0] <= full_mmvq[2]:
                    qweight, qweight_type, _ = full_mmvq
                    if ENABLE_GGUF_LINEAR_PROFILE:
                        _gguf_linear_profile_hit(
                            "sharded_full_mmvq", x, [qweight], [qweight_type]
                        )
                    out = _fused_mul_mat_gguf_cached_mmvq(
                        x, qweight, qweight_type, layer, "_gguf_full_mmvq_quant_x"
                    )
                else:
                    coalesced = getattr(layer, "_gguf_coalesced_shards", None)
                    if coalesced is not None and x.shape[0] <= coalesced[2]:
                        qweights, qweight_types = coalesced[0], coalesced[1]
                        profile_path = "sharded_coalesced_mmvq"
                    else:
                        profile_path = "sharded_mmvq"
                    if ENABLE_GGUF_LINEAR_PROFILE:
                        _gguf_linear_profile_hit(
                            profile_path, x, qweights, qweight_types
                        )
                    out = _fused_mul_mat_gguf_sharded_cached_mmvq(
                        x,
                        qweights,
                        qweight_types,
                        layer,
                        "_gguf_sharded_mmvq_quant_x",
                    )
            else:
                if ENABLE_GGUF_LINEAR_PROFILE:
                    _gguf_linear_profile_hit(
                        "sharded_cat_fallback", x, qweights, qweight_types
                    )
                result = [
                    fused_mul_mat_gguf(x, shard, qweight_type)
                    for shard, qweight_type in zip(qweights, qweight_types)
                ]
                out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = _fused_mul_mat_gguf_cached_mmvq(
                x, qweight, qweight_type, layer, "_gguf_single_mmvq_quant_x"
            )
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(
        self,
        quant_config: GGUFConfig,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        # gate up proj
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        # gate down proj
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert activation == "silu", "Only SiLU activation is supported."
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        topk_weights, topk_ids, _ = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )
        return fused_moe_gguf(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            layer.w13_qweight_type.weight_type,
            layer.w2_qweight_type.weight_type,
            activation,
        )


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]

        return apply_gguf_embedding(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: list[torch.Tensor]
