# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from datetime import timedelta
from functools import cache, lru_cache, wraps
from typing import TYPE_CHECKING

import regex as re
import torch
from torch.distributed import PrefixStore, ProcessGroup
from torch.distributed.distributed_c10d import is_nccl_available

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import cuda_device_count_stateless
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType

logger = init_logger(__name__)

try:
    from amdsmi import (
        AmdSmiMemoryType,
        AmdSmiException,
        amdsmi_get_gpu_asic_info,
        amdsmi_get_gpu_device_uuid,
        amdsmi_get_gpu_memory_total,
        amdsmi_get_gpu_memory_usage,
        amdsmi_get_processor_handles,
        amdsmi_init,
        amdsmi_shut_down,
        amdsmi_topo_get_link_type,
    )
    _AMDSMI_AVAILABLE = True
except ImportError as e:
    _AMDSMI_AVAILABLE = False
    logger.warning(
        "Failed to import from amdsmi with %r. "
        "AMD GPU monitoring will be unavailable. "
        "This is optional and won't affect core functionality.",
        e,
    )

try:
    import vllm._C  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Failed to import vllm._C. This is a required module for vLLM. "
        "Please ensure vLLM is properly compiled. "
        f"Original error: {e}"
    ) from e

# import custom ops, trigger op registration
try:
    import vllm._rocm_C  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Failed to import vllm._rocm_C. This is a required module for ROCm builds. "
        "Please ensure vLLM is properly compiled with ROCm support. "
        f"Original error: {e}"
    ) from e

# Models not supported by ROCm.
_ROCM_UNSUPPORTED_MODELS: list[str] = []

# Models partially supported by ROCm.
# Architecture -> Reason.
_ROCM_PARTIALLY_SUPPORTED_MODELS: dict[str, str] = {}
_ROCM_DEVICE_ID_NAME_MAP: dict[str, str] = {
    "0x74a0": "AMD_Instinct_MI300A",
    "0x74a1": "AMD_Instinct_MI300X",
    "0x74b5": "AMD_Instinct_MI300X",  # MI300X VF
    "0x74a2": "AMD_Instinct_MI308X",
    "0x74a5": "AMD_Instinct_MI325X",
    "0x74b9": "AMD_Instinct_MI325X",  # MI325X VF
    "0x74a9": "AMD_Instinct_MI300X_HF",
    "0x74bd": "AMD_Instinct_MI300X_HF",
    "0x744c": "AMD_Radeon_RX7900XTX",
}

# Known amdsmi target_graphics_version quirks.
# Some ROCm versions may return non-standard names like
# "gfx9006" instead of the canonical "gfx906".
_AMDSMI_GFX_NORMALIZATION: dict[str, str] = {
    "gfx9006": "gfx906",
}


def _sync_hip_cuda_env_vars():
    """Ensure HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES are consistent.
    Treats empty string as unset. Raises on genuine conflicts."""
    hip_val = os.environ.get("HIP_VISIBLE_DEVICES") or None
    cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES") or None

    if hip_val is not None and cuda_val is not None:
        if hip_val != cuda_val:
            raise ValueError(
                f"Inconsistent GPU visibility env vars: "
                f"HIP_VISIBLE_DEVICES='{hip_val}' vs "
                f"CUDA_VISIBLE_DEVICES='{cuda_val}'. "
                f"Please set only one, or ensure they match."
            )
    elif hip_val is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_val
    elif cuda_val is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = cuda_val


# Sync at import time - catches misconfigurations from process start.
_sync_hip_cuda_env_vars()


def _set_rocm_nccl_workarounds():
    """Reduce ROCm NCCL watchdog/monitoring interference during graph capture.

    Keep these as setdefault so advanced users can still override them
    explicitly from their environment.
    """
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "0")


_set_rocm_nccl_workarounds()

# AMDSMI utils
# Note that NVML is not affected by `{CUDA/HIP_VISIBLE_DEVICES}`,
# all the related functions work on real physical device ids.
# the major benefit of using AMDSMI is that it will not initialize CUDA


def with_amdsmi_context(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _AMDSMI_AVAILABLE:
            return fn(*args, **kwargs)
        amdsmi_init()
        try:
            return fn(*args, **kwargs)
        finally:
            amdsmi_shut_down()

    return wrapper


@with_amdsmi_context
def _query_gcn_arch_from_amdsmi() -> str:
    """Query GCN arch from amdsmi. Raises if not available."""
    handles = amdsmi_get_processor_handles()
    if handles:
        asic_info = amdsmi_get_gpu_asic_info(handles[0])
        # Use target_graphics_version which contains the gfx name
        # e.g., 'gfx942' for MI300X/MI325X
        target_gfx = asic_info.get("target_graphics_version", "")
        # FIX: Validate amdsmi return value
        if target_gfx and target_gfx not in ["gfx0", "", None]:
            normalized_gfx = _AMDSMI_GFX_NORMALIZATION.get(target_gfx, target_gfx)
            if normalized_gfx != target_gfx:
                logger.warning_once(
                    "amdsmi returned non-standard GCN arch '%s'; "
                    "normalizing to '%s'.",
                    target_gfx,
                    normalized_gfx,
                )
            return normalized_gfx
        # If amdsmi returns invalid value, fall through to torch.cuda
        logger.debug(
            f"amdsmi returned invalid arch '{target_gfx}', falling back to torch.cuda"
        )
    # Always raise to force torch.cuda fallback if amdsmi fails or returns invalid value
    raise RuntimeError("amdsmi did not return valid GCN arch")


@with_amdsmi_context
def _query_total_memory_from_amdsmi(physical_device_id: int) -> int:
    if not _AMDSMI_AVAILABLE:
        raise RuntimeError("AMDSMI is unavailable")
    handles = amdsmi_get_processor_handles()
    handle = handles[physical_device_id]
    return amdsmi_get_gpu_memory_total(handle, AmdSmiMemoryType.VRAM)


def _query_gcn_arch_from_rocminfo() -> str:
    try:
        result = subprocess.run(
            [
                "rocminfo",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError("rocminfo did not return valid GCN arch") from e

    matches = re.findall(r"\bgfx\d+[a-z]?\b", result.stdout)
    for arch in matches:
        if arch != "gfx0":
            return arch

    raise RuntimeError("rocminfo did not return valid GCN arch")


def _get_gcn_arch() -> str:
    """
    Get GCN arch via rocminfo first, then amdsmi, then torch.cuda.
    Called once at module level; result stored in _GCN_ARCH.
    """
    try:
        arch = _query_gcn_arch_from_rocminfo()
        logger.info("Resolved GCN arch from rocminfo: %s", arch)
        return arch
    except Exception as e:
        logger.debug("Failed to get GCN arch via rocminfo: %s", e)

    try:
        return _query_gcn_arch_from_amdsmi()
    except Exception as e:
        logger.debug("Failed to get GCN arch via amdsmi: %s", e)
        logger.warning_once(
            "Failed to get GCN arch via rocminfo and amdsmi, falling back to torch.cuda. "
            "This will initialize CUDA and may cause "
            "issues if CUDA_VISIBLE_DEVICES is not set yet."
        )

    # Ultimate fallback: use torch.cuda (will initialize CUDA)
    arch = torch.cuda.get_device_properties("cuda").gcnArchName

    # Extract base architecture name (remove suffixes like ":sramecc+:xnack-")
    if ":" in arch:
        arch = arch.split(":")[0]
        logger.info(f"Extracted base architecture from torch.cuda: {arch}")

    # Validate return value - reject invalid arch names like 'gfx0'
    if not arch or arch == "gfx0" or not arch.startswith("gfx"):
        # Fallback to environment variable or error
        arch = os.environ.get("VLLM_GCN_ARCH", "")
        if not arch or not arch.startswith("gfx"):
            raise RuntimeError(
                f"Invalid GCN architecture '{arch}' from torch.cuda. "
                f"Please check your GPU driver/ROCm installation or set VLLM_GCN_ARCH environment variable."
            )
        logger.info(
            "Using GCN architecture from environment variable VLLM_GCN_ARCH=%s", arch
        )

    return arch


# Prefer an explicit env override first; otherwise initialize lazily to avoid
# blocking import-time startup on problematic ROCm setups.
_GCN_ARCH = os.environ.get("VLLM_GCN_ARCH", "")


def _refresh_gcn_flags() -> None:
    global _ON_GFX1X, _ON_MI3XX, _ON_GFX9, _ON_GFX942, _ON_GFX950
    _ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
    _ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])
    _ON_GFX9 = any(
        arch in _GCN_ARCH for arch in ["gfx906", "gfx90a", "gfx942", "gfx950"]
    )
    _ON_GFX942 = "gfx942" in _GCN_ARCH
    _ON_GFX950 = "gfx950" in _GCN_ARCH


def _ensure_gcn_arch_initialized() -> None:
    global _GCN_ARCH
    if _GCN_ARCH:
        return
    _GCN_ARCH = _get_gcn_arch()
    _refresh_gcn_flags()


_ON_GFX1X = False
_ON_MI3XX = False
_ON_GFX9 = False
_ON_GFX942 = False
_ON_GFX950 = False
_refresh_gcn_flags()


def _capability_from_gcn_arch(gcn_arch: str) -> tuple[int, int] | None:
    """
    Parse (major, minor) from a GCN arch string, mirroring how
    HIP derives hipDeviceProp_t.major / .minor.

    Format: gfx<MAJOR><MINOR><STEPPING>
      - 1-digit major  (gfx9xx):  "gfx" + M + m + stepping
      - 2-digit major  (gfx1xxx): "gfx" + MM + m + stepping

    Examples:
      gfx90a  -> (9, 0)    gfx942  -> (9, 4)    gfx950 -> (9, 5)
      gfx1100 -> (11, 0)   gfx1101 -> (11, 0)   gfx1200 -> (12, 0)

    Returns None only when the string is not gfx-prefixed at all
    (i.e. not a ROCm arch string). Raises on any string that looks
    like a GCN arch but does not match a known layout.
    """
    m = re.match(r"gfx(\d+)", gcn_arch)
    if not m:
        # Not a gfx string at all — caller should fall back to torch.cuda
        return None

    digits = m.group(1)
    n = len(digits)

    if n < 2:
        raise ValueError(
            f"GCN arch '{gcn_arch}' has too few digits ({n}) after 'gfx' "
            f"to derive a (major, minor) capability. "
            f"Please file a vLLM issue with your GPU model."
        )

    if n in (2, 3):
        # 1-digit major: gfx9 family
        # len 2: major + minor          (e.g. gfx90 from gfx90a)
        # len 3: major + minor + step   (e.g. gfx942)
        major = int(digits[0])
        minor = int(digits[1])
    elif n == 4:
        # 2-digit major: gfx10xx, gfx11xx, gfx12xx
        # major(2) + minor(1) + stepping(1)
        major = int(digits[:2])
        minor = int(digits[2])
    elif n >= 5:
        raise ValueError(
            f"GCN arch '{gcn_arch}' has {n} digits after 'gfx', which "
            f"exceeds the known 4-digit layout (MMms). Cannot determine "
            f"major/minor split unambiguously. "
            f"Please file a vLLM issue with your GPU model."
        )

    if major < 9:
        raise ValueError(
            f"Parsed unknown ROCm architecture from GCN arch '{gcn_arch}': "
            f"major={major}, minor={minor}. "
            f"Major version < 9 is not expected for any supported AMD GPU. "
            f"Please file a vLLM issue with your GPU model."
        )

    if major > 12:
        raise ValueError(
            f"Parsed unknown ROCm architecture from GCN arch '{gcn_arch}': "
            f"major={major}, minor={minor}. "
            f"Major version > 12 is beyond currently known AMD generations. "
            f"Please file a vLLM issue with your GPU model so support "
            f"can be added."
        )

    return (major, minor)


def on_gfx1x() -> bool:
    _ensure_gcn_arch_initialized()
    return _ON_GFX1X


def on_mi3xx() -> bool:
    _ensure_gcn_arch_initialized()
    return _ON_MI3XX


def on_gfx9() -> bool:
    _ensure_gcn_arch_initialized()
    return _ON_GFX9


def on_gfx942() -> bool:
    _ensure_gcn_arch_initialized()
    return _ON_GFX942


def on_gfx950() -> bool:
    _ensure_gcn_arch_initialized()
    return _ON_GFX950


def on_gfx906() -> bool:
    """检测当前GPU是否为gfx906架构"""
    _ensure_gcn_arch_initialized()
    return "gfx906" in _GCN_ARCH


def _rocm_supported_dtypes() -> list[torch.dtype]:
    """ROCm supported dtypes; shared so the ``supported_dtypes`` property
    (instance) and ``check_if_supports_dtype`` (classmethod) agree.

    gfx906 lacks native bfloat16, so returns fp16/fp32 there; other GFX9
    (gfx90a+) also include bfloat16.
    """
    _ensure_gcn_arch_initialized()
    if "gfx906" in _GCN_ARCH:
        return [torch.float16, torch.float32]
    return [torch.float16, torch.bfloat16, torch.float32]


@cache
def use_rocm_custom_paged_attention(
    qtype: torch.dtype,
    head_size: int,
    block_size: int,
    gqa_ratio: int,
    max_seq_len: int,
    sliding_window: int,
    kv_cache_dtype: str,
    alibi_slopes: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> bool:
    _ensure_gcn_arch_initialized()
    # custom paged attn always supported on V0. On V1, requires sliding window
    # disabled due to observed numerical discrepancy.
    if _ON_GFX9:
        # gfx906不支持bfloat16，需要特殊判断
        is_gfx906 = "gfx906" in _GCN_ARCH

        return (
            (sliding_window == 0 or sliding_window == (-1, -1))
            and (
                qtype == torch.half  # gfx906只用FP16
                or (qtype == torch.bfloat16 and not is_gfx906)  # 其他架构可用BF16
            )
            and (head_size == 64 or head_size == 128)
            and (block_size == 16 or block_size == 32)
            and (gqa_ratio >= 1 and gqa_ratio <= 16)
            and max_seq_len <= 128 * 1024
            and (envs.VLLM_ROCM_CUSTOM_PAGED_ATTN)
            and sinks is None
        )

    else:
        return (
            _ON_GFX1X
            and (sliding_window == 0 or sliding_window == (-1, -1))
            and (qtype == torch.half or qtype == torch.bfloat16)
            and head_size == 128
            and block_size == 16
            and (gqa_ratio >= 3 and gqa_ratio <= 16)
            and max_seq_len <= 128 * 1024
            and alibi_slopes is None
            and kv_cache_dtype == "auto"
            and envs.VLLM_ROCM_CUSTOM_PAGED_ATTN
            and sinks is None
        )


@cache
def flash_attn_triton_available() -> bool:
    _ensure_gcn_arch_initialized()
    if not on_gfx1x():
        return False
    try:
        from importlib.util import find_spec

        if find_spec("flash_attn") is None:
            return False
        if find_spec("flash_attn.flash_attn_triton_amd") is None:
            return False
        if os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE") != "TRUE":
            logger.info_once(
                "Set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE to enable "
                "Flash Attention Triton backend on RDNA."
            )
            return False
        return True
    except ImportError:
        return False


def _get_backend_priorities(
    use_mla: bool,
    use_sparse: bool,
) -> list[AttentionBackendEnum]:
    from vllm._aiter_ops import rocm_aiter_ops

    if use_sparse:
        return [AttentionBackendEnum.ROCM_AITER_MLA_SPARSE]

    if use_mla:
        if rocm_aiter_ops.is_mla_enabled():
            backends = [
                AttentionBackendEnum.ROCM_AITER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.ROCM_AITER_TRITON_MLA,
            ]
            if envs.VLLM_ROCM_USE_GFX906_TRITON_PRIORITY and on_gfx906():
                logger.info_once(
                    "VLLM_ROCM_USE_GFX906_TRITON_PRIORITY enabled: preferring "
                    "TRITON_MLA first on gfx906."
                )
                backends.remove(AttentionBackendEnum.TRITON_MLA)
                backends.insert(0, AttentionBackendEnum.TRITON_MLA)
            return backends
        else:
            return [
                AttentionBackendEnum.TRITON_MLA,
            ]

    backends = []

    # Priority 1: Check for AITER Unified Attention (must check before MHA)
    if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:
        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)

    # Priority 2: Check for AITER MHA (Flash Attention)
    if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MHA:
        backends.append(AttentionBackendEnum.ROCM_AITER_FA)

    # Priority 3: Check for ROCM_ATTN (prefill-decode split)
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if (
        vllm_config is not None
        and vllm_config.attention_config.use_prefill_decode_attention
    ):
        backends.append(AttentionBackendEnum.ROCM_ATTN)

    # Default: Triton Unified Attention
    backends.append(AttentionBackendEnum.TRITON_ATTN)

    if envs.VLLM_ROCM_USE_GFX906_TRITON_PRIORITY and on_gfx906():
        logger.info_once(
            "VLLM_ROCM_USE_GFX906_TRITON_PRIORITY enabled: preferring "
            "TRITON_ATTN first on gfx906."
        )
        backends.remove(AttentionBackendEnum.TRITON_ATTN)
        backends.insert(0, AttentionBackendEnum.TRITON_ATTN)

    return backends


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    # rocm shares the same device control env var as CUDA
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    ]
    supported_quantization: list[str] = [
        "awq",
        "awq_marlin",  # will be overwritten with awq
        "gptq",
        "gptq_marlin",  # will be overwritten with gptq
        "fp8",
        "compressed-tensors",
        "fbgemm_fp8",
        "gguf",
        "quark",
        "ptpc_fp8",
        "mxfp4",
        "petit_nvfp4",
        "torchao",
    ]
    if not _ON_GFX9:
        supported_quantization.append("bitsandbytes")

    @classmethod
    def import_kernels(cls) -> None:
        """Import ROCm-specific kernels."""
        super().import_kernels()

        import contextlib

        # Import ROCm-specific extension
        with contextlib.suppress(ImportError):
            import vllm._rocm_C  # noqa: F401

    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]:
        valid_backends_priorities = []
        invalid_reasons = {}

        backend_priorities = _get_backend_priorities(
            use_mla,
            use_sparse,
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    head_size=head_size,
                    dtype=dtype,
                    kv_cache_dtype=kv_cache_dtype,
                    block_size=block_size,
                    use_mla=use_mla,
                    has_sink=has_sink,
                    use_sparse=use_sparse,
                    attn_type=attn_type,
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = invalid_reasons_i
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
    ) -> str:
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=device_capability,
                    head_size=head_size,
                    dtype=dtype,
                    kv_cache_dtype=kv_cache_dtype,
                    block_size=block_size,
                    use_mla=use_mla,
                    has_sink=has_sink,
                    use_sparse=use_sparse,
                    attn_type=attn_type,
                )
            except ImportError:
                invalid_reasons = ["ImportError"]
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return selected_backend.get_path()

        # No selected backend or the selected backend is invalid,
        # so we try finding a valid backend.
        valid_backends_priorities, invalid_reasons = cls.get_valid_backends(
            device_capability=device_capability,
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            attn_type=attn_type,
        )
        reasons_str = (
            "{"
            + ", ".join(
                f"{backend.name}: [{', '.join(reasons)}]"
                for backend, reasons in invalid_reasons.items()
            )
            + "}"
        )
        config_str = (
            f"head_size={head_size}, dtype={dtype}, "
            f"kv_cache_dtype={kv_cache_dtype}, block_size={block_size}, "
            f"use_mla={use_mla}, has_sink={has_sink}, "
            f"use_sparse={use_sparse}, attn_type={attn_type}"
        )
        logger.debug_once(
            f"Some attention backends are not valid for {cls.device_name} with "
            f"{config_str}. Reasons: {reasons_str}."
        )
        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )

        # We have found some valid backends. Select the one with the
        # highest priority.
        sorted_indices = sorted(
            range(len(valid_backends_priorities)),
            key=lambda i: valid_backends_priorities[i][1],
        )
        selected_index = sorted_indices[0]
        selected_backend = valid_backends_priorities[selected_index][0]
        logger.info_once(
            "Using %s attention backend out of potential backends: %s.",
            selected_backend.name,
            "[" + ", ".join(f"'{b[0].name}'" for b in valid_backends_priorities) + "]",
            scope="local",
        )

        return selected_backend.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        # Import V0 version for ViT models (used by qwen3_vl.py, etc.)
        from vllm.attention.backends.registry import (
            AttentionBackendEnum as ViTAttentionBackendEnum,
        )

        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        from importlib.util import find_spec

        from vllm._aiter_ops import rocm_aiter_ops

        if rocm_aiter_ops.is_enabled() and on_gfx9():
            logger.info_once("Using AITER Flash Attention backend for ViT model.")
            return ViTAttentionBackendEnum.ROCM_AITER_FA

        # gfx906 has limited flash-attn support, use TORCH_SDPA instead
        # Only use flash-attn for gfx942+/gfx950+, not gfx906
        if (
            (on_gfx942() or on_gfx950())  # gfx90a equivalent: newer gfx9 arch
            and find_spec("flash_attn") is not None
            and (dtype == torch.float16 or dtype == torch.bfloat16)
        ):
            logger.info_once("Using Flash Attention backend for ViT model.")
            return ViTAttentionBackendEnum.FLASH_ATTN

        # RDNA3/RDNA4 (gfx11xx/gfx12xx): Use Flash Attention Triton backend
        if (
            on_gfx1x()
            and flash_attn_triton_available()
            and (dtype == torch.float16 or dtype == torch.bfloat16)
        ):
            logger.info_once(
                "Using Flash Attention (Triton backend) for ViT model on RDNA."
            )
            return ViTAttentionBackendEnum.FLASH_ATTN

        # RDNA3/RDNA4 (gfx11xx/gfx12xx): Use Flash Attention Triton backend
        if (
            on_gfx1x()
            and flash_attn_triton_available()
            and (dtype == torch.float16 or dtype == torch.bfloat16)
        ):
            logger.info_once(
                "Using Flash Attention (Triton backend) for ViT model on RDNA."
            )
            return ViTAttentionBackendEnum.FLASH_ATTN

        logger.info_once("Using Torch SDPA backend for ViT model.")
        return ViTAttentionBackendEnum.TORCH_SDPA

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        """Supported dtypes for the current ROCm device.

        gfx906硬件不支持bfloat16，因此只返回float16和float32。
        其他GFX9架构(gfx90a+)支持bfloat16。

        Property (not classmethod) to match the Platform base contract;
        access as ``current_platform.supported_dtypes`` without parentheses.
        """
        return _rocm_supported_dtypes()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        _ensure_gcn_arch_initialized()
        cap = _capability_from_gcn_arch(_GCN_ARCH)
        if cap is not None:
            return DeviceCapability(major=cap[0], minor=cap[1])

        logger.warning_once(
            "Could not derive device capability from GCN arch '%s', "
            "falling back to torch.cuda (this will initialize CUDA).",
            _GCN_ARCH,
        )
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @with_amdsmi_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        Query if the set of gpus are fully connected by xgmi (1 hop)
        """
        handles = [amdsmi_get_processor_handles()[i] for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        link_type = amdsmi_topo_get_link_type(handle, peer_handle)
                        # type is 2 for XGMI
                        if link_type["hops"] != 1 or link_type["type"] != 2:
                            return False
                    except AmdSmiException as error:
                        logger.error("AMD 1 hop XGMI detection failed.", exc_info=error)
                        return False
        return True

    @classmethod
    @with_amdsmi_context
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        if not _AMDSMI_AVAILABLE:
            return torch.cuda.get_device_name(device_id)
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        asic_info = amdsmi_get_gpu_asic_info(handle)
        device_name: str = asic_info["device_id"]
        if device_name in _ROCM_DEVICE_ID_NAME_MAP:
            return _ROCM_DEVICE_ID_NAME_MAP[device_name]
        return asic_info["market_name"]

    @classmethod
    @with_amdsmi_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        if not _AMDSMI_AVAILABLE:
            return ""
        physical_device_id = cls.device_id_to_physical_device_id(device_id)

        try:
            handles = amdsmi_get_processor_handles()
            handle = handles[physical_device_id]
        except (AmdSmiException, IndexError) as error:
            logger.error("amdsmi device query failed", exc_info=error)
            return ""

        try:
            return amdsmi_get_gpu_device_uuid(handle)
        except AmdSmiException as error:
            logger.error("amdsmi device uuid query failed", exc_info=error)
            return ""

    @classmethod
    @with_amdsmi_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        if not _AMDSMI_AVAILABLE:
            device_props = torch.cuda.get_device_properties(device_id)
            return device_props.total_memory
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handles = amdsmi_get_processor_handles()
        if physical_device_id < len(handles):
            return amdsmi_get_gpu_memory_total(
                handles[physical_device_id], AmdSmiMemoryType.VRAM
            )

        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.config.compilation import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        is_eager_execution = compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        use_aiter_fused_moe = rocm_aiter_ops.is_fused_moe_enabled()
        use_aiter_rms_norm = rocm_aiter_ops.is_rmsnorm_enabled()
        use_aiter_fp8_linear = rocm_aiter_ops.is_linear_fp8_enabled()
        use_aiter_fused_se = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        #  Aiter rms norm perform best when CUDA Graph capture is enabled.
        if (
            use_aiter_rms_norm
            and not is_eager_execution
            and "-rms_norm" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+rms_norm")

        if use_aiter_fp8_linear and "-quant_fp8" not in compilation_config.custom_ops:
            compilation_config.custom_ops.append("+quant_fp8")

        if use_aiter_fused_se and "-grouped_topk" in compilation_config.custom_ops:
            logger.warning_once(
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled, which "
                "requires the 'grouped_topk' custom op. Overriding the "
                "user-provided '-grouped_topk'."
            )
            compilation_config.custom_ops.remove("-grouped_topk")
        # Ensure grouped_topk is always enabled when using AITER if
        # its not disabled by user
        if (
            use_aiter_fused_moe
            and "+grouped_topk" not in compilation_config.custom_ops
            and "-grouped_topk" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+grouped_topk")
        # Enable rotary embedding customop when using AITER if not disabled by user
        if (
            rocm_aiter_ops.is_enabled()
            and "+rotary_embedding" not in compilation_config.custom_ops
            and "-rotary_embedding" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+rotary_embedding")

        # Default dispatch to rocm's sparse_attn_indexer implementation
        compilation_config.custom_ops.append("+sparse_attn_indexer")

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config.compilation import CUDAGraphMode

        cache_config = vllm_config.cache_config
        compilation_config = vllm_config.compilation_config
        parallel_config = vllm_config.parallel_config

        if compilation_config.cudagraph_mode.has_full_cudagraphs():
            # decode context parallel does not support full cudagraphs
            if parallel_config.decode_context_parallel_size > 1:
                logger.warning_once(
                    "Decode context parallel (DCP) is enabled, which is "
                    "incompatible with full CUDA graphs. "
                    "Overriding cudagraph_mode to PIECEWISE."
                )
                compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
            # prefill context parallel do not support full cudagraphs
            elif parallel_config.prefill_context_parallel_size > 1:
                logger.warning_once(
                    "Prefill context parallel (PCP) is enabled, which is "
                    "incompatible with full CUDA graphs. "
                    "Overriding cudagraph_mode to PIECEWISE."
                )
                compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

        if cache_config and cache_config.block_size is None:
            if (
                envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION and envs.VLLM_ROCM_USE_AITER
                # NOTE: This block has been deprecated
                # or get_env_variable_attn_backend()
                # == AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN
                # TODO: monitor https://github.com/vllm-project/vllm/pull/30396
                # to see how we can transition to the new way of selecting
                # attention backends
            ):
                cache_config.block_size = 64
                logger.warning(
                    "[ROCM_AITER_UNIFIED_ATTTN]: Setting kv cache block size to 64."
                )
            else:
                cache_config.block_size = 16

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        if model_arch in _ROCM_UNSUPPORTED_MODELS:
            raise ValueError(
                f"Model architecture '{model_arch}' is not supported by ROCm for now."
            )

        if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
            msg = _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch]
            logger.warning(
                "Model architecture '%s' is partially supported by ROCm: %s",
                model_arch,
                msg,
            )

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        _ensure_gcn_arch_initialized()
        super().verify_quantization(quant)

        # gfx906不支持bitsandbytes检查
        if quant == "bitsandbytes" and "gfx906" in _GCN_ARCH:
            raise ValueError(
                f"Quantization '{quant}' is not supported on gfx906 architecture "
                "due to warp size 64 limitation. "
                "Please use alternative quantization methods such as awq or gptq."
            )

        if quant == "awq" and not envs.VLLM_USE_TRITON_AWQ:
            logger.warning(
                "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
                " is not set, enabling VLLM_USE_TRITON_AWQ."
            )
        os.environ["VLLM_USE_TRITON_AWQ"] = "1"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        # Flush the caching allocator first so freed-but-cached blocks (e.g.
        # the draft model's embed_tokens/lm_head copies dropped by MTP
        # sharing) are not reported as in use, mirroring cuda.py's
        # empty_cache() before measuring.
        #
        # Use the driver view from mem_get_info instead of amdsmi: on this
        # gfx906 setup the amdsmi VRAM counter does not decrement when
        # empty_cache() releases blocks back to the driver (measured
        # 2026-08-22: after freeing 1 GiB and calling empty_cache, amdsmi
        # kept reporting it, then accumulated to a 9 GiB high-water mark
        # while mem_get_info tracked every transition exactly), which both
        # overstated weights memory and left stale readings. The driver view
        # stays device-level, which MemorySnapshot needs to derive non-torch
        # memory (gfx906 custom kernels allocate outside the torch
        # allocator). Do not reset peak stats here: MemorySnapshot reads
        # torch peak counters across snapshots.
        device_obj = (
            torch.device(device) if device is not None else torch.device("cuda:0")
        )
        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info(device_obj)
        return total_mem - free_mem

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa
        )

    @classmethod
    def supports_mx(cls) -> bool:
        _ensure_gcn_arch_initialized()
        return any(gfx in _GCN_ARCH for gfx in ["gfx95"])

    @classmethod
    def supports_fp8(cls) -> bool:
        _ensure_gcn_arch_initialized()
        return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95", "gfx12"])

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        _ensure_gcn_arch_initialized()
        return "gfx94" in _GCN_ARCH

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.is_fp8_fnuz():
            return torch.float8_e4m3fnuz
        else:
            return torch.float8_e4m3fn

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        # We only enable custom allreduce for MI300 series
        _ensure_gcn_arch_initialized()
        return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def is_navi(cls) -> bool:
        _ensure_gcn_arch_initialized()
        return "gfx1" in _GCN_ARCH

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: PrefixStore,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> ProcessGroup:
        assert is_nccl_available()
        pg: ProcessGroup = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        backend_options = ProcessGroupNCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupNCCL(
            prefix_store, group_rank, group_size, backend_options
        )
        backend_type = ProcessGroup.BackendType.NCCL
        device = torch.device("cuda")
        pg._set_default_backend(backend_type)
        backend_class._set_sequence_number_for_group()

        pg._register_backend(device, backend_type, backend_class)
        return pg

    @classmethod
    def device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if on_gfx906() and dtype == torch.bfloat16:
            gpu_name = cls.get_device_name()
            raise ValueError(
                f"Dtype {dtype} is not supported on {gpu_name} (gfx906). "
                "gfx906 does not natively support bfloat16, and some model "
                "configs default to bfloat16. Please launch with "
                "--dtype=half or --dtype=float16 to avoid float32 fallback "
                "and ROCm instability on gfx906."
            )

        # Classmethod cannot read a @property off `cls`; use the shared helper.
        supported = _rocm_supported_dtypes()
        if dtype not in supported:
            gpu_name = cls.get_device_name()
            raise ValueError(
                f"Dtype {dtype} is not supported on {gpu_name}. "
                f"Supported dtypes are: {supported}. "
                f"Please use --dtype=half for float16."
            )

        # 原有的capability检查保留
        if dtype == torch.bfloat16:  # noqa: SIM102
            if not cls.has_device_capability(80):
                capability = cls.get_device_capability()
                gpu_name = cls.get_device_name()

                if capability is None:
                    compute_str = "does not have a compute capability"
                else:
                    version_str = capability.as_version_str()
                    compute_str = f"has compute capability {version_str}"

                raise ValueError(
                    "Bfloat16 is only supported on GPUs "
                    "with compute capability of at least 8.0. "
                    f"Your {gpu_name} GPU {compute_str}. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half."
                )

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        _ensure_gcn_arch_initialized()
        return "gfx906" not in _GCN_ARCH

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return True
