# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger
import vllm.envs as envs
from vllm.v1.kv_cache_interface import KVQuantMode


def vllm_is_batch_invariant() -> bool:
    return bool(envs.VLLM_BATCH_INVARIANT)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
is_batch_invariant = vllm_is_batch_invariant()
float8_info = torch.finfo(current_platform.fp8_dtype())

# gfx906 decode scheduling knobs, migrated from the legacy attention op path.
# On gfx906 the tuning applies unconditionally; the env var is only an
# emergency kill switch for debugging.
ENABLE_GFX906_ATTN_SCHED_TUNING = os.getenv(
    "VLLM_GFX906_TRITON_ATTN_TUNING", "1"
).lower() in {"1", "true", "yes", "on"}

# Route tuned prefill shapes through the split-KV (3D) kernel on gfx906.
# Independent switch so prefill can be routed independently of the decode
# policy. Enabled by default: interleaved process-level A/B on a 27B
# head-256/GQA6 GGUF model shows +8% end-to-end prefill throughput at 64k
# prompts (growing from ~1-3% at 4k-16k shapes) with greedy outputs
# bit-identical to the 2D path. The path costs up to ~0.4 GiB of lazily
# allocated fp32 segment workspace at the default chunk size; verified to
# fit at gpu-memory-utilization=0.95 with 132k context. Set the env var to
# 0 to restore the 2D-only prefill. Only applies to plain causal attention
# (no sliding window, sinks, alibi, softcap, qq-bias, mm-prefix or fused
# fp8 output) on the tuned head-256/GQA6 shape, and never during CUDA-graph
# capture (the workspace is allocated lazily on first use).
ENABLE_GFX906_ATTN_PREFILL_3D = os.getenv(
    "VLLM_GFX906_TRITON_ATTN_PREFILL_3D", "1"
).lower() in {"1", "true", "yes", "on"}

# Route the plain-causal prefill 3D path through a specialized slim kernel
# on gfx906. Same math and launch grid as kernel_unified_attention_3d
# (bitwise-identical outputs verified at 32k/64k/128k contexts on the
# head-256/GQA6 shape), with three structural changes that measure
# 1.46-1.86x kernel-level on long-context prefill shapes:
#   - two-phase causal loop: tiles fully below the block's causal diagonal
#     skip mask construction and masked loads entirely;
#   - int32 cache-offset arithmetic when the caches fit below 2^31 elements
#     (halves the offset-vector register footprint);
#   - scalar per-tile page-index load when BLOCK_SIZE % TILE_SIZE == 0 (a
#     32-token tile never straddles an 800-token page, so the per-tile
#     physical-block gather reduces to one scalar load).
# Additional gates beyond the 3D path: exact GQA packing
# (num_query_heads == num_kv_heads * num_queries_per_kv), power-of-two
# head_size, and innermost cache strides of 1. Set to 0 to restore the
# generic 3D kernel.
ENABLE_GFX906_ATTN_PREFILL_SLIM = os.getenv(
    "VLLM_GFX906_TRITON_ATTN_PREFILL_SLIM", "1"
).lower() in {"1", "true", "yes", "on"}

# Route the gated slim prefill path through a handwritten HIP kernel on
# gfx906 (csrc/attention/gfx906_prefill_attn.cu). Same segment-workspace
# contract and online-softmax semantics as the Triton slim scanner with
# fp32 accumulation over fp16 KV throughout; kernel-level it measures
# 1.75-2.4x faster than the Triton scanner at 16k-131k contexts on the
# head-256/GQA6 shape, with rel_rms vs an fp32 reference at the same 2.7e-4
# level as Triton. Single sequence only — multi-sequence batches, other
# shapes, or any non-default attention option fall back to the Triton
# path. Set 0 to restore the Triton slim scanner.
ENABLE_GFX906_ATTN_PREFILL_HIP = os.getenv(
    "VLLM_GFX906_ATTN_PREFILL_HIP", "1"
).lower() in {"1", "true", "yes", "on"}

# Debug readout for the handwritten scanner's address guards: the kernel
# clamps stale block-table/seq-lens inputs instead of faulting and counts
# the events; with this env set the dispatch syncs and logs the counters.
ENABLE_GFX906_HIP_ATTN_GUARD = os.getenv("VLLM_GFX906_HIP_ATTN_GUARD", "0") in {
    "1", "true", "yes", "on"}
_hip_dbg_buf: torch.Tensor | None = None

# Upper bound for split-KV segments the decode policy may request. The
# backend sizes its softmax segment buffers with this value on gfx906.
GFX906_MAX_DECODE_SEGMENTS = 128

# Split-KV segments for the gfx906 prefill 3D path. Prefill already fills the
# GPU with q-block x kv-head CTAs; a small fixed split balances the causal
# load skew between early and late query blocks. Interleaved sweeps over
# segments {2,4,8,16} at 4k-16k serving shapes put 2 and 4 within ~1% of
# each other (8+ regress); 2 halves the fp32 segment workspace, which keeps
# gpu_memory_utilization=0.95 runs within budget at the default chunk size.
GFX906_PREFILL_SEGMENTS = 2

# Route speculative-decode forwards (target verify + draft steps) through
# the split-KV path on gfx906. Those forwards carry only a handful of query
# tokens per sequence but scan the full decode context; they arrive with
# max_seqlen_q > 1, which used to push them into the tuned-prefill 3D route
# (fixed 2 segments) or, under CUDA-graph capture, the no-split 2D fallback.
# Either way the grid is ~q_blocks x kv_heads x <=2 CTAs and a 60-CU gfx906
# streams the context KV at a few percent of achievable bandwidth, which is
# what makes MTP decode collapse at long contexts (27B Q6 MTP=2: 20.8 tok/s
# at 4k falling to 2.9 at 64k while draft acceptance stays ~93%; profiled
# as kernel_unified_attention_3d_slim at ~14-43 GB/s per call). The spec
# path reuses the backend-provided decode segment buffers with a fixed
# segment count so warmup, capture and replay share one launch
# configuration (graph capture runs with seq_lens=1, freezing any
# context-derived count at the capture-time context). Set to 0 to restore
# the prefill-3D / 2D routing.
ENABLE_GFX906_ATTN_SPEC_SPLIT_KV = os.getenv(
    "VLLM_GFX906_TRITON_ATTN_SPEC_SPLIT_KV", "1"
).lower() in {"1", "true", "yes", "on"}
# Query-tokens-per-sequence ceiling for treating a max_seqlen_q > 1 batch
# as decode-like; spec-decode steps schedule num_spec_tokens + 1 or fewer.
GFX906_SPEC_MAX_QUERY_LEN = 8
# Fixed split-KV segments for the spec path, bounded by the decode segment
# buffer capacity the backend allocates (GFX906_MAX_DECODE_SEGMENTS).
GFX906_SPEC_ATTN_SEGMENTS = 128

# Lazily grown workspace for the prefill split-KV path. Unlike the decode
# segment buffers (sized for the decode batch threshold and provided by the
# backend), prefill needs one row per scheduled token, so the workspace is
# allocated on first use and grown to the largest batch seen. Allocation
# happens outside CUDA-graph capture only, and never runs before KV-cache
# profiling (the profiling forward passes attention metadata as None).
_prefill_segm_state: dict | None = None


def _prefill_segment_buffers(
    num_tokens: int,
    num_query_heads: int,
    num_segments: int,
    headdim_padded: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _prefill_segm_state
    rows = triton.next_power_of_2(num_tokens)
    st = _prefill_segm_state
    if (
        st is None
        or st["rows"] < rows
        or st["heads"] != num_query_heads
        or st["segs"] != num_segments
        or st["dim"] != headdim_padded
        or st["device"] != device
    ):
        _prefill_segm_state = {
            "rows": rows,
            "heads": num_query_heads,
            "segs": num_segments,
            "dim": headdim_padded,
            "device": device,
            "output": torch.empty(
                (rows, num_query_heads, num_segments, headdim_padded),
                dtype=torch.float32,
                device=device,
            ),
            "max": torch.empty(
                (rows, num_query_heads, num_segments),
                dtype=torch.float32,
                device=device,
            ),
            "expsum": torch.empty(
                (rows, num_query_heads, num_segments),
                dtype=torch.float32,
                device=device,
            ),
        }
        st = _prefill_segm_state
    return st["output"], st["max"], st["expsum"]


@lru_cache(maxsize=1)
def _is_gfx906_rocm() -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_rocm()
        and capability is not None
        and capability.major == 9
        and capability.minor == 0
    )


def _gfx906_sched_tuning_enabled() -> bool:
    return ENABLE_GFX906_ATTN_SCHED_TUNING and _is_gfx906_rocm()


@lru_cache(maxsize=1)
def _log_hip_prefill_scan_once() -> bool:
    logger.info(
        "gfx906 handwritten HIP prefill attention scan enabled "
        "(single-sequence fp16 head-256/GQA6/page-800 shape)"
    )
    return True


@lru_cache(maxsize=4)
def _log_hip_prefill_miss_once(num_seqs: int, q_dtype: str, k_dtype: str,
                               head_size: int, num_query_heads: int,
                               num_kv_heads: int, block_size: int,
                               q_contig: bool, k_strides: tuple,
                               v_strides: tuple, bt_dtype: str, bt_stride1: int,
                               op_present: bool, slim: bool) -> bool:
    logger.info(
        "gfx906 HIP prefill scan skipped: slim=%s op=%s num_seqs=%d "
        "q=%s k=%s hs=%d qh=%d kvh=%d bs=%d qcont=%s kstrides=%s vstrides=%s "
        "bt=%s btstride1=%d", slim, op_present, num_seqs, q_dtype, k_dtype,
        head_size, num_query_heads, num_kv_heads, block_size, q_contig,
        k_strides, v_strides, bt_dtype, bt_stride1)
    return True


def gfx906_decode_segments_capacity() -> int:
    """Segment-buffer capacity the backend should allocate on this platform."""
    return GFX906_MAX_DECODE_SEGMENTS if _gfx906_sched_tuning_enabled() else 0


def _decode_num_warps(
    head_size: int, max_seqlen_k: int, num_queries_per_kv: int = 0
) -> int:
    # Wide-head decodes used to run 4 warps once the KV scan was long
    # enough, but on the one-token-per-block shape (the 8-row GQA6-256
    # blocks from _decode_block_m) the 8x16 tiles cannot feed 4 warps and
    # the extra warps only add scheduling overhead: 2 warps measure ~1.7x
    # faster at every context length (116 vs 206us at 4k, 313 vs 530us at
    # 16k, 1140 vs 2009us at 64k per call through the decode dispatch).
    # Other wide-head shapes keep the previous 4-warp choice.
    if (
        head_size > 128
        and max_seqlen_k >= 1536
        and _decode_block_m(head_size, num_queries_per_kv) is None
    ):
        return 4
    return 2


def _decode_block_m(head_size: int, num_queries_per_kv: int) -> int | None:
    # GQA6 with 256-dim heads: one query token spans exactly 6 rows, so an
    # 8-row block covers a single token without padding a second one.
    if head_size == 256 and num_queries_per_kv == 6:
        return 8
    return None


def _prefill_block_m(head_size: int, num_queries_per_kv: int) -> int | None:
    # GQA6 with 256-dim heads: a 32-row block spans 5 query tokens (30 of 32
    # rows carry work) and reuses each K/V tile across them, versus only 2
    # tokens per 16-row block. Sweeps over BLOCK_M/TILE/num_warps/num_stages
    # at 4k-16k prefill shapes favor 32/32/4/1 uniformly on gfx906.
    if head_size == 256 and num_queries_per_kv == 6:
        return 32
    return None


def _num_query_blocks(num_query_tokens: int, num_seqs: int, block_q: int) -> int:
    # Exact block count for single-sequence batches. For multiple sequences,
    # retain the mapping gaps expected by find_seq_idx and use the safe
    # upper bound without realizing query_lens on the CPU.
    if num_seqs == 1:
        return (num_query_tokens + block_q - 1) // block_q
    return num_query_tokens // block_q + num_seqs


def _decode_num_segments(
    max_seqlen_k: int,
    num_query_blocks: int,
    num_kv_heads: int,
    num_queries_per_kv: int,
) -> int:
    # Keep short-context decode at 16 segments, but increase split-KV
    # parallelism on gfx906 once the KV scan is large enough to amortize
    # the extra reduction work.
    if max_seqlen_k < 2048:
        return 16

    base_grid_size = max(num_query_blocks * num_kv_heads, 1)
    target_grid_size = 1024 if num_queries_per_kv <= 2 else 512
    if max_seqlen_k >= 8192:
        target_grid_size = max(target_grid_size, 1024)
    if max_seqlen_k >= 16384 and num_queries_per_kv == 1:
        target_grid_size = 2048
    if max_seqlen_k >= 24576:
        target_grid_size = 4096 if num_queries_per_kv == 1 else 2048

    segments = (target_grid_size + base_grid_size - 1) // base_grid_size
    return min(128, max(16, triton.next_power_of_2(segments)))


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        # Keep softmax weights representable when the P @ V dot uses FP8
        # tensor cores. Undo the power-of-two scale after the dot.
        if V.dtype == tl.float8e4nv:
            p_scale = 256.0
        elif V.dtype.is_fp8():
            p_scale = 128.0
        else:
            p_scale = 1.0
        acc += tl.dot((P * p_scale).to(V.dtype), V) / p_scale

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        # Keep softmax weights representable when the P @ V dot uses FP8
        # tensor cores. Undo the power-of-two scale after the dot.
        if V.dtype == tl.float8e4nv:
            p_scale = 256.0
        elif V.dtype.is_fp8():
            p_scale = 128.0
        else:
            p_scale = 1.0
        acc += tl.dot((P * p_scale).to(V.dtype), V) / p_scale

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def kernel_unified_attention_3d_slim(
    segm_output_ptr,  # [num_tokens, num_query_heads, num_segments, head_size] fp32
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size//x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    query_start_len_ptr,  # [num_seqs+1]
    scale,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int (== 1 on the gated shape)
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int (== 1 on the gated shape)
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int, power of 2 (== padded; gate guarantees)
    BLOCK_Q: tl.constexpr,  # int
    num_seqs,  # int
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    OFFSETS_I32: tl.constexpr,  # bool: cache offsets computed in int32
    SCALAR_PHYS: tl.constexpr,  # bool: tile never straddles a page
):
    # Specialized plain-causal prefill variant of kernel_unified_attention_3d.
    # The launcher gates it to shapes where every feature branch of the
    # generic kernel is compile-time dead, and verifies bitwise-identical
    # outputs against it.
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    # exact GQA packing is a gate: num_kv_heads * num_queries_per_kv covers
    # every (token, query head) row, so no query_mask_1 is needed
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )
    query_mask_0 = query_pos < cur_batch_query_len

    # Q : (BLOCK_M, HEAD_SIZE), rows past the query length read as 0
    Q = tl.load(query_ptr + query_offset, mask=query_mask_0[:, None], other=0.0)

    if OFFSETS_I32:
        bt_stride = block_table_stride.to(tl.int32)
        sk0 = stride_k_cache_0.to(tl.int32)
        sk1 = stride_k_cache_1.to(tl.int32)
        sk2 = stride_k_cache_2.to(tl.int32)
        sv0 = stride_v_cache_0.to(tl.int32)
        sv1 = stride_v_cache_1.to(tl.int32)
        sv2 = stride_v_cache_2.to(tl.int32)
    else:
        bt_stride = block_table_stride
        sk0 = stride_k_cache_0
        sk1 = stride_k_cache_1
        sk2 = stride_k_cache_2
        sv0 = stride_v_cache_0
        sv1 = stride_v_cache_1
        sv2 = stride_v_cache_2
    block_table_offset = seq_idx * bt_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = tl.minimum(
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1,
        seq_len,
    )
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    q_lo = q_block_local_idx * BLOCK_Q

    seg_lo = segm_idx * tiles_per_segment
    seg_hi = tl.minimum((segm_idx + 1) * tiles_per_segment, num_tiles)
    # tiles strictly below the block's causal diagonal are fully visible to
    # every real row and fully in-bounds: no masks of any kind needed there
    full_end = (context_len + q_lo) // TILE_SIZE

    # ---- phase A: fully-unmasked tiles ----
    for j in range(seg_lo, tl.minimum(seg_hi, full_end)):
        seq_offset = j * TILE_SIZE + offs_t
        if SCALAR_PHYS:
            page_idx = (j * TILE_SIZE) // BLOCK_SIZE
            pbi_s = tl.load(block_tables_ptr + block_table_offset + page_idx)
            if not OFFSETS_I32:
                pbi_s = pbi_s.to(tl.int64)
            in_page = seq_offset - page_idx * BLOCK_SIZE

            k_offset = (
                pbi_s * sk0
                + kv_head_idx * sk2
                + offs_d[:, None] * stride_k_cache_3
                + in_page[None, :] * sk1
            )
            v_offset = (
                pbi_s * sv0
                + kv_head_idx * sv2
                + offs_d[None, :] * stride_v_cache_3
                + in_page[:, None] * sv1
            )
        else:
            pbi = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            )
            if OFFSETS_I32:
                physical_block_idx = pbi
            else:
                physical_block_idx = pbi.to(tl.int64)

            k_offset = (
                physical_block_idx[None, :] * sk0
                + kv_head_idx * sk2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * sk1
            )
            v_offset = (
                physical_block_idx[:, None] * sv0
                + kv_head_idx * sv2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * sv1
            )

        K = tl.load(key_cache_ptr + k_offset)
        V = tl.load(value_cache_ptr + v_offset)

        S = scale * tl.dot(Q, K)

        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        L = L * alpha + l_j
        acc = acc * alpha[:, None]
        # Keep softmax weights representable when the P @ V dot uses FP8
        # tensor cores. Undo the power-of-two scale after the dot.
        if V.dtype == tl.float8e4nv:
            p_scale = 256.0
        elif V.dtype.is_fp8():
            p_scale = 128.0
        else:
            p_scale = 1.0
        acc += tl.dot((P * p_scale).to(V.dtype), V) / p_scale
        M = m_j

    # ---- phase B: diagonal band, masked (generic path) ----
    for j in range(tl.maximum(seg_lo, full_end), seg_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        if SCALAR_PHYS:
            page_idx = (j * TILE_SIZE) // BLOCK_SIZE
            pbi_s = tl.load(block_tables_ptr + block_table_offset + page_idx)
            if not OFFSETS_I32:
                pbi_s = pbi_s.to(tl.int64)
            in_page = seq_offset - page_idx * BLOCK_SIZE

            k_offset = (
                pbi_s * sk0
                + kv_head_idx * sk2
                + offs_d[:, None] * stride_k_cache_3
                + in_page[None, :] * sk1
            )
            v_offset = (
                pbi_s * sv0
                + kv_head_idx * sv2
                + offs_d[None, :] * stride_v_cache_3
                + in_page[:, None] * sv1
            )
        else:
            pbi = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            )
            if OFFSETS_I32:
                physical_block_idx = pbi
            else:
                physical_block_idx = pbi.to(tl.int64)

            k_offset = (
                physical_block_idx[None, :] * sk0
                + kv_head_idx * sk2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * sk1
            )
            v_offset = (
                physical_block_idx[:, None] * sv0
                + kv_head_idx * sv2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * sv1
            )

        K = tl.load(key_cache_ptr + k_offset, mask=tile_mask[None, :], other=0.0)
        V = tl.load(value_cache_ptr + v_offset, mask=tile_mask[:, None], other=0.0)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        S = scale * tl.dot(Q, K)
        S = tl.where(query_mask_0[:, None] & seq_mask, S, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        L = L * alpha + l_j
        acc = acc * alpha[:, None]
        # Keep softmax weights representable when the P @ V dot uses FP8
        # tensor cores. Undo the power-of-two scale after the dot.
        if V.dtype == tl.float8e4nv:
            p_scale = 256.0
        elif V.dtype.is_fp8():
            p_scale = 128.0
        else:
            p_scale = 1.0
        acc += tl.dot((P * p_scale).to(V.dtype), V) / p_scale
        M = m_j

    # ---- epilogue into the fp32 segment workspace ----
    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE)
        + segm_idx * HEAD_SIZE
        + offs_d[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset, acc, mask=query_mask_0[:, None]
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    # Graph padding must be rejected before sequence lookup or scratch access.
    if query_token_idx >= tl.load(query_start_len_ptr + num_seqs):
        return

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if seq_len <= 0:
        return

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    # Empty segments must not set the maximum. Unvisited segments retain
    # the M=-inf,L=1 initial state.
    valid_segment = segm_mask & (segm_expsum > 0) & (segm_max != float("-inf"))
    segm_max = tl.where(valid_segment, segm_max, float("-inf"))
    overall_max = tl.max(segm_max)
    segment_scale = tl.where(valid_segment, tl.exp(segm_max - overall_max), 0.0)
    segm_expsum = segm_expsum * segment_scale
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= segment_scale[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization.

    For Gemma3, use 32 for both prefill and decode to better utilize
    the larger head dimension (128/256). For other models, use
    the default vLLM behavior.
    """
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    # The upstream triton_attn backend passes these newer knobs by keyword.
    # This gfx906 fork implements none of the features behind them yet;
    # accept them for call compatibility and fail loudly when a run
    # actually requests one (see the guard at the top of the body).
    rswa_prefix_lens=None,
    rswa_window: int | None = None,
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    k_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    chunk_lookback: int = -1,
    # Tensor-descriptor loads (Intel Xe2/Xe3) are numerics-neutral; this
    # fork never uses tensor descriptors, so the flag is accepted and
    # ignored.
    use_td: bool = False,
    # Gemma4's backend sets this for sliding-attention layers. The mask this
    # kernel builds is already (causal AND sliding window) OR mm_prefix, which
    # is what Transformers composes for Gemma4 (masking_utils: or_masks(
    # sliding_window_causal_mask_function, blockwise_overlay)), so the
    # bidirectional image block already overrides the window and no separate
    # clamping is needed. Accepted for call compatibility; inert by design.
    mm_prefix_clamp_sliding_window: bool = False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if (
        rswa_prefix_lens is not None
        or rswa_window is not None
        or chunk_lookback != -1
        or k_scale_cache is not None
        or v_scale_cache is not None
        or kv_quant_mode not in (KVQuantMode.NONE, KVQuantMode.FP8_PER_TENSOR)
    ):
        # Per-tensor FP8 needs no new plumbing: its scales arrive through
        # the k_descale / v_descale args above, as before the refactor.
        raise NotImplementedError(
            "The gfx906 Triton unified-attention fork does not implement "
            "R-SWA, chunked attention or per-token-head/INT4 KV caches "
            f"(kv_quant_mode={kv_quant_mode}, chunk_lookback={chunk_lookback}, "
            f"rswa_window={rswa_window})."
        )

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    gfx906_sched = _gfx906_sched_tuning_enabled()

    # Spec-decode forwards (verify + draft) look like tiny prefills to the
    # max_seqlen_q > 1 dispatch below but scan the whole decode context;
    # flag them so they take the split-KV decode path instead (see
    # ENABLE_GFX906_ATTN_SPEC_SPLIT_KV). The segment workspace is indexed
    # by absolute token position (one row per scheduled token) and the
    # backend sizes it to the decode batch threshold, so batches whose
    # total spec tokens exceed that (multi-sequence speculative decoding)
    # keep the previous routing through this bound (tuned-prefill split-KV
    # when eager, the 2D kernel under CUDA-graph capture).
    _spec_decode_batch = (
        ENABLE_GFX906_ATTN_SPEC_SPLIT_KV
        and gfx906_sched
        and 1 < max_seqlen_q <= GFX906_SPEC_MAX_QUERY_LEN
        and softmax_segm_output is not None
        and softmax_segm_max is not None
        and softmax_segm_expsum is not None
        and q.shape[0] <= softmax_segm_output.shape[0]
    )

    # Spec batches share the decode row packing (one query token per block,
    # reusing the KV-scan shape the decode path is tuned for) rather than
    # the prefill BLOCK_M: on gfx906 GQA6/head-256, sweeping the spec shape
    # (q=3) shows the 8-row/16-token/2-warp split-KV configuration
    # 1.7-1.9x faster than the prefill 32/32/4 one at 4k-64k contexts
    # (475->260us at 4k, 1563->788us at 16k, ~5.0->2.9ms at 64k per call).
    decode_block_m = (
        _decode_block_m(head_size, num_queries_per_kv)
        if gfx906_sched and (max_seqlen_q == 1 or _spec_decode_batch)
        else None
    )
    prefill_block_m = (
        _prefill_block_m(head_size, num_queries_per_kv)
        if gfx906_sched and max_seqlen_q > 1
        else None
    )
    BLOCK_M = decode_block_m or prefill_block_m or (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    if gfx906_sched:
        total_num_q_blocks = _num_query_blocks(q.shape[0], num_seqs, BLOCK_Q)
    else:
        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_PREFILL = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=True,
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=False,
    )

    def _launch_split_kv(
        segm_output,
        segm_max,
        segm_expsum,
        num_segments,
        tile_size,
        num_warps,
    ):
        if use_slim_3d:
            kernel_unified_attention_3d_slim[
                (total_num_q_blocks, num_kv_heads, num_segments)
            ](
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                query_start_len_ptr=cu_seqlens_q,
                scale=softmax_scale,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                BLOCK_SIZE=block_size,
                TILE_SIZE=tile_size,
                HEAD_SIZE=head_size,
                BLOCK_Q=BLOCK_Q,
                num_seqs=num_seqs,
                BLOCK_M=BLOCK_M,
                NUM_SEGMENTS_PER_SEQ=num_segments,
                OFFSETS_I32=True,
                SCALAR_PHYS=scalar_phys,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            kernel_unified_attention_3d[
            (total_num_q_blocks, num_kv_heads, num_segments)
        ](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=tile_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_segments,
            # num_warps=None keeps the Triton defaults on non-gfx906 platforms
            **({"num_warps": num_warps, "num_stages": 1} if num_warps else {}),
        )
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=tile_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_segments,
            USE_FP8=output_scale is not None,
        )

    # gfx906: run tuned-shape prefill through the split-KV (3D) kernel. The
    # path is restricted to plain causal attention so the swept configuration
    # (BLOCK_M 32 / TILE 32 / 4 warps / 1 stage / 2 segments) is the only one
    # exercised, and is disabled under CUDA-graph capture because the segment
    # workspace cannot be allocated inside a capture. Spec-decode batches are
    # excluded: they take the split-KV decode path with backend-provided
    # buffers instead (see _spec_decode_batch above).
    _prefill_3d_shape = (
        ENABLE_GFX906_ATTN_PREFILL_3D
        and gfx906_sched
        and prefill_block_m is not None
        and sliding_window_val == 0
        and not use_mm_prefix
        and not use_qq_bias
        and not use_alibi_slopes
        and sinks is None
        and softcap == 0
        and output_scale is None
        and not is_batch_invariant
    )
    use_prefill_3d = (
        _prefill_3d_shape
        and not _spec_decode_batch
        and not torch.cuda.is_current_stream_capturing()
    )

    # Slim plain-causal prefill kernel: same math as the gated 3D path,
    # specialized for exact-GQA packing with power-of-two heads (see the
    # ENABLE_GFX906_ATTN_PREFILL_SLIM comment for the structural
    # changes). Gated on the plain-causal shape family rather than on
    # use_prefill_3d itself so the spec-decode split-KV launch (which runs
    # under CUDA-graph capture with backend-provided buffers) can also use
    # it. The slim kernel has no descale branch, and its tl.dot
    # cannot take an fp8 KV cache, so a 1-byte-element KV cache (fp8
    # e4m3/e5m2, which the backend always pairs with descale tensors)
    # must fall back to the generic 3D kernel. Descale tensors on a
    # 2-byte fp16/bf16 cache are inert (the generic kernel only applies
    # them when the loaded cache is fp8; the GGUF backend passes unit
    # scales unconditionally), so slim still takes those shapes.
    use_slim_3d = (
        _prefill_3d_shape
        and ENABLE_GFX906_ATTN_PREFILL_SLIM
        and k.element_size() > 1
        and v.element_size() > 1
        and num_query_heads == num_kv_heads * num_queries_per_kv
        and head_size == triton.next_power_of_2(head_size)
        and k.stride(3) == 1
        and v.stride(3) == 1
        and k.numel() < 2**31
        and v.numel() < 2**31
    )
    # a 32-token tile never straddles an 800-token page, so the per-tile
    # physical-block gather collapses to one scalar load
    scalar_phys = block_size % TILE_SIZE_PREFILL == 0

    # Handwritten gfx906 HIP scanner: single-sequence fp16 fast path on the
    # tuned shape. Writes the same fp32 segment workspace the Triton path
    # consumes, so only the scanner launch is replaced.
    use_hip_scan = (
        use_slim_3d
        and ENABLE_GFX906_ATTN_PREFILL_HIP
        and hasattr(torch.ops._C, "gfx906_prefill_attn_scan")
        and num_seqs == 1
        and q.dtype == torch.float16
        and k.dtype == torch.float16
        and head_size == 256
        and num_query_heads == 24
        and num_kv_heads == 4
        and block_size == 800
        and q.is_contiguous()
        and k.stride(3) == 1
        and v.stride(3) == 1
        # the HIP scanner stages K/V rows with 16B vector loads; every row
        # stride must be a multiple of 8 halves (contiguous and serve
        # interleaved layouts both qualify; the strides are in the miss log)
        and k.stride(0) % 8 == 0
        and k.stride(1) % 8 == 0
        and k.stride(2) % 8 == 0
        and v.stride(0) % 8 == 0
        and v.stride(1) % 8 == 0
        and v.stride(2) % 8 == 0
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
    )

    if use_prefill_3d:
        num_segments = GFX906_PREFILL_SEGMENTS
        segm_output, segm_max, segm_expsum = _prefill_segment_buffers(
            q.shape[0],
            num_query_heads,
            num_segments,
            triton.next_power_of_2(head_size),
            q.device,
        )
        if use_hip_scan:
            _log_hip_prefill_scan_once()
            global _hip_dbg_buf
            if _hip_dbg_buf is None or _hip_dbg_buf.device != q.device:
                _hip_dbg_buf = torch.zeros(
                    2, dtype=torch.int32, device=q.device)
            _hip_dbg = _hip_dbg_buf
            torch.ops._C.gfx906_prefill_attn_scan(
                q,
                k,
                v,
                block_table,
                seqused_k,
                segm_output,
                segm_max,
                segm_expsum,
                softmax_scale,
                q.shape[0],
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                _hip_dbg,
            )
            if ENABLE_GFX906_HIP_ATTN_GUARD:
                hits = int(_hip_dbg[0].item()) + int(_hip_dbg[1].item())
                if hits:
                    logger.warning(
                        "gfx906 hip scan clamped %d stale page/tin and %d "
                        "stale pbi inputs", int(_hip_dbg[0].item()),
                        int(_hip_dbg[1].item()))
                    _hip_dbg.zero_()
            # the HIP scanner uses 64-row q blocks (BLOCK_Q 10) and 32-token
            # KV tiles; reduce_segments recomputes the segment layout from
            # these, so both must match the scanner above
            reduce_segments[(q.shape[0], num_query_heads)](
                output_ptr=out,
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                seq_lens_ptr=seqused_k,
                num_seqs=1,
                num_query_heads=num_query_heads,
                out_scale_inv=1.0,
                output_stride_0=out.stride(0),
                output_stride_1=out.stride(1),
                block_table_stride=block_table.stride(0),
                TILE_SIZE=32,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                query_start_len_ptr=cu_seqlens_q,
                BLOCK_Q=10,
                NUM_SEGMENTS_PER_SEQ=num_segments,
                USE_FP8=False,
            )
            return
        _log_hip_prefill_miss_once(
            num_seqs, str(q.dtype), str(k.dtype), head_size, num_query_heads,
            num_kv_heads, block_size, q.is_contiguous(), tuple(k.stride()),
            tuple(v.stride()), str(block_table.dtype), block_table.stride(1),
            hasattr(torch.ops._C, "gfx906_prefill_attn_scan"), use_slim_3d)
        _launch_split_kv(
            segm_output,
            segm_max,
            segm_expsum,
            num_segments,
            TILE_SIZE_PREFILL,
            4,
        )
        return

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    # Spec-decode batches (few query tokens per sequence, see
    # _spec_decode_batch) are excluded from (2) so they fall through to the
    # split-KV path below.
    if (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or (max_seqlen_q > 1 and not _spec_decode_batch)
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    ):
        kernel_unified_attention_2d[
            (
                total_num_q_blocks,
                num_kv_heads,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
            **(
                # tuned prefill shapes run 4 warps; everything else (incl. the
                # decode fallback through the 2D kernel) keeps 2 warps
                {"num_warps": 4 if prefill_block_m else 2, "num_stages": 1}
                if gfx906_sched
                else {}
            ),
        )
    else:
        # gfx906: scale split-KV parallelism with the decode batch shape and
        # context length instead of the fixed default, bounded by the segment
        # buffers allocated by the backend.
        num_segments = num_par_softmax_segments
        if gfx906_sched and softmax_segm_output is not None:
            num_segments = min(
                _decode_num_segments(
                    max_seqlen_k,
                    total_num_q_blocks,
                    num_kv_heads,
                    num_queries_per_kv,
                ),
                softmax_segm_output.shape[2],
            )
        if _spec_decode_batch:
            # Fixed segment count: graph capture runs with seq_lens=1, so a
            # context-derived count would freeze at the capture-time context
            # while replay scans the real one. On the decode-tuned shape the
            # batch shares the decode tile/warp configuration (see
            # decode_block_m above); other shapes keep the swept prefill
            # configuration their q blocks were sized against.
            num_segments = min(
                GFX906_SPEC_ATTN_SEGMENTS, softmax_segm_output.shape[2]
            )
            if decode_block_m is not None:
                splitkv_tile_size = TILE_SIZE_DECODE
                splitkv_num_warps = 2
            else:
                splitkv_tile_size = TILE_SIZE_PREFILL
                splitkv_num_warps = 4
        else:
            splitkv_tile_size = TILE_SIZE_DECODE
            splitkv_num_warps = (
                _decode_num_warps(head_size, max_seqlen_k, num_queries_per_kv)
                if gfx906_sched
                else None
            )
        _launch_split_kv(
            softmax_segm_output,
            softmax_segm_max,
            softmax_segm_expsum,
            num_segments,
            splitkv_tile_size,
            splitkv_num_warps,
        )
