# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed expert residency using the idle prefill staging allocation.

Enabled by default for single-request eager FP16 qwen4_exp GGUF on gfx906
without DBO. Override with VLLM_GGUF_MOE_EXPERT_CACHE=0 or 1. The existing
VLLM_GGUF_MOE_PREFILL_STAGING_MB budget bounds residency; the cache does not
allocate another weight buffer. VLLM_GGUF_MOE_EXPERT_CACHE_REFRESH controls
the refresh interval (256 decode steps by default). Filling the cache can
outweigh savings on short outputs; other serving configurations retain opt-in.
"""

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.gguf_moe_staging import (
    _STAGING,
    GGUFMoEStaging,
)


@triton.jit
def _observe_routes(
    ids,
    counts,
    resident,
    stats,
    N: tl.constexpr,
    PAIR_BYTES: tl.constexpr,
    DECODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    experts = tl.load(ids + offsets, valid, other=0)
    tl.atomic_add(counts + experts, 1.0, valid, sem="relaxed")
    if DECODE:
        hit = tl.load(resident + experts, valid, other=0)
        routes = tl.sum(valid.to(tl.int32), 0).to(tl.int64)
        hits = tl.sum((hit & valid).to(tl.int32), 0).to(tl.int64)
        tl.atomic_add(stats, routes, sem="relaxed")
        tl.atomic_add(stats + 1, hits, sem="relaxed")
        tl.atomic_add(stats + 2, (routes - hits) * PAIR_BYTES, sem="relaxed")


@dataclass
class ExpertCacheEntry:
    weights: tuple[torch.Tensor, torch.Tensor]
    counts: torch.Tensor
    resident: torch.Tensor
    stats: torch.Tensor
    pointers: tuple[torch.Tensor, torch.Tensor]
    slots: int = 0

    @property
    def pair_bytes(self) -> int:
        return sum(w[0].numel() * w.element_size() for w in self.weights)


class GGUFMoEExpertCache:
    """One eager stream owns the shared staging/cache storage on this device.

    Prefill invalidates residency before staging writes any weights. The first
    decode fills the hottest experts seen in prefill, then retains them across
    tokens. Periodic refresh uses accumulated routes without a host readback.
    """

    def __init__(self, device: torch.device, pool: GGUFMoEStaging):
        self.device = device
        self.pool = pool
        self.entries: dict[tuple[int, int], ExpertCacheEntry] = {}
        self.phase = "prefill"
        self.ready = False
        self.steps = 0
        self.refreshes = 0
        self.generation = -1
        self.stream: int | None = None
        self.refresh_interval = max(
            1, int(os.getenv("VLLM_GGUF_MOE_EXPERT_CACHE_REFRESH", "256"))
        )

    def observe(
        self, w1: torch.Tensor, w2: torch.Tensor, ids: torch.Tensor
    ) -> ExpertCacheEntry | None:
        stream = torch.cuda.current_stream(self.device)
        if self.stream is None:
            self.stream = stream.cuda_stream
        if self.stream != stream.cuda_stream:
            raise RuntimeError("GGUF expert residency requires one eager model stream")
        key = (w1.data_ptr(), w2.data_ptr())
        if key not in self.entries:
            self.ready = False
            experts = w1.shape[0]
            self.entries[key] = ExpertCacheEntry(
                (w1, w2),
                torch.zeros(experts, device=self.device, dtype=torch.float32),
                torch.zeros(experts, device=self.device, dtype=torch.bool),
                torch.zeros(4, device=self.device, dtype=torch.int64),
                tuple(
                    torch.arange(experts, device=self.device, dtype=torch.int64)
                    * w.stride(0)
                    * w.element_size()
                    + w.data_ptr()
                    for w in (w1, w2)
                ),
            )
        entry = self.entries[key]
        decode = ids.shape[0] == 1
        if not decode:
            if self.phase == "decode":
                for value in self.entries.values():
                    value.counts.zero_()
                    value.stats.zero_()
                self.steps = 0
            self.phase = "prefill"
            self.ready = False
        else:
            self.phase = "decode"
            if self.generation != self.pool.generation:
                self.ready = False
            if key == next(iter(self.entries)):
                self.steps += 1
                if (self.steps - 1) % self.refresh_interval == 0:
                    self.ready = False
            if not self.ready:
                self._fill(stream)
        _observe_routes[(triton.cdiv(ids.numel(), 256),)](
            ids,
            entry.counts,
            entry.resident,
            entry.stats,
            N=ids.numel(),
            PAIR_BYTES=entry.pair_bytes,
            DECODE=decode and self.ready,
            BLOCK=256,
        )
        return entry if decode and self.ready and entry.slots else None

    def _fill(self, stream: torch.cuda.Stream) -> None:
        with self.pool.lock:
            if self.pool.max_bytes <= 0:
                return
            if self.pool.buffer is None:
                try:
                    self.pool.buffer = torch.empty(
                        self.pool.max_bytes, device=self.device, dtype=torch.uint8
                    )
                except torch.cuda.OutOfMemoryError:
                    self.pool.max_bytes = 0
                    return
            if self.pool.done is not None:
                stream.wait_event(self.pool.done)
            storage = self.pool.buffer
            storage.record_stream(stream)
            budget = storage.numel() // len(self.entries)
            offset = 0
            for entry in self.entries.values():
                entry.slots = min(entry.counts.numel(), budget // entry.pair_bytes)
                selected = torch.topk(entry.counts, entry.slots, sorted=False).indices
                entry.resident.zero_()
                entry.resident.scatter_(0, selected, True)
                for weight, pointers in zip(entry.weights, entry.pointers):
                    expert_bytes = weight[0].numel() * weight.element_size()
                    pointers.copy_(
                        torch.arange(
                            weight.shape[0], device=self.device, dtype=torch.int64
                        )
                        * expert_bytes
                        + weight.data_ptr()
                    )
                    if not entry.slots:
                        continue
                    size = entry.slots * expert_bytes
                    cached = (
                        storage[offset : offset + size]
                        .view(weight.dtype)
                        .view(entry.slots, *weight.shape[1:])
                    )
                    torch.index_select(weight, 0, selected, out=cached)
                    addresses = (
                        torch.arange(entry.slots, device=self.device, dtype=torch.int64)
                        * expert_bytes
                        + cached.data_ptr()
                    )
                    pointers.scatter_(0, selected, addresses)
                    entry.stats[3].add_(size)
                    offset += size
                # Let subsequent decode routes outweigh an old long prompt.
                entry.counts.mul_(0.25)
            self.ready = True
            self.generation = self.pool.generation
            self.refreshes += 1


_CACHES: dict[torch.device, GGUFMoEExpertCache] = {}


def get_gguf_expert_cache(device: torch.device) -> GGUFMoEExpertCache:
    if device not in _CACHES:
        if device not in _STAGING:
            size = max(0, int(os.getenv("VLLM_GGUF_MOE_PREFILL_STAGING_MB", "512")))
            _STAGING[device] = GGUFMoEStaging(device, size * 1024**2)
        _CACHES[device] = GGUFMoEExpertCache(device, _STAGING[device])
    return _CACHES[device]
