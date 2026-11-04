# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, shared staging for UVA-backed GGUF expert matrices."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class GGUFMoEStaging:
    def __init__(self, device: torch.device, max_bytes: int):
        self.device = device
        self.max_bytes = max_bytes
        self.buffer: torch.Tensor | None = None
        self.lock = Lock()
        self.done: torch.cuda.Event | None = None

    @contextmanager
    def lease(self) -> Iterator["GGUFMoEStaging"]:
        # Keep enqueueing atomic across streams. The event protects the buffer
        # until its last consumer, not just its last copy, has finished.
        with self.lock:
            stream = torch.cuda.current_stream(self.device)
            if self.done is not None:
                stream.wait_event(self.done)
            try:
                yield self
            finally:
                if self.buffer is not None:
                    if self.done is None:
                        self.done = torch.cuda.Event()
                    self.done.record(stream)

    def stage(self, weight: torch.Tensor, offloaded: bool) -> torch.Tensor:
        size = weight.numel() * weight.element_size()
        if not offloaded or not weight.is_contiguous() or size > self.max_bytes:
            return weight
        if self.buffer is None:
            try:
                self.buffer = torch.empty(
                    self.max_bytes, dtype=torch.uint8, device=self.device
                )
            except torch.cuda.OutOfMemoryError:
                self.max_bytes = 0
                logger.warning_once(
                    "Insufficient memory for GGUF MoE prefill staging; "
                    "using UVA expert reads. Increase CPU offload or reduce "
                    "VLLM_GGUF_MOE_PREFILL_STAGING_MB."
                )
                return weight
        staged = self.buffer[:size].view(weight.dtype).view(weight.shape)
        staged.copy_(weight, non_blocking=True)
        return staged


_STAGING: dict[torch.device, GGUFMoEStaging] = {}
_STAGING_LOCK = Lock()


def get_gguf_moe_staging(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_offloaded: bool,
    w2_offloaded: bool,
) -> GGUFMoEStaging | None:
    if (
        not (w1_offloaded or w2_offloaded)
        or not x.is_cuda
        or x.shape[0] < 32
        or topk_ids.numel() < 2 * w1.shape[0]
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    with _STAGING_LOCK:
        if x.device not in _STAGING:
            max_bytes = (
                max(0, int(os.getenv("VLLM_GGUF_MOE_PREFILL_STAGING_MB", "512")))
                * 1024**2
            )
            _STAGING[x.device] = GGUFMoEStaging(x.device, max_bytes)
        staging = _STAGING[x.device]
    if not any(
        offloaded
        and weight.is_contiguous()
        and weight.numel() * weight.element_size() <= staging.max_bytes
        for weight, offloaded in [(w1, w1_offloaded), (w2, w2_offloaded)]
    ):
        return None
    return staging
