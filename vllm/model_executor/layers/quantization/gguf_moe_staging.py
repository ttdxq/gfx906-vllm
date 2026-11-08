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


def use_gfx906_qwen4exp_decode_defaults(config) -> bool:
    """Use measured decode policies only for the validated serving workload."""
    if config is None:
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    model = config.model_config
    if model is None:
        return False
    if (
        model.hf_text_config.model_type not in ("qwen4_exp", "qwen4_exp_text")
        or model.quantization != "gguf"
        or model.dtype != torch.float16
        or not model.enforce_eager
        or config.scheduler_config.max_num_seqs != 1
        or config.parallel_config.enable_dbo
    ):
        return False
    from vllm.platforms.rocm import on_gfx906

    return on_gfx906()


class GGUFMoEStaging:
    def __init__(self, device: torch.device, max_bytes: int):
        self.device = device
        self.max_bytes = max_bytes
        self.buffer: torch.Tensor | None = None
        self.lock = Lock()
        self.done: torch.cuda.Event | None = None
        self.generation = 0
        self._route_ids: torch.Tensor | None = None
        self._route_index: torch.Tensor | None = None
        self._route_count = 0

    @contextmanager
    def lease(
        self,
        topk_ids: torch.Tensor | None = None,
        expert_count: int | None = None,
        compact: bool = False,
    ) -> Iterator["GGUFMoEStaging"]:
        # Keep enqueueing atomic across streams. The event protects the buffer
        # until its last consumer, not just its last copy, has finished.
        with self.lock:
            self.generation += 1
            stream = torch.cuda.current_stream(self.device)
            if self.done is not None:
                stream.wait_event(self.done)
            self._route_ids = None
            self._route_index = None
            self._route_count = 0
            if compact and topk_ids is not None and expert_count is not None:
                route_ids, route_index = torch.unique(
                    topk_ids.reshape(-1), sorted=True, return_inverse=True
                )
                # A nearly dense route does not save enough transfer to pay for
                # the unique/remap work and the compact kernel indexing.
                if route_ids.numel() * 4 < expert_count * 3:
                    self._route_ids = route_ids
                    self._route_index = route_index.to(topk_ids.dtype).view_as(
                        topk_ids
                    )
                    self._route_count = route_ids.numel()
            try:
                yield self
            finally:
                if self.buffer is not None:
                    if self.done is None:
                        self.done = torch.cuda.Event()
                    self.done.record(stream)
                self._route_ids = None
                self._route_index = None
                self._route_count = 0

    def can_compact(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_offloaded: bool,
        w2_offloaded: bool,
    ) -> bool:
        return all(
            offloaded
            and weight.is_contiguous()
            and weight.numel() * weight.element_size() <= self.max_bytes
            for weight, offloaded in ((w1, w1_offloaded), (w2, w2_offloaded))
        )

    def remap_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self._route_index is None:
            return topk_ids
        return self._route_index

    def effective_expert_count(self, default: int) -> int:
        return self._route_count or default

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
        if self._route_ids is not None:
            row_shape = weight.shape[1:]
            size = self._route_count * weight[0].numel() * weight.element_size()
            if size > self.max_bytes:
                return weight
            staged = self.buffer[:size].view(weight.dtype).view(
                (self._route_count, *row_shape)
            )
            torch.index_select(weight, 0, self._route_ids, out=staged)
            return staged
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
