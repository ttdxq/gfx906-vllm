# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager native submission of the existing single-token QSA kernel chain.

Enabled by default on gfx906 with FP16 caches; disable with
VLLM_QWEN4EXP_NATIVE_QSA_DECODE=0.
Inputs and metadata are rebound on every call; per-stream workspaces retain
ordinary compiled HIP kernels, without GPU graph capture. The existing top-k
selection remains unordered, so long-context attention retains its existing
FP16 reduction-order variability.
"""

import struct
from contextvars import ContextVar

import torch

_RECORDER: ContextVar = ContextVar("qsa_decode_recorder", default=None)


def launch_qsa(kernel, grid, *args, **kwargs):
    compiled = kernel[grid](*args, **kwargs)
    recorder = _RECORDER.get()
    if recorder is not None:
        recorder.kernel(kernel, compiled, grid, args, kwargs)
    return compiled


def qsa_topk(op, logits, next_n, lengths, indices, rows, stride0, stride1, topk):
    op(logits, next_n, lengths, indices, rows, stride0, stride1, topk)
    recorder = _RECORDER.get()
    if recorder is not None:
        assert next_n == rows == 1
        slots = [recorder.tensor(t) for t in (logits, lengths, indices)]
        assert all(offset == 0 for _, offset in slots)
        recorder.plan.add_topk([slot for slot, _ in slots] + [topk])


def _layout(tensors):
    groups = {}
    result = []
    for tensor in tensors:
        storage = tensor.untyped_storage().data_ptr()
        group, base = groups.setdefault(storage, (len(groups), tensor.data_ptr()))
        result.append(
            (
                tensor.shape,
                tensor.stride(),
                tensor.dtype,
                tensor.device,
                group,
                tensor.data_ptr() - base,
            )
        )
    return tuple(result)


class _RecordedQSA:
    def __init__(self, anchors):
        self.plan = torch.classes._C.Qwen4ExpQSADecodePlan(anchors[0].device.index)
        self.anchors = anchors
        self.static = []
        self.kernels = []

    def tensor(self, tensor):
        for i, anchor in enumerate(self.anchors):
            if tensor is anchor:
                return i, 0
        storage = tensor.untyped_storage().data_ptr()
        for i, anchor in enumerate(self.anchors):
            if storage == anchor.untyped_storage().data_ptr():
                return i, tensor.data_ptr() - anchor.data_ptr()
        for i, value in enumerate(self.static):
            if tensor is value:
                return len(self.anchors) + i, 0
        self.static.append(tensor)
        return len(self.anchors) + len(self.static) - 1, 0

    def kernel(self, jit, compiled, grid, args, kwargs):
        metadata = compiled.metadata
        assert (
            metadata.backend_name == "hip"
            and metadata.num_ctas == 1
            and not metadata.launch_cooperative_grid
            and metadata.global_scratch_size == metadata.profile_scratch_size == 0
        ), "QSA native submission requires ordinary HIP kernels without scratch"
        bound = dict(zip(jit.arg_names, args))
        bound.update({k: v for k, v in kwargs.items() if k in jit.arg_names})
        values, bindings = [], []
        for name, kind in compiled.src.signature.items():
            if kind == "constexpr":
                continue
            value = bound[name]
            if kind.startswith("*"):
                if value is not None:
                    slot, offset = self.tensor(value)
                    bindings.extend((len(values), slot, offset))
                value = 0
            elif kind == "fp32":
                value = int.from_bytes(struct.pack("<f", value), "little")
            elif kind == "fp64":
                value = int.from_bytes(struct.pack("<d", value), "little", signed=True)
            else:
                assert kind in ("i1", "i8", "i16", "i32", "i64", "u32", "u64"), kind
                value = int(value)
            values.append(value)
        # Triton's HIP launcher appends global and profiling scratch pointers.
        values.extend((0, 0))
        grid = tuple(grid) + (1,) * (3 - len(grid))
        self.plan.add_kernel(
            compiled.function,
            list(grid),
            metadata.num_warps * metadata.warp_size,
            metadata.shared,
            values,
            bindings,
        )
        # The driver module must outlive every native launch using its handle.
        self.kernels.append(compiled)

    def finish(self):
        self.anchors = []

    def run(self, anchors):
        self.plan.run(anchors + self.static)


class QSADecodePlans:
    """Bounded, per-attention eager workspaces with runtime tensor rebinding."""

    def __init__(self):
        self.entries = {}
        self.builds = 0
        self.replays = 0

    def run(self, anchors, body):
        key = (
            torch.cuda.current_stream(anchors[0].device).cuda_stream,
            _layout(anchors),
        )
        entry = self.entries.get(key)
        if entry is not None:
            entry.run(anchors)
            self.replays += 1
            return
        entry = _RecordedQSA(anchors)
        token = _RECORDER.set(entry)
        try:
            body()
        finally:
            _RECORDER.reset(token)
        entry.finish()
        if len(self.entries) >= 4:
            self.entries.pop(next(iter(self.entries)))
        self.entries[key] = entry
        self.builds += 1


def run_qsa_decode(attn, hidden, positions, qkv, output, gate, main, raw, compressed):
    """Prepare/select/attend using the same kernels and FP16 boundaries."""
    indexer = attn.indexer
    projected_q, raw_keys = indexer.project(hidden)
    anchors = [
        projected_q,
        raw_keys,
        positions,
        qkv,
        output,
        gate,
        attn.kv_cache,
        indexer.raw_key_cache.kv_cache,
        indexer.compressed_key_cache.kv_cache,
        attn.topk_indices_buffer,
        raw.slot_mapping,
        raw.block_table,
        raw.query_start_loc,
        raw.logical_positions,
        compressed.slot_mapping,
        compressed.k_work_metadata,
        compressed.block_table,
        compressed.token_to_req,
        compressed.logical_positions,
        compressed.seq_lens,
        main.slot_mapping,
        main.block_table,
        raw.token_to_req,
    ]

    def body():
        q, (main_q, _) = indexer._fused_prepare(
            projected_q,
            raw_keys,
            positions,
            raw,
            compressed,
            attn,
            qkv,
            main.slot_mapping,
            gate,
        )
        indexer._select(q, compressed, attn.topk_indices_buffer[:1])
        attn.impl.forward(
            attn,
            main_q,
            attn.kv_cache,
            main,
            output,
            token_to_req=raw.token_to_req,
        )

    plans = getattr(attn, "_native_decode_plans", None)
    if plans is None:
        attn._native_decode_plans = plans = QSADecodePlans()
    plans.run(anchors, body)
