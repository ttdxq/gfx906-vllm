# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""mlock helpers for GPU-registered host shmem (Linux-only)."""

import re

import pytest
import torch

from vllm.utils.host_memory_lock import (
    iter_writable_shared_mappings,
    maybe_mlock_pinned_host_buffer,
    mlock_range,
    mlock_shared_mappings,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "cuda") or not torch.cuda.is_available(),
    reason="pinned host memory needs an accelerator",
)


def _vmflags_for(ptr: int) -> str:
    current = None
    flags = ""
    with open("/proc/self/smaps") as smaps:
        for line in smaps:
            if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                if current:
                    lo, hi = (int(x, 16) for x in current.split("-"))
                    if lo <= ptr < hi:
                        return flags
                current = line.split()[0]
                flags = ""
            elif line.startswith("VmFlags:"):
                flags = line.strip()
    return flags


def test_mlock_range_marks_mapping_unevictable():
    tensor = torch.empty(1 << 20, dtype=torch.uint8, device="cpu", pin_memory=True)
    ptr = tensor.data_ptr()
    before = _vmflags_for(ptr)
    assert " lo " not in before

    assert mlock_range(ptr, tensor.nbytes) is None

    after = _vmflags_for(ptr)
    assert " lo " in after, after


def test_mlock_shared_mappings_locks_pinned_buffer():
    tensor = torch.empty(2 << 20, dtype=torch.uint8, device="cpu", pin_memory=True)
    assert any(
        start <= tensor.data_ptr() < end
        for start, end, _ in iter_writable_shared_mappings()
    )
    locked, failed = mlock_shared_mappings(min_bytes=1 << 20)
    assert failed == 0
    assert locked >= tensor.nbytes
    assert " lo " in _vmflags_for(tensor.data_ptr())


def test_maybe_mlock_pinned_host_buffer_opt_out(monkeypatch):
    from vllm import envs
    from vllm.utils import host_memory_lock as hml

    calls: list[tuple] = []
    monkeypatch.setattr(
        hml, "mlock_range", lambda *a, **k: calls.append(a) or "stubbed"
    )
    tensor = torch.empty(1 << 20, dtype=torch.uint8, device="cpu", pin_memory=True)

    monkeypatch.setattr(envs, "VLLM_ROCM_MLOCK_HOST_SHMEM", False)
    maybe_mlock_pinned_host_buffer(tensor)
    assert calls == []


def test_mlock_private_mappings_locks_plain_host_tensor():
    from vllm.utils.host_memory_lock import mlock_private_mappings

    tensor = torch.empty(2 << 20, dtype=torch.uint8)  # plain anonymous pages
    ptr = tensor.data_ptr()
    assert " lo " not in _vmflags_for(ptr)

    locked, failed = mlock_private_mappings(min_bytes=1 << 20)

    assert failed == 0
    assert locked >= tensor.nbytes
    assert " lo " in _vmflags_for(ptr)
