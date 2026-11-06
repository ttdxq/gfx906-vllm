# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mlock helpers that keep GPU-registered host shmem from being reclaimed.

On ROCm, pinned host allocations (``pin_memory=True`` / ``cudaHostAlloc``) are
backed by unlinked MAP_SHARED shmem mappings. The kernel is free to reclaim or
compact those pages, and every MMU-notifier invalidation stops all KFD queues
of the process until the entire buffer has been re-faulted, which shows up as
multi-second stalls (TTFT outliers, simultaneous pauses on both TP ranks).

``mlock`` makes the resident pages unevictable so reclaim skips them, and
``mlock2(MLOCK_ONFAULT)`` extends that to pages faulting in later without
pre-faulting sparse tails. Compaction must also be stopped from migrating the
locked pages via ``vm.compact_unevictable_allowed=0`` (a machine-wide sysctl),
because migration invalidates PTEs the same way reclaim does.
"""

import ctypes
import os
import re
import threading

from vllm.logger import init_logger

logger = init_logger(__name__)

# sys/mman.h: mlock2 flags
_MLOCK_ONFAULT = 0x01

_MAP_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) (\S{4}) ")


def mlock_range(addr: int, num_bytes: int, *, onfault: bool = False) -> str | None:
    """Lock ``num_bytes`` at ``addr`` with mlock; returns an error string or
    None on success."""
    libc = ctypes.CDLL(None, use_errno=True)
    if onfault:
        mlock2 = getattr(libc, "mlock2", None)
        if mlock2 is None:
            return "glibc mlock2 is unavailable"
        mlock2.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        rc = mlock2(ctypes.c_void_p(addr), ctypes.c_size_t(num_bytes), _MLOCK_ONFAULT)
    else:
        libc.mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        rc = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(num_bytes))
    return os.strerror(ctypes.get_errno()) if rc != 0 else None


def iter_writable_shared_mappings() -> list[tuple[int, int, str]]:
    """Yield (start, end, name) of this process's writable MAP_SHARED mappings."""
    mappings = []
    with open("/proc/self/smaps") as smaps:
        for line in smaps:
            match = _MAP_HEADER.match(line)
            if match is None:
                continue
            fields = line.split(None, 5)
            if "w" in match[3] and "s" in match[3]:
                name = fields[5].strip() if len(fields) > 5 else "(anonymous)"
                mappings.append((int(match[1], 16), int(match[2], 16), name))
    return mappings


def mlock_shared_mappings(
    min_bytes: int = 1 << 20, *, verbose: bool = True
) -> tuple[int, int]:
    """Lock every writable MAP_SHARED mapping with mlock2(MLOCK_ONFAULT).

    Skips mappings smaller than ``min_bytes`` (IPC rings and similar). Returns
    (locked_bytes, failed_bytes); failures are logged once per mapping.
    """
    locked = failed = 0
    for start, end, name in iter_writable_shared_mappings():
        num_bytes = end - start
        if num_bytes < min_bytes:
            continue
        error = mlock_range(start, num_bytes, onfault=True)
        if error is None:
            locked += num_bytes
        else:
            failed += num_bytes
            logger.warning(
                "mlock2(MLOCK_ONFAULT) failed on %d-byte shared mapping %s: %s",
                num_bytes,
                name or "(anonymous)",
                error,
            )
    if verbose and (locked or failed):
        logger.info(
            "mlocked %.2f GiB of shared host mappings (%.2f GiB failed)",
            locked / 2**30,
            failed / 2**30,
        )
    return locked, failed


def maybe_mlock_pinned_host_buffer(tensor) -> None:
    """Mlock one freshly pinned host tensor when the ROCm opt-in is active.

    Intended for long-lived GPU-registered tables that exist while the rest of
    the weights stream in: locking them at allocation time removes the biggest
    eviction surface during model load.
    """
    from vllm import envs
    from vllm.platforms import current_platform

    if (
        not envs.VLLM_ROCM_MLOCK_HOST_SHMEM
        or not current_platform.is_rocm()
        or not tensor.is_pinned()
        or tensor.numel() == 0
    ):
        return
    error = mlock_range(tensor.data_ptr(), tensor.nbytes)
    if error is None:
        logger.info(
            "mlocked %.2f GiB pinned host buffer at 0x%x",
            tensor.nbytes / 2**30,
            tensor.data_ptr(),
        )
    else:
        logger.warning(
            "mlock of a %d-byte pinned host buffer failed: %s "
            "(check RLIMIT_MEMLOCK / CAP_IPC_LOCK)",
            tensor.nbytes,
            error,
        )


def sweep_shared_mappings_until(
    stop: threading.Event,
    interval_s: float = 2.0,
) -> None:
    """Re-sweep shared mappings on a timer until ``stop`` is set.

    The weight load pins fresh host shmem blocks as it streams; re-locking
    every couple of seconds keeps reclaim from finding an unlocked registered
    window. mlock2 on an already-locked range is a cheap no-op.
    """
    assert isinstance(stop, threading.Event)
    while not stop.is_set():
        mlock_shared_mappings(verbose=False)
        stop.wait(interval_s)
