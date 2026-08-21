# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler

pytestmark = pytest.mark.cpu_test

MAMBA_BLOCK_SIZE = 1600
PROMPT_LEN = 2002


@pytest.mark.parametrize(
    ("mamba_block_size", "enable_prefix_caching", "mamba_cache_mode", "expected"),
    [
        (MAMBA_BLOCK_SIZE, True, "align", True),
        (MAMBA_BLOCK_SIZE, False, "align", False),
        (MAMBA_BLOCK_SIZE, True, "none", False),
        (MAMBA_BLOCK_SIZE, True, "all", False),
        (None, True, "align", False),
    ],
)
def test_mamba_block_aligned_split_enablement(
    mamba_block_size: int | None,
    enable_prefix_caching: bool,
    mamba_cache_mode: str,
    expected: bool,
) -> None:
    assert (
        Scheduler._needs_mamba_block_aligned_split(
            mamba_block_size,
            enable_prefix_caching,
            mamba_cache_mode,
        )
        is expected
    )


def _split(
    start: int,
    num_new_tokens: int,
    *,
    prompt_len: int = PROMPT_LEN,
    max_num_scheduled_tokens: int = 16384,
) -> int:
    request = SimpleNamespace(
        num_computed_tokens=start,
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
    )
    scheduler = SimpleNamespace(
        mamba_block_size=MAMBA_BLOCK_SIZE,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
    )
    return Scheduler._mamba_block_aligned_split(
        scheduler, request, num_new_tokens
    )


@pytest.mark.parametrize("budget", [364, 800, 1599])
def test_short_intermediate_chunk_waits_for_block_boundary(budget: int) -> None:
    assert _split(0, budget) == 0


@pytest.mark.parametrize("budget", [1600, 1601, 2000])
def test_intermediate_chunk_ends_on_block_boundary(budget: int) -> None:
    assert _split(0, budget) == MAMBA_BLOCK_SIZE


@pytest.mark.parametrize("start", [331, 1599, 1601, 2531, 3011])
def test_unaligned_resume_never_crosses_next_block(start: int) -> None:
    prompt_len = 3602
    num_new_tokens = _split(start, prompt_len - start, prompt_len=prompt_len)
    next_boundary = (start // MAMBA_BLOCK_SIZE + 1) * MAMBA_BLOCK_SIZE
    assert start + num_new_tokens <= next_boundary


def test_small_scheduler_budget_can_make_sub_block_progress() -> None:
    assert _split(0, 364, max_num_scheduled_tokens=512) == 364
    assert _split(364, 1236, max_num_scheduled_tokens=512) == 1236


def test_final_prefill_chunk_need_not_be_block_aligned() -> None:
    assert _split(MAMBA_BLOCK_SIZE, PROMPT_LEN - MAMBA_BLOCK_SIZE) == 402
