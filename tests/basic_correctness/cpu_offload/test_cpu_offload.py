# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from operator import attrgetter

import pytest
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.model_executor.layers.quantization.gguf import GGUFUninitializedParameter
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.model_executor.offloader import (
    PrefetchOffloader,
    UVAOffloader,
    get_offloader,
    set_offloader,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from ...utils import compare_two_settings


@pytest.mark.parametrize("disable_uva", [False, True])
def test_gguf_lazy_offload_preserves_hot_weights_and_counts_loaded_bytes(
    monkeypatch, disable_uva
):
    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", str(int(disable_uva)))
    module = nn.Module()
    module.register_parameter(
        "expert", GGUFUninitializedParameter(device="cuda", requires_grad=False)
    )
    module.add_module("hot", nn.Linear(32, 32, bias=False, device="cuda"))
    offloader = UVAOffloader(4096, {"expert"})
    offloader.wrap_modules(iter([module]))

    assert not getattr(module.expert, "_vllm_is_uva_offloaded", False)
    assert offloader.cpu_offload_bytes == 0
    before = torch.cuda.memory_allocated()
    module.expert.materialize((4096,), dtype=torch.uint8)
    assert offloader.cpu_offload_bytes == 4096
    assert torch.cuda.memory_allocated() == before
    expected = torch.arange(4096).to(torch.uint8)
    module.expert.data.copy_(expected)
    torch.testing.assert_close(module.expert.cpu(), expected, rtol=0, atol=0)
    assert module.hot.weight.is_cuda
    assert not getattr(module.hot.weight, "_vllm_is_uva_offloaded", False)


def test_gguf_lazy_offload_stops_at_materialized_budget():
    module = nn.Module()
    for name in ["first", "second"]:
        module.register_parameter(
            name, GGUFUninitializedParameter(device="cuda", requires_grad=False)
        )
    offloader = UVAOffloader(4096)
    offloader.wrap_modules(iter([module]))
    module.first.materialize((4096,), dtype=torch.uint8)
    before = torch.cuda.memory_allocated()
    module.second.materialize((4096,), dtype=torch.uint8)
    assert offloader.cpu_offload_bytes == 4096
    assert torch.cuda.memory_allocated() - before == 4096
    assert getattr(module.first, "_vllm_is_uva_offloaded", False)
    assert not getattr(module.second, "_vllm_is_uva_offloaded", False)


def test_uva_postload_restores_replaced_storage_with_retained_marker():
    module = nn.Linear(32, 32, bias=False, device="cuda")
    expected = module.weight.detach().cpu()
    UVAOffloader(4096).wrap_modules(iter([module]))
    before = torch.cuda.memory_allocated()
    with device_loading_context(module, torch.device("cuda")):
        module.weight.data = module.weight.data.clone()
    assert torch.cuda.memory_allocated() == before
    torch.testing.assert_close(module.weight.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("disable_pin_memory", [False, True])
@pytest.mark.parametrize("disable_uva", [False, True])
def test_cpu_offload(disable_pin_memory, disable_uva):
    env_vars = {
        "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": str(int(disable_pin_memory)),
        "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA": str(int(disable_uva)),
    }

    args = ["--cpu-offload-gb", "1"]

    # cuda graph only works with UVA offloading
    if disable_uva:
        args.append("--enforce-eager")

    compare_two_settings(
        model="hmellor/tiny-random-LlamaForCausalLM",
        arg1=[],
        arg2=args,
        env1=None,
        env2=env_vars,
    )


@pytest.mark.parametrize(
    ("offload_kwargs", "offloader_type"),
    [
        ({"cpu_offload_gb": 1}, UVAOffloader),
        (
            {
                "offload_group_size": 1,
                "offload_num_in_group": 1,
                "offload_prefetch_step": 1,
            },
            PrefetchOffloader,
        ),
    ],
)
def test_mrv2_weight_offloading(
    vllm_runner, monkeypatch, offload_kwargs, offloader_type
):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    envs.disable_envs_cache()
    original_offloader = get_offloader()

    try:
        with vllm_runner(
            "hmellor/tiny-random-LlamaForCausalLM",
            enforce_eager=True,
            gpu_memory_utilization=0.03,
            max_model_len=128,
            max_num_seqs=1,
            **offload_kwargs,
        ) as vllm_model:
            engine_core = vllm_model.llm.llm_engine.engine_core.engine_core
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner
            assert isinstance(model_runner, GPUModelRunner)

            offloader = get_offloader()
            assert isinstance(offloader, offloader_type)
            if isinstance(offloader, UVAOffloader):
                assert offloader.cpu_offload_bytes > 0
            else:
                assert offloader.total_offloaded_bytes > 0
                assert offloader.buffer_pool is not None
    finally:
        set_offloader(original_offloader)
        envs.disable_envs_cache()


def _is_offloaded(p: nn.Parameter) -> bool:
    return p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False)


@pytest.mark.parametrize("disable_uva", [False, True])
def test_tower_weight_offloading(vllm_runner, monkeypatch, disable_uva):
    """`cpu_offload_params` segments must reach towers built outside make_layers.

    Regression test: `wrap_modules` was only called from `make_layers`, so a
    directly-constructed vision tower never reached the offloader and segments
    targeting it silently matched nothing.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if disable_uva:
        monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "1")
    envs.disable_envs_cache()
    original_offloader = get_offloader()

    try:
        with vllm_runner(
            "Qwen/Qwen3.5-0.8B",
            enforce_eager=True,
            # allocate more vram as Qwen 3.5 has 1.6 GiB of weights
            gpu_memory_utilization=0.3,
            max_model_len=128,
            max_num_seqs=1,
            enable_prefix_caching=False,
            cpu_offload_gb=1,
            cpu_offload_params={"visual"},
        ) as vllm_model:
            engine_core = vllm_model.llm.llm_engine.engine_core.engine_core
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner

            offloader = get_offloader()
            assert isinstance(offloader, UVAOffloader)
            assert offloader.cpu_offload_bytes > 0

            model = model_runner.get_model()
            assert model._tower_model_names
            for name in model._tower_model_names:
                tower = attrgetter(name)(model)
                params = list(tower.parameters())
                assert params
                assert all(_is_offloaded(p) for p in params)
                if disable_uva:
                    # non-UVA fallback: weights live on CPU and are moved
                    # back on first forward
                    assert "forward" in vars(tower)

            # The language model must stay resident.
            assert not any(_is_offloaded(p) for p in model.language_model.parameters())
    finally:
        set_offloader(original_offloader)
        envs.disable_envs_cache()
