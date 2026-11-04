# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from operator import attrgetter

import pytest
import torch
import torch.nn as nn
from gguf import GGMLQuantizationType, quantize

import vllm.envs as envs
from vllm.model_executor.layers.quantization.gguf import (
    GGUFUninitializedParameter,
    fused_moe_gguf,
)
from vllm.model_executor.layers.quantization.gguf_moe_staging import (
    GGUFMoEStaging,
    get_gguf_moe_staging,
)
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.model_executor.offloader import (
    PrefetchOffloader,
    UVAOffloader,
    get_offloader,
    set_offloader,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from ...utils import compare_two_settings


def test_gguf_prefill_staging_reuses_bounded_storage_across_streams():
    module = nn.Module()
    for index, size in enumerate([4096, 2048, 8192]):
        data = torch.arange(size, device="cuda").to(torch.uint8)
        module.register_parameter(
            f"weight{index}", nn.Parameter(data, requires_grad=False)
        )
    UVAOffloader(16384).wrap_modules(iter([module]))
    staging = GGUFMoEStaging(module.weight0.device, 4096)
    assert staging.stage(module.weight2, True) is module.weight2
    assert staging.stage(module.weight0, False) is module.weight0
    assert staging.buffer is None

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    outputs = []
    pointers = []
    for index in range(8):
        weight = getattr(module, f"weight{index % 2}")
        with torch.cuda.stream(streams[index % 2]), staging.lease():
            staged = staging.stage(weight, True)
            pointers.append(staged.data_ptr())
            outputs.append(staged.clone())
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert len(set(pointers)) == 1
    assert staging.buffer.numel() == 4096
    for index, output in enumerate(outputs):
        weight = getattr(module, f"weight{index % 2}")
        torch.testing.assert_close(output, weight, rtol=0, atol=0)


@pytest.mark.parametrize("host_weights", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize(
    "w1_type", [GGMLQuantizationType.Q4_0, GGMLQuantizationType.IQ4_NL]
)
@torch.inference_mode()
def test_gguf_prefill_staging_matches_uva_moe_exactly(
    monkeypatch, host_weights, w1_type
):
    monkeypatch.setenv("VLLM_GGUF_MOE_PREFILL_STAGING_MB", "1")
    torch.manual_seed(0)
    module = nn.Module()
    for name, shape, quant_type in [
        ("w1", (8, 512, 256), GGMLQuantizationType.Q4_0),
        ("w2", (8, 256, 256), GGMLQuantizationType.Q8_0),
    ]:
        # IQ4_NL has the same scale/nibble layout as Q4_0. Interpreting
        # these finite blocks as IQ4_NL also exercises the MMVQ path.
        packed = quantize((torch.randn(shape) * 0.01).numpy(), quant_type)
        module.register_parameter(
            name, nn.Parameter(torch.from_numpy(packed).cuda(), requires_grad=False)
        )
    original = [weight.cpu() for weight in module.parameters()]
    host_names = {name for name, host in zip(["w1", "w2"], host_weights) if host}
    UVAOffloader(2**30, host_names).wrap_modules(iter([module]))
    x = torch.randn((64, 256), device="cuda", dtype=torch.float16)
    ids = torch.randint(8, (64, 2), device="cuda", dtype=torch.int32)
    weights = torch.softmax(torch.randn(64, 2, device="cuda"), dim=-1)
    args = (x, module.w1, module.w2, weights, ids, int(w1_type), 8, "silu")
    reference = fused_moe_gguf(*args).clone()
    assert torch.isfinite(reference).all()
    actual = fused_moe_gguf(*args, *host_weights)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    # A later layer may reuse the scratch, but not the returned output.
    second = nn.Module()
    second.register_parameter("w1", module.w1)
    second.register_parameter(
        "w2", nn.Parameter(module.w2.clone(), requires_grad=False)
    )
    if host_weights[1]:
        UVAOffloader(2**30, {"w2"}).wrap_modules(iter([second]))
    fused_moe_gguf(
        x * 0.5,
        second.w1,
        second.w2,
        weights,
        ids,
        int(w1_type),
        8,
        "silu",
        *host_weights,
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for weight, expected in zip(module.parameters(), original):
        torch.testing.assert_close(weight.cpu(), expected, rtol=0, atol=0)

    # Decode and device-only weights must not acquire the prefill scratch.
    assert (
        get_gguf_moe_staging(x[:1], module.w1, module.w2, ids[:1], *host_weights)
        is None
    )
    assert get_gguf_moe_staging(x, module.w1, module.w2, ids, False, False) is None


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
