# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import pytest
import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.gguf_moe_expert_cache import (
    GGUFMoEExpertCache,
)
from vllm.model_executor.layers.quantization.gguf_moe_staging import GGUFMoEStaging
from vllm.model_executor.offloader import UVAOffloader
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "variant",
    ["supported", "qwen3_5", "non_gguf", "float32", "non_eager", "batch", "dbo"],
)
def test_flash_gguf_default_policy_uses_text_config(tmp_path, variant):
    from transformers import Qwen4ExpTextConfig

    from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
    from vllm.model_executor.layers.quantization.gguf_moe_staging import (
        use_gfx906_qwen4exp_decode_defaults,
    )

    Qwen4ExpTextConfig(
        architectures=["Qwen4ExpForCausalLM"], eos_token_id=1, ple_layer_ids=[]
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="half",
            max_model_len=4096,
            quantization="gguf",
            enforce_eager=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_model_len=4096, is_encoder_decoder=False
        ),
    )
    assert config.model_config.hf_text_config.model_type == "qwen4_exp_text"
    if variant == "qwen3_5":
        config.model_config.hf_text_config.model_type = "qwen3_5"
    elif variant == "non_gguf":
        config.model_config.quantization = None
    elif variant == "float32":
        config.model_config.dtype = torch.float32
    elif variant == "non_eager":
        config.model_config.enforce_eager = False
    elif variant == "batch":
        config.scheduler_config.max_num_seqs = 2
    elif variant == "dbo":
        config.parallel_config.enable_dbo = True
    supported = False
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx906

        supported = on_gfx906()
    assert use_gfx906_qwen4exp_decode_defaults(config) == (
        variant == "supported" and supported
    )


@pytest.mark.parametrize("top_k", [2, 10])
@pytest.mark.parametrize("cache_env", [None, "1"])
@torch.inference_mode()
def test_cached_fused_moe_matches_original(monkeypatch, top_k, cache_env):
    from vllm.model_executor.layers.quantization import gguf as gg
    from vllm.model_executor.layers.quantization import gguf_moe_expert_cache as ec
    from vllm.model_executor.layers.quantization import gguf_moe_staging as gs

    device = torch.device("cuda", torch.cuda.current_device())
    module = torch.nn.Module()
    for name, shape in [("w1", (16, 512, 256)), ("w2", (16, 256, 256))]:
        packed = gguf.quantize(
            (torch.randn(shape) * 0.01).numpy(), gguf.GGMLQuantizationType.Q8_0
        )
        module.register_parameter(
            name,
            torch.nn.Parameter(torch.from_numpy(packed).cuda(), requires_grad=False),
        )
    UVAOffloader(2**30, {"w1", "w2"}).wrap_modules(iter([module]))
    pool = GGUFMoEStaging(device, 2**20)
    cache = GGUFMoEExpertCache(device, pool)
    monkeypatch.setitem(gs._STAGING, device, pool)
    monkeypatch.setitem(ec._CACHES, device, cache)
    prompt = torch.randint(16, (32, top_k), device=device, dtype=torch.int32)
    x = torch.randn(1, 256, device=device, dtype=torch.float16)
    routes = prompt[:1]
    weights = torch.softmax(torch.randn(1, top_k, device=device), dim=-1)
    args = (x, module.w1, module.w2, weights, routes, 8, 8, "silu", True, True)
    monkeypatch.setenv("VLLM_GGUF_MOE_EXPERT_CACHE", "0")
    expected = gg.fused_moe_gguf(*args, expert_cache_default=True).clone()
    assert cache.steps == 0
    if cache_env is None:
        monkeypatch.delenv("VLLM_GGUF_MOE_EXPERT_CACHE", raising=False)
    else:
        monkeypatch.setenv("VLLM_GGUF_MOE_EXPERT_CACHE", cache_env)
    cache.observe(module.w1, module.w2, prompt)
    for _ in range(3):
        actual = gg.fused_moe_gguf(*args, expert_cache_default=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cache_supported = False
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx906

        cache_supported = on_gfx906()
    assert cache.steps == (3 if cache_supported else 0)


@pytest.mark.parametrize("interval", [1, 4])
@torch.inference_mode()
def test_expert_cache_retains_refreshes_and_rebuilds(monkeypatch, interval):
    """Exercise real UVA misses, retained HBM hits and prefill scratch reuse."""
    monkeypatch.setenv("VLLM_GGUF_MOE_EXPERT_CACHE_REFRESH", str(interval))
    device = torch.device("cuda", torch.cuda.current_device())
    layers = []
    for i in range(2):
        layer = torch.nn.Module()
        for name in ("w1", "w2"):
            data = gguf.quantize(
                (torch.randn(8, 32, 256) * (0.01 + i)).numpy(),
                gguf.GGMLQuantizationType.Q8_0,
            )
            layer.register_parameter(
                name,
                torch.nn.Parameter(torch.from_numpy(data).cuda(), requires_grad=False),
            )
        UVAOffloader(2**30, {"w1", "w2"}).wrap_modules(iter([layer]))
        layers.append(layer)
    pair_bytes = sum(w[0].numel() for w in layers[0].parameters())
    pool = GGUFMoEStaging(device, 4 * pair_bytes)
    cache = GGUFMoEExpertCache(device, pool)
    prompt = torch.tensor([[7, 2]] * 32, device=device, dtype=torch.int32)
    for layer in layers:
        assert cache.observe(layer.w1, layer.w2, prompt) is None
    routes = prompt[:1]
    x = torch.randn(1, 256, device=device, dtype=torch.float16)
    quant_x = ops.ggml_quantize_row_q8_1(x)

    def check(layer, ids):
        entry = cache.observe(layer.w1, layer.w2, ids)
        assert entry is not None
        assert entry.slots == 2
        for weight, pointers in zip(entry.weights, entry.pointers):
            expected = ops.ggml_moe_q8_vec(
                quant_x, weight, ids, 2, 8, 32, 1, 256, x.dtype
            )
            actual = torch.ops._C.ggml_moe_q8_vec_indirect(
                quant_x, pointers, ids, 8, 32, 1, 256, x.dtype
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return entry

    for _ in range(3):
        for layer in layers:
            entry = check(layer, routes)
            assert entry.resident[7] and entry.resident[2]
    assert cache.refreshes == (3 if interval == 1 else 1)
    refreshes = cache.refreshes
    miss = torch.tensor([[0, 6]], device=device, dtype=torch.int32)
    for layer in layers:
        entry = check(layer, miss)
        assert entry.stats[0].item() == 8
        assert entry.stats[1].item() == 6
        assert entry.stats[2].item() == 2 * pair_bytes
    # Invalidate before the prefill staging lease overwrites every cached byte.
    new_prompt = torch.tensor([[1, 5]] * 32, device=device, dtype=torch.int32)
    for layer in layers:
        assert cache.observe(layer.w1, layer.w2, new_prompt) is None
        with pool.lease():
            pool.buffer.fill_(255)
    for layer in layers:
        entry = check(layer, new_prompt[:1])
        assert entry.resident[1] and entry.resident[5]
        assert entry.stats[0].item() == 2
        assert entry.stats[1].item() == 2
    assert cache.refreshes == refreshes + (2 if interval == 1 else 1)
    # Another staging user can invalidate the allocation without observing routes.
    with pool.lease():
        pool.buffer.fill_(255)
    refreshes = cache.refreshes
    check(layers[0], new_prompt[:1])
    assert cache.refreshes == refreshes + 1
    tiny = GGUFMoEExpertCache(device, GGUFMoEStaging(device, 1))
    assert tiny.observe(layers[0].w1, layers[0].w2, prompt) is None
    assert tiny.observe(layers[0].w1, layers[0].w2, routes) is None
    entry = next(iter(tiny.entries.values()))
    assert entry.slots == 0
    assert entry.stats[3].item() == 0
    for weight, pointers in zip(entry.weights, entry.pointers):
        expected = (
            torch.arange(8, device=device) * weight[0].numel() + weight.data_ptr()
        )
        torch.testing.assert_close(pointers, expected)
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(RuntimeError, match="one eager model stream"),
    ):
        cache.observe(layers[0].w1, layers[0].w2, routes)
