import pytest

from vllm.attention.backends.abstract import AttentionType
from vllm.platforms import current_platform
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)


def test_cpu_attention_backend_capabilities() -> None:
    assert CPUAttentionBackend.supports_non_causal()
    assert CPUAttentionBackend.supports_attn_type(AttentionType.ENCODER_DECODER)
    assert CPUAttentionBackend.get_supported_head_sizes() == [
        32,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        512,
    ]
