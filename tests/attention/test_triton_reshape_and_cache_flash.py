import torch

from vllm.attention.ops.triton_reshape_and_cache_flash import (
    _reshape_and_cache_flash_eager,
)


def test_reshape_and_cache_flash_eager_ignores_padded_kv_rows():
    key = torch.arange(4 * 1 * 2, dtype=torch.float16).reshape(4, 1, 2)
    value = key + 100
    key_cache = torch.zeros((1, 4, 1, 2), dtype=torch.float16)
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.tensor([2, -1, 0], dtype=torch.long)

    _reshape_and_cache_flash_eager(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="auto",
        k_scale=torch.tensor(1.0),
        v_scale=torch.tensor(1.0),
    )

    torch.testing.assert_close(key_cache[0, 2], key[0])
    torch.testing.assert_close(value_cache[0, 2], value[0])
    torch.testing.assert_close(key_cache[0, 0], key[2])
    torch.testing.assert_close(value_cache[0, 0], value[2])
    torch.testing.assert_close(key_cache[0, 1], torch.zeros_like(key_cache[0, 1]))
    torch.testing.assert_close(value_cache[0, 1], torch.zeros_like(value_cache[0, 1]))
