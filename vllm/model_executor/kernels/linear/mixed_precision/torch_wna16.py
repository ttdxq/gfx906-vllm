import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx906
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class TorchWNA16LinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.uint8,
        scalar_types.uint8b128,
    ]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm() or not on_gfx906():
            return False, "TorchWNA16LinearKernel is only enabled on ROCm gfx906"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Unsupported quant type {c.weight_type}"

        if c.has_g_idx:
            return False, "Activation reordering is not supported"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for name in [self.w_q_name, self.w_s_name, self.w_zp_name, self.w_gidx_name]:
            if name is None or getattr(layer, name, None) is None:
                continue
            param = getattr(layer, name)
            replace_parameter(
                layer,
                name,
                torch.nn.Parameter(param.data.contiguous(), requires_grad=False),
            )

    def _dequantize_weight(self, layer: torch.nn.Module) -> torch.Tensor:
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)
        c = self.config

        unpacked = unpack_quantized_values_into_int32(
            w_q.data, c.weight_type, packed_dim=1
        )
        unpacked = unpacked.to(w_s.dtype)

        if c.zero_points:
            assert w_zp is not None
            zp = unpack_quantized_values_into_int32(
                w_zp.data, c.weight_type, packed_dim=0
            )
            if zp.shape != w_s.shape and zp.transpose(0, 1).shape == w_s.shape:
                zp = zp.transpose(0, 1).contiguous()
            zp = zp.to(w_s.dtype)
        else:
            zp = torch.full_like(unpacked, float(c.weight_type.bias))

        if c.group_size == -1:
            scales = w_s.data.expand(-1, unpacked.shape[1])
            if c.zero_points:
                zp = zp.expand(-1, unpacked.shape[1])
        else:
            scales = w_s.data.repeat_interleave(c.group_size, dim=1)
            scales = scales[:, : unpacked.shape[1]]
            if c.zero_points:
                zp = zp.repeat_interleave(c.group_size, dim=1)
                zp = zp[:, : unpacked.shape[1]]

        return (unpacked - zp) * scales

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self._dequantize_weight(layer)
        output = torch.matmul(x, weight.transpose(0, 1))
        if bias is not None:
            output.add_(bias)
        return output
