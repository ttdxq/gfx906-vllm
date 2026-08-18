# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass
from statistics import mean, median

import torch
from gguf import GGUFReader

from vllm import _custom_ops as ops
from vllm.model_executor.utils import set_random_seed
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

DEFAULT_TENSORS = [
    "blk.0.ffn_gate.weight",
    "blk.0.attn_qkv.weight",
    "blk.0.ffn_down.weight",
    "blk.0.ssm_out.weight",
]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    quant_type: int
    row: int
    col: int
    qweight: torch.Tensor


def _time_cuda(fn, iters: int) -> float:
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) / iters


def _time_cuda_event(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.accelerator.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters / 1000.0


def _summarize_us(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0] * 1e6:.3f}"
    return (
        f"mean={mean(values) * 1e6:.3f} "
        f"median={median(values) * 1e6:.3f} "
        f"min={min(values) * 1e6:.3f}"
    )


def _load_specs(model: str, names: list[str]) -> list[TensorSpec]:
    wanted = set(names)
    specs: list[TensorSpec] = []
    reader = GGUFReader(model)
    for tensor in reader.tensors:
        if tensor.name not in wanted:
            continue
        shape = [int(dim) for dim in tensor.shape]
        if len(shape) != 2:
            raise ValueError(f"{tensor.name} has non-matrix shape {shape}")
        col, row = shape
        qweight = torch.tensor(tensor.data, device="cuda").contiguous()
        specs.append(
            TensorSpec(
                name=tensor.name,
                quant_type=int(tensor.tensor_type),
                row=row,
                col=col,
                qweight=qweight,
            )
        )

    missing = wanted - {spec.name for spec in specs}
    if missing:
        raise ValueError(f"Missing tensors in GGUF file: {sorted(missing)}")
    return specs


def _limit_rows(spec: TensorSpec, row_limit: int | None) -> TensorSpec:
    if row_limit is None or row_limit >= spec.row:
        return spec
    if row_limit <= 0:
        raise ValueError("--row-limit must be positive")
    return TensorSpec(
        name=f"{spec.name}[:{row_limit}]",
        quant_type=spec.quant_type,
        row=row_limit,
        col=spec.col,
        qweight=spec.qweight[:row_limit].contiguous(),
    )


def _bench_spec(
    spec: TensorSpec,
    dtype: torch.dtype,
    batch_size: int,
    warmup_iters: int,
    iters: int,
    repeat: int,
    timing: str,
) -> None:
    x = torch.randn((batch_size, spec.col), dtype=dtype, device="cuda")
    quant_x = ops.ggml_quantize_row_q8_1(x)
    y = torch.empty((batch_size, spec.row), dtype=dtype, device="cuda")

    def quantize():
        ops.ggml_quantize_row_q8_1_out(x, quant_x)

    def mmvq():
        ops.ggml_mul_mat_vec_q8_out(
            spec.qweight, quant_x, y, spec.quant_type, spec.row, spec.col
        )

    def total():
        ops.ggml_quantize_row_q8_1_out(x, quant_x)
        ops.ggml_mul_mat_vec_q8_out(
            spec.qweight, quant_x, y, spec.quant_type, spec.row, spec.col
        )

    def mmq():
        ops.ggml_mul_mat_a8(spec.qweight, x, spec.quant_type, spec.row)

    for fn in (quantize, mmvq, total, mmq):
        for _ in range(warmup_iters):
            fn()

    timer = _time_cuda_event if timing == "event" else _time_cuda
    quant_times = [timer(quantize, iters) for _ in range(repeat)]
    mmvq_times = [timer(mmvq, iters) for _ in range(repeat)]
    total_times = [timer(total, iters) for _ in range(repeat)]
    mmq_times = [timer(mmq, iters) for _ in range(repeat)]

    parts = [
        f"name={spec.name}",
        f"type={spec.quant_type}",
        f"row={spec.row}",
        f"col={spec.col}",
        f"batch={batch_size}",
        f"timing={timing}",
        f"quant_us={_summarize_us(quant_times)}",
        f"mmvq_us={_summarize_us(mmvq_times)}",
        f"total_us={_summarize_us(total_times)}",
        f"mmq_total_us={_summarize_us(mmq_times)}",
    ]
    print(" ".join(parts), flush=True)


def main(args) -> None:
    set_random_seed(args.seed)
    torch.set_default_device("cuda")
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    tensor_names = args.tensor if args.tensor is not None else DEFAULT_TENSORS
    specs = _load_specs(args.model, tensor_names)
    print(args)
    for spec in specs:
        spec = _limit_rows(spec, args.row_limit)
        _bench_spec(
            spec,
            dtype,
            batch_size=args.batch_size,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            repeat=args.repeat,
            timing=args.timing,
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark real GGUF MMVQ tensors.")
    parser.add_argument(
        "--model",
        default="/root/model/Qwen3.5-4B-UD-Q4_K_XL.gguf",
        help="GGUF model file to read real quantized tensors from.",
    )
    parser.add_argument(
        "--tensor",
        action="append",
        default=None,
        help="Tensor name to benchmark. May be passed multiple times.",
    )
    parser.add_argument(
        "--dtype",
        choices=["half", "bfloat16", "float"],
        default="half",
    )
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--timing", choices=["event", "cpu"], default="event")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
