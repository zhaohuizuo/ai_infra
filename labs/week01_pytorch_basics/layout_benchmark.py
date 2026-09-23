#!/usr/bin/env python3
"""Compare contiguous and strided device-to-device copies."""

from __future__ import annotations

import argparse

import torch

from common.benchmark import benchmark_cuda, effective_bandwidth_gbps


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA GPU")

    dtype = DTYPES[args.dtype]
    source = torch.randn((args.rows, args.cols), device="cuda", dtype=dtype)
    contiguous_output = torch.empty_like(source)
    transposed = source.t()
    strided_output = torch.empty(
        transposed.shape,
        device="cuda",
        dtype=dtype,
    )

    contiguous_result = benchmark_cuda(
        lambda: contiguous_output.copy_(source),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    strided_result = benchmark_cuda(
        lambda: strided_output.copy_(transposed),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    # One full input read plus one full output write.
    bytes_moved = 2 * source.numel() * source.element_size()
    contiguous_bandwidth = effective_bandwidth_gbps(
        bytes_moved, contiguous_result.median_ms
    )
    strided_bandwidth = effective_bandwidth_gbps(bytes_moved, strided_result.median_ms)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"shape=({args.rows}, {args.cols}), dtype={args.dtype}")
    print(f"source stride:     {source.stride()}")
    print(f"transposed stride: {transposed.stride()}")
    print(f"contiguous copy: {contiguous_result}, {contiguous_bandwidth:.1f} GB/s")
    print(f"strided copy:    {strided_result}, {strided_bandwidth:.1f} GB/s")
    print(f"bandwidth ratio (contiguous/strided): {contiguous_bandwidth / strided_bandwidth:.2f}x")


if __name__ == "__main__":
    main()

