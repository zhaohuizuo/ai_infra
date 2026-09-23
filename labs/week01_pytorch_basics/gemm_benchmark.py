#!/usr/bin/env python3
"""Measure square GEMM throughput across inference-relevant dtypes."""

from __future__ import annotations

import argparse

import torch

from common.benchmark import benchmark_cuda, gemm_tflops


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DTYPES,
        default=list(DTYPES),
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA GPU")
    if any(size <= 0 for size in args.sizes):
        raise ValueError("all sizes must be positive")

    # Avoid silently using TF32 for the FP32 row in this introductory comparison.
    torch.set_float32_matmul_precision("highest")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("float32 matmul precision: highest\n")
    print(f"{'M=N=K':>8s} {'dtype':>10s} {'median ms':>12s} {'P90 ms':>12s} {'TFLOPS':>10s}")

    for size in args.sizes:
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            left = torch.randn((size, size), device="cuda", dtype=dtype)
            right = torch.randn((size, size), device="cuda", dtype=dtype)
            output = torch.empty((size, size), device="cuda", dtype=dtype)

            result = benchmark_cuda(
                lambda: torch.mm(left, right, out=output),
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            throughput = gemm_tflops(size, size, size, result.median_ms)
            print(
                f"{size:8d} {dtype_name:>10s} "
                f"{result.median_ms:12.4f} {result.p90_ms:12.4f} {throughput:10.2f}"
            )


if __name__ == "__main__":
    main()

