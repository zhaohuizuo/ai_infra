#!/usr/bin/env python3
"""Benchmark and trace a simplified, causal, single-head attention block.

Q, K, and V are all the RMS-normalized input so the shape flow stays visible.
This is a learning experiment, not a complete Transformer attention layer.
"""

from __future__ import annotations

import argparse
import math
from typing import Callable, TypeVar

import torch

from common.benchmark import benchmark_cuda


T = TypeVar("T")


def nvtx_stage(name: str, fn: Callable[[], T]) -> T:
    torch.cuda.nvtx.range_push(name)
    try:
        return fn()
    finally:
        torch.cuda.nvtx.range_pop()


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    scale = torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + 1e-6)
    return (x32 * scale).to(x.dtype)


def qk_scores(h: torch.Tensor) -> torch.Tensor:
    return (h @ h.transpose(-1, -2)) / math.sqrt(h.shape[-1])


def masked_softmax(scores: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    return torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)


def attention(x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    h = nvtx_stage("01 RMSNorm", lambda: rmsnorm(x))
    scores = nvtx_stage("02 QK scores", lambda: qk_scores(h))
    weights = nvtx_stage("03 mask + Softmax", lambda: masked_softmax(scores, future))
    return nvtx_stage("04 weights @ V", lambda: weights @ h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "profile"), default="benchmark")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA GPU is required")
    if min(args.batch, args.tokens, args.hidden) <= 0:
        raise ValueError("batch, tokens, and hidden must be positive")

    x = torch.randn(
        (args.batch, args.tokens, args.hidden), device="cuda", dtype=torch.float16
    )
    future = torch.ones(
        (args.tokens, args.tokens), device="cuda", dtype=torch.bool
    ).triu(diagonal=1)

    h = rmsnorm(x)
    scores = qk_scores(h)
    weights = masked_softmax(scores, future)
    output = weights @ h
    assert output.shape == x.shape
    assert torch.isfinite(output).all().item()
    assert torch.allclose(
        weights.float().sum(dim=-1),
        torch.ones_like(weights[..., 0], dtype=torch.float32),
        atol=2e-3,
    )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"x/h={tuple(x.shape)}, scores/weights={tuple(scores.shape)}, dtype={x.dtype}")

    if args.mode == "benchmark":
        stages = (
            ("RMSNorm", lambda: rmsnorm(x)),
            ("QK scores", lambda: qk_scores(h)),
            ("mask + Softmax", lambda: masked_softmax(scores, future)),
            ("weights @ V", lambda: weights @ h),
            ("full attention", lambda: attention(x, future)),
        )
        for name, fn in stages:
            result = benchmark_cuda(
                fn, warmup=args.warmup, repetitions=args.repetitions
            )
            print(f"{name:16s} {result}")
    else:
        for _ in range(args.warmup):
            attention(x, future)
        torch.cuda.synchronize()
        for _ in range(args.repetitions):
            attention(x, future)
        torch.cuda.synchronize()
        print(f"Profiled {args.repetitions} attention passes after {args.warmup} warmups")


if __name__ == "__main__":
    main()
