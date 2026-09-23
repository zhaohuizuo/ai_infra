"""Small, explicit CUDA benchmarking helpers.

The implementation intentionally exposes warmup, synchronization, and percentile
calculation so the measurement methodology stays visible while learning.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    median_ms: float
    p90_ms: float
    minimum_ms: float
    repetitions: int

    def __str__(self) -> str:
        return (
            f"median={self.median_ms:.4f} ms, "
            f"p90={self.p90_ms:.4f} ms, "
            f"min={self.minimum_ms:.4f} ms, "
            f"repetitions={self.repetitions}"
        )


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty list."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_cuda(
    fn: Callable[[], Any],
    *,
    warmup: int = 25,
    repetitions: int = 100,
) -> BenchmarkResult:
    """Measure a callable on the current CUDA stream using CUDA events."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_cuda")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]

    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()

    torch.cuda.synchronize()
    timings_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return BenchmarkResult(
        median_ms=statistics.median(timings_ms),
        p90_ms=_percentile(timings_ms, 0.90),
        minimum_ms=min(timings_ms),
        repetitions=repetitions,
    )


def effective_bandwidth_gbps(bytes_moved: int, latency_ms: float) -> float:
    """Convert bytes moved and milliseconds to decimal GB/s."""
    if bytes_moved < 0:
        raise ValueError("bytes_moved must be non-negative")
    if latency_ms <= 0:
        raise ValueError("latency_ms must be positive")
    return bytes_moved / (latency_ms * 1e-3) / 1e9


def gemm_tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    """Return 2*M*N*K FLOPs divided by elapsed time in decimal TFLOPS."""
    if min(m, n, k) <= 0:
        raise ValueError("GEMM dimensions must be positive")
    if latency_ms <= 0:
        raise ValueError("latency_ms must be positive")
    return (2 * m * n * k) / (latency_ms * 1e-3) / 1e12

