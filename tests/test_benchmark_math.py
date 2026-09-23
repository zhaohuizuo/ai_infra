from __future__ import annotations

import pytest

from common.benchmark import _percentile, effective_bandwidth_gbps, gemm_tflops


def test_percentile_interpolates() -> None:
    assert _percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)
    assert _percentile([3.0], 0.9) == pytest.approx(3.0)


def test_bandwidth_conversion() -> None:
    assert effective_bandwidth_gbps(1_000_000_000, 1000.0) == pytest.approx(1.0)


def test_gemm_tflops_conversion() -> None:
    assert gemm_tflops(1000, 1000, 1000, 2.0) == pytest.approx(1.0)


@pytest.mark.parametrize("latency", [0.0, -1.0])
def test_invalid_latency(latency: float) -> None:
    with pytest.raises(ValueError):
        effective_bandwidth_gbps(1, latency)
    with pytest.raises(ValueError):
        gemm_tflops(1, 1, 1, latency)

