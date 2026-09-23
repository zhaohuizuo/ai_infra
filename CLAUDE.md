# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal 16-week AI Infra learning lab (GPU kernels + LLM inference), not a product codebase. Docs (README, ROADMAP, PROGRESS, per-week READMEs/NOTES) are written in Chinese; keep new docs in Chinese to match. `ROADMAP.md` defines the weekly topics and stage gates; `PROGRESS.md` tracks weekly status and holds the weekly retrospective template.

Two environments:
- **Local**: Apple Silicon macOS — editing and writing reports only. No CUDA here.
- **Remote**: Linux + NVIDIA RTX A6000 48GB (Ampere, sm_86) — where all CUDA/Triton experiments actually run. Hopper-only features (e.g. FP8) are out of scope by design.

## Commands

Run everything from the repo root as modules (`python -m ...`) so `common` is importable — there is no package install / `pyproject.toml`.

```bash
# Environment check on the GPU server (nvidia-smi, nvcc, torch CUDA, Triton, smoke GEMM)
python scripts/check_gpu_env.py

# Tests (use `python -m pytest` so the repo root is on sys.path; torch must be installed
# because common/benchmark.py imports it, though these tests don't need a GPU)
python -m pytest
python -m pytest tests/test_benchmark_math.py::test_percentile_interpolates

# Week 1 labs
python -m labs.week01_pytorch_basics.tensor_layout --device cuda   # also works with --device cpu
python -m labs.week01_pytorch_basics.layout_benchmark [--rows N --cols N --dtype float32]
python -m labs.week01_pytorch_basics.gemm_benchmark [--sizes 1024 2048 --dtypes float16 bfloat16]

# Nsight Systems profiling (outputs go in profiles/, which is gitignored)
nsys profile -o profiles/week01_gemm python -m labs.week01_pytorch_basics.gemm_benchmark --sizes 2048
```

PyTorch is intentionally not pinned in `requirements-dev.txt` — it is installed per the server's CUDA/driver version. Python 3.11 recommended.

## Architecture

- `common/benchmark.py` — the shared measurement primitive every lab uses. `benchmark_cuda(fn, warmup, repetitions)` runs warmup, synchronizes, times each repetition with per-iteration CUDA Events, and returns a `BenchmarkResult` (median / P90 / min). `effective_bandwidth_gbps` and `gemm_tflops` convert latency into throughput (decimal GB/s, TFLOPS = 2·M·N·K). The code is deliberately explicit rather than using `torch.utils.benchmark` / `triton.testing` so the methodology stays visible — preserve that.
- `labs/weekNN_<topic>/` — one package per week, each with runnable scripts plus `README.md` (daily plan + acceptance checklist) and `NOTES.md` (lab-report template the learner fills in). Lab scripts are standalone argparse CLIs that raise if CUDA is unavailable, allocate outputs up front, and pass a lambda to `benchmark_cuda`.
- New week directories are created only when that stage begins — don't pre-generate scaffolding for future weeks.

## Benchmarking rules (from README)

Every performance claim must have: a correctness reference, warmup + GPU sync, explicit shape/dtype/device/software versions, median and P90 over ≥100 measurements, bandwidth or TFLOPS in addition to latency, and profiler evidence or a testable hypothesis. Never time async CUDA work with `time.time()`. For FP32 GEMM comparisons, set `torch.set_float32_matmul_precision("highest")` to avoid silent TF32.

Per ROADMAP, before week 12 avoid going deep on hand-written PTX/SASS, CUTLASS/CuTe internals, RDMA, large-scale training, or multi-node scheduling.
