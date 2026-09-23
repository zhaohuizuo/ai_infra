#!/usr/bin/env python3
"""Validate the remote NVIDIA environment used by the learning labs."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"ERROR: {error}"
    return completed.stdout.strip()


def main() -> int:
    failures: list[str] = []
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")

    if shutil.which("nvidia-smi"):
        query = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
        print(f"nvidia-smi: {command_output(query)}")
    else:
        failures.append("nvidia-smi was not found")

    if shutil.which("nvcc"):
        version_lines = command_output(["nvcc", "--version"]).splitlines()
        print(f"nvcc: {version_lines[-1] if version_lines else 'unknown'}")
    else:
        failures.append("nvcc was not found; CUDA C++ labs cannot be compiled")

    try:
        import torch
    except ImportError:
        failures.append("PyTorch is not installed")
    else:
        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")
        if not torch.cuda.is_available():
            failures.append("torch.cuda.is_available() is False")
        else:
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            memory_gib = properties.total_memory / 1024**3
            print(f"GPU: {properties.name}")
            print(f"Compute capability: {properties.major}.{properties.minor}")
            print(f"VRAM: {memory_gib:.1f} GiB")

            left = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
            right = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
            result = left @ right
            torch.cuda.synchronize()
            if not torch.isfinite(result).all().item():
                failures.append("CUDA smoke-test GEMM produced non-finite values")
            else:
                print("CUDA smoke test: PASS")

    if importlib.util.find_spec("triton") is None:
        failures.append("Triton is not installed")
    else:
        import triton

        print(f"Triton: {triton.__version__}")

    if failures:
        print("\nEnvironment check: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nEnvironment check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
