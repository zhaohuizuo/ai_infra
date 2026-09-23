#!/usr/bin/env python3
"""Inspect shape, stride, storage aliasing, view, and reshape behavior."""

from __future__ import annotations

import argparse

import torch


def storage_pointer(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def describe(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name:12s} shape={str(tuple(tensor.shape)):12s} "
        f"stride={str(tensor.stride()):12s} "
        f"offset={tensor.storage_offset():2d} "
        f"contiguous={str(tensor.is_contiguous()):5s} "
        f"storage_ptr={storage_pointer(tensor)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    base = torch.arange(24, device=args.device, dtype=torch.float32)
    matrix = base.reshape(2, 3, 4)
    transposed = matrix.transpose(1, 2)
    sliced = matrix[:, :, 1:]
    reshaped = transposed.reshape(-1)
    materialized = transposed.contiguous()

    print(f"device={args.device}\n")
    describe("base", base)
    describe("matrix", matrix)
    describe("transposed", transposed)
    describe("sliced", sliced)
    describe("reshaped", reshaped)
    describe("materialized", materialized)

    print("\nStorage relationships")
    print(f"matrix aliases base:          {storage_pointer(matrix) == storage_pointer(base)}")
    print(
        "transposed aliases matrix:   "
        f"{storage_pointer(transposed) == storage_pointer(matrix)}"
    )
    print(
        "reshaped aliases transposed: "
        f"{storage_pointer(reshaped) == storage_pointer(transposed)}"
    )

    print("\nAttempt transposed.view(-1)")
    try:
        transposed.view(-1)
    except RuntimeError as error:
        print(f"view failed as expected: {str(error).splitlines()[0]}")
    else:
        print("view succeeded; inspect the chosen shape and stride assumptions")


if __name__ == "__main__":
    main()

