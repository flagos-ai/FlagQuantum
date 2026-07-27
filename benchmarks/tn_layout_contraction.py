"""Benchmark direct high-rank addressing against materialized canonical BMM.

This is local single-device engineering evidence, not a scalability benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagquantum.simulation.triton_kernels import (  # noqa: E402
    fused_complex_bmm,
    fused_complex_layout_bmm,
)


def _measure(operation, *, warmup: int, iterations: int, repeats: int) -> dict:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    samples = []
    peaks = []
    peak_increments = []
    for _ in range(repeats):
        baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / iterations)
        peak = int(torch.cuda.max_memory_allocated())
        peaks.append(peak)
        peak_increments.append(peak - baseline)
    return {
        "median_step_seconds": statistics.median(samples),
        "samples_seconds": samples,
        "peak_memory_bytes": max(peaks),
        "peak_increment_bytes": max(peak_increments),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k-left", type=int, default=8)
    parser.add_argument("--k-right", type=int, default=16)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(7)
    device = torch.device("cuda")
    # Canonical orders are zabc and zcbd.  The physical orders below force
    # non-view permutations on both operands.
    left = torch.randn(
        args.m,
        args.batch,
        args.k_right,
        args.k_left,
        device=device,
        dtype=torch.complex64,
        requires_grad=True,
    )
    right = torch.randn(
        args.k_right,
        args.batch,
        args.n,
        args.k_left,
        device=device,
        dtype=torch.complex64,
        requires_grad=True,
    )
    left_permutation = (1, 0, 3, 2)
    right_permutation = (1, 3, 0, 2)
    shapes = (
        (args.batch,),
        (args.m,),
        (args.k_left, args.k_right),
        (args.n,),
    )
    reduction = args.k_left * args.k_right
    gradient = torch.randn(
        args.batch, args.m, args.n, device=device, dtype=torch.complex64
    )

    def direct() -> None:
        output = fused_complex_layout_bmm(
            left, right, left_permutation, right_permutation, shapes
        )
        torch.autograd.grad(output, (left, right), gradient)

    def materialized() -> None:
        left_matrix = left.permute(left_permutation).reshape(
            args.batch, args.m, reduction
        )
        right_matrix = right.permute(right_permutation).reshape(
            args.batch, reduction, args.n
        )
        output = fused_complex_bmm(left_matrix, right_matrix)
        torch.autograd.grad(output, (left, right), gradient)

    equation = "azcb,czdb->zad"

    def native() -> None:
        output = torch.einsum(equation, left, right)
        torch.autograd.grad(output, (left, right), gradient)

    inference_left = left.detach()
    inference_right = right.detach()

    def direct_forward() -> None:
        fused_complex_layout_bmm(
            inference_left,
            inference_right,
            left_permutation,
            right_permutation,
            shapes,
        )

    def native_forward() -> None:
        torch.einsum(equation, inference_left, inference_right)

    started = time.perf_counter()
    direct()
    torch.cuda.synchronize()
    direct_cold = time.perf_counter() - started
    actual = fused_complex_layout_bmm(
        left, right, left_permutation, right_permutation, shapes
    )
    reference = torch.bmm(
        left.permute(left_permutation).reshape(args.batch, args.m, reduction),
        right.permute(right_permutation).reshape(args.batch, reduction, args.n),
    )
    direct_result = _measure(
        direct,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    materialized_result = _measure(
        materialized,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    native_result = _measure(
        native,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    direct_forward_result = _measure(
        direct_forward,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    native_forward_result = _measure(
        native_forward,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    payload = {
        "benchmark": "tn_layout_contraction",
        "benchmark_evidence_class": "local",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "logical_shape": [
            args.batch,
            args.m,
            reduction,
            args.n,
        ],
        "physical_left_shape": list(left.shape),
        "physical_right_shape": list(right.shape),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "direct_cold_seconds": direct_cold,
        "direct_layout": direct_result,
        "materialized_layout": materialized_result,
        "native": native_result,
        "direct_forward": direct_forward_result,
        "native_forward": native_forward_result,
        "speedup": (
            materialized_result["median_step_seconds"]
            / direct_result["median_step_seconds"]
        ),
        "peak_memory_ratio": (
            direct_result["peak_memory_bytes"]
            / materialized_result["peak_memory_bytes"]
        ),
        "native_over_direct": (
            native_result["median_step_seconds"]
            / direct_result["median_step_seconds"]
        ),
        "native_over_direct_forward": (
            native_forward_result["median_step_seconds"]
            / direct_forward_result["median_step_seconds"]
        ),
        "max_abs_error": float(torch.max(torch.abs(actual - reference)).item()),
    }
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
