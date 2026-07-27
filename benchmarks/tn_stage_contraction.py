"""Benchmark one canonical TN contraction stage on CUDA.

This isolates the operation used by compiled TN stages and compares the fused
complex Triton BMM route with native complex ``torch.einsum``.  It is local
single-device engineering evidence, not a scalability benchmark.
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

from flagquantum.simulation.triton_kernels import fused_complex_bmm  # noqa: E402


def _measure(operation, *, warmup: int, iterations: int, repeats: int) -> dict:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / iterations)
    return {
        "median_step_seconds": statistics.median(samples),
        "samples_seconds": samples,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(7)
    device = torch.device("cuda")
    left = torch.randn(
        args.batch, args.m, args.k, device=device, dtype=torch.complex64
    ).requires_grad_(True)
    right = torch.randn(
        args.batch, args.k, args.n, device=device, dtype=torch.complex64
    ).requires_grad_(True)
    gradient = torch.randn(
        args.batch, args.m, args.n, device=device, dtype=torch.complex64
    )
    equation = "zab,zbc->zac"

    def fused() -> None:
        output = fused_complex_bmm(left, right)
        torch.autograd.grad(output, (left, right), gradient)

    def native() -> None:
        output = torch.einsum(equation, left, right)
        torch.autograd.grad(output, (left, right), gradient)

    inference_left = left.detach()
    inference_right = right.detach()

    def fused_forward() -> None:
        fused_complex_bmm(inference_left, inference_right)

    def native_forward() -> None:
        torch.einsum(equation, inference_left, inference_right)

    started = time.perf_counter()
    fused()
    torch.cuda.synchronize()
    fused_cold = time.perf_counter() - started
    actual = fused_complex_bmm(left, right)
    reference = torch.einsum(equation, left, right)

    fused_result = _measure(
        fused,
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
    fused_forward_result = _measure(
        fused_forward,
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
    fused_seconds = fused_result["median_step_seconds"]
    native_seconds = native_result["median_step_seconds"]
    payload = {
        "benchmark": "tn_stage_contraction",
        "benchmark_evidence_class": "local",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "shape": [args.batch, args.m, args.k, args.n],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "fused_cold_seconds": fused_cold,
        "fused": fused_result,
        "native": native_result,
        "speedup": native_seconds / fused_seconds,
        "fused_forward": fused_forward_result,
        "native_forward": native_forward_result,
        "forward_speedup": (
            native_forward_result["median_step_seconds"]
            / fused_forward_result["median_step_seconds"]
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
