#!/usr/bin/env python3
"""Reproducible local dense-statevector optimization benchmark.

This benchmark compares the public optimized ``Circuit.state`` path with a
deliberately unfused dense-matrix reference over the exact same FlagQuantum IR.
It is local performance evidence only and can never promote a distributed
scalability claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402, I001
from flagquantum.circuit import _apply_matrix, _gate_matrix  # noqa: E402


SCHEMA = "flagquantum.statevector.local_performance.v1"


def build_workload(
    *, n_wires: int, batch_size: int, layers: int, device: str
) -> fq.Circuit:
    generator = torch.Generator(device=device).manual_seed(4417)
    initial = torch.randn(
        batch_size,
        2**n_wires,
        dtype=torch.complex64,
        device=device,
        generator=generator,
    )
    initial = initial / torch.linalg.vector_norm(initial, dim=-1, keepdim=True)
    values = torch.linspace(
        -0.37,
        0.41,
        steps=layers * n_wires * 3,
        dtype=torch.float32,
        device=device,
    ).reshape(layers, n_wires, 3)
    circuit = fq.Circuit(n_wires, bsz=batch_size, device=device, inputs=initial)
    for layer in range(layers):
        for wire in range(n_wires):
            circuit.rx(wire, values[layer, wire, 0])
            circuit.ry(wire, values[layer, wire, 1])
            circuit.rz(wire, values[layer, wire, 2])
        circuit.x(layer % n_wires)
        circuit.y((layer + 1) % n_wires)
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
        if n_wires > 1:
            circuit.swap(0, n_wires - 1)
            circuit.rzz(0, n_wires - 1, values[layer, 0, 0] * 0.25)
    return circuit


def sequential_reference(circuit: fq.Circuit) -> torch.Tensor:
    state = circuit.initial_state()
    for instruction in circuit.to_ir().instructions:
        matrix = _gate_matrix(
            instruction,
            bsz=state.shape[0],
            device=state.device,
            dtype=state.dtype,
        )
        state = _apply_matrix(
            state, matrix, instruction.wires, circuit.n_wires
        )
    return state


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _measure(
    function: Callable[[], torch.Tensor], *, warmup: int, iterations: int, device: str
) -> tuple[tuple[float, ...], torch.Tensor]:
    output = function()
    for _ in range(warmup):
        output = function()
    _sync(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = function()
        _sync(device)
        samples.append(time.perf_counter() - started)
    return tuple(samples), output


def _timing(samples: tuple[float, ...]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "coefficient_of_variation": (
            statistics.pstdev(samples) / mean if len(samples) > 1 and mean else 0.0
        ),
    }


def run_benchmark(
    *,
    n_wires: int,
    batch_size: int,
    layers: int,
    device: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if n_wires < 2 or batch_size < 1 or layers < 1:
        raise ValueError("n_wires >= 2, batch_size >= 1 and layers >= 1 required")
    if warmup < 0 or iterations < 2:
        raise ValueError("warmup must be non-negative and iterations must be >= 2")
    circuit = build_workload(
        n_wires=n_wires,
        batch_size=batch_size,
        layers=layers,
        device=device,
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    optimized_samples, optimized = _measure(
        lambda: circuit.state(refresh=True),
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    optimized_peak_memory = (
        int(torch.cuda.max_memory_allocated(torch.device(device)))
        if str(device).startswith("cuda")
        else 0
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    reference_samples, reference = _measure(
        lambda: sequential_reference(circuit),
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    reference_peak_memory = (
        int(torch.cuda.max_memory_allocated(torch.device(device)))
        if str(device).startswith("cuda")
        else 0
    )
    optimized_timing = _timing(optimized_samples)
    reference_timing = _timing(reference_samples)
    max_error = float(torch.max(torch.abs(optimized - reference)).item())
    tolerance = 2e-5
    return {
        "schema_version": SCHEMA,
        "benchmark": "local_dense_statevector_optimization",
        "artifact_class": "measured_local_run",
        "claim_evidence_type": "local_performance",
        "distribution_semantics": "single_device_fast_path",
        "world_size": 1,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(torch.device(device))
                if str(device).startswith("cuda")
                else platform.processor() or "cpu"
            ),
            "seed": 4417,
        },
        "workload": {
            "n_wires": n_wires,
            "batch_size": batch_size,
            "layers": layers,
            "gate_count": len(circuit),
            "dtype": "complex64",
            "warmup": warmup,
            "iterations": iterations,
        },
        "correctness": {
            "passed": max_error <= tolerance,
            "max_abs_error": max_error,
            "absolute_tolerance": tolerance,
            "reference": "same_ir_sequential_dense_matrix_application",
        },
        "optimized": {
            **optimized_timing,
            "peak_memory_allocated_bytes": optimized_peak_memory,
            "runtime": dict(circuit._last_statevector_runtime),
        },
        "sequential_reference": {
            **reference_timing,
            "peak_memory_allocated_bytes": reference_peak_memory,
        },
        "speedup": (
            reference_timing["median_seconds"] / optimized_timing["median_seconds"]
        ),
        "regression_thresholds": {
            "max_coefficient_of_variation": 0.20,
            "minimum_sample_count": 2,
        },
        "stability_gate_passed": (
            optimized_timing["coefficient_of_variation"] <= 0.20
            and len(optimized_samples) >= 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        n_wires=args.n_wires,
        batch_size=args.batch_size,
        layers=args.layers,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
    if not payload["correctness"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
