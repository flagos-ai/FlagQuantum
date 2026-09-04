"""Measure the legacy CircuitIR path before internal QuantumIR work begins."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable, Sequence
from typing import Any

import torch

import flagquantum as fq
from flagquantum.compiler import simple_compile


def _build_circuit(gate_count: int) -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex64)
    builders: tuple[Callable[[int], None], ...] = (
        lambda index: circuit.rx(index % 4, theta=0.013 * (index + 1)),
        lambda index: circuit.cx(index % 4, (index + 1) % 4),
        lambda index: circuit.ry((index + 2) % 4, theta=-0.017 * (index + 1)),
        lambda index: circuit.cz((index + 1) % 4, (index + 3) % 4),
        lambda index: circuit.rz((index + 3) % 4, theta=0.019 * (index + 1)),
    )
    for index in range(int(gate_count)):
        builders[index % len(builders)](index)
    return circuit


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timings_ms(
    operation: Callable[[], Any], iterations: int, warmup: int
) -> dict[str, float]:
    gc.collect()
    for _ in range(warmup):
        operation()
    values = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(iterations):
            started = time.perf_counter_ns()
            operation()
            values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "min": round(min(values), 6),
        "p50": round(statistics.median(values), 6),
        "p95": round(_percentile(values, 0.95), 6),
        "max": round(max(values), 6),
    }


def _peak_bytes(operation: Callable[[], Any]) -> int:
    tracemalloc.start()
    try:
        operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def _case(
    gate_count: int, *, iterations: int, warmup: int, run_max_gates: int
) -> dict[str, Any]:
    circuit = _build_circuit(gate_count)
    ir = circuit.to_ir()
    operations: dict[str, Callable[[], Any]] = {
        "circuit_build_to_ir": lambda: _build_circuit(gate_count).to_ir(),
        "ir_to_json": ir.to_json,
        "legacy_simple_compile": lambda: simple_compile(ir),
        "legacy_plan": lambda: fq.plan(ir),
    }
    if gate_count <= run_max_gates:
        operations["legacy_run"] = lambda: fq.run(ir)

    return {
        "gate_count": int(gate_count),
        "serialized_ir_bytes": len(ir.to_json().encode("utf-8")),
        "timings_ms": {
            name: _timings_ms(operation, iterations, warmup)
            for name, operation in operations.items()
        },
        "peak_host_memory_bytes": {
            name: _peak_bytes(operation) for name, operation in operations.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-counts", nargs="+", type=int, default=(10, 100, 1000, 10000)
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--run-max-gates", type=int, default=1000)
    parser.add_argument("--source-commit", default="working-tree")
    args = parser.parse_args()
    if args.iterations < 3:
        raise ValueError("iterations must be at least 3")
    if args.warmup < 1:
        raise ValueError("warmup must be at least 1")
    if any(count < 1 for count in args.gate_counts):
        raise ValueError("gate counts must be positive")

    torch.set_num_threads(1)
    payload = {
        "schema_version": "1.0",
        "status": "phase0_development_baseline",
        "claim_scope": "local CPU characterization; not a scalability claim",
        "source_commit": str(args.source_commit),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_num_threads": torch.get_num_threads(),
            "dtype": "complex64",
            "n_wires": 4,
        },
        "method": {
            "iterations": int(args.iterations),
            "warmup": int(args.warmup),
            "run_max_gates": int(args.run_max_gates),
            "clock": "time.perf_counter_ns",
            "peak_memory": "tracemalloc",
            "timing_gc": "disabled after explicit collection",
        },
        "cases": [
            _case(
                count,
                iterations=args.iterations,
                warmup=args.warmup,
                run_max_gates=args.run_max_gates,
            )
            for count in args.gate_counts
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
