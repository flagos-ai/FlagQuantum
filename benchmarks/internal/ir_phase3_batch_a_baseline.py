"""Measure private TargetCapabilities construction, identity, and comparison."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
import tracemalloc
from collections.abc import Callable, Sequence
from typing import Any

from flagquantum._compiler.capability_comparison import compare_target_capabilities
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
from flagquantum._compiler.target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    TargetCapabilities,
    TargetClass,
)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_edges(entry_count: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    n_qubits = max(2, math.ceil((1 + math.sqrt(1 + 4 * entry_count)) / 2))
    edges = tuple(
        (left, right)
        for left in range(n_qubits)
        for right in range(n_qubits)
        if left != right
    )[:entry_count]
    assert len(edges) == entry_count
    return n_qubits, edges


def build_target(
    n_qubits: int,
    edges: tuple[tuple[int, int], ...],
) -> TargetCapabilities:
    return TargetCapabilities(
        target_class=TargetClass.QASM_TEXT,
        logical_qubit_capacity=n_qubits,
        physical_qubit_capacity=n_qubits,
        native_gates=(GateCapability("rz"), GateCapability("sx"), GateCapability("cx")),
        measurement_results=(MeasurementResult.COUNTS,),
        artifact_profiles=(ArtifactProfile(ArtifactFormat.OPENQASM_3_STATIC, "3.0"),),
        topology=DirectedCouplingGraph(n_qubits, edges),
        maximum_shots=100000,
        maximum_program_operations=1000000,
    )


def _time(
    operation: Callable[[], str], iterations: int
) -> tuple[list[float], list[str]]:
    timings = []
    identities = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        identities.append(operation())
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return timings, identities


def measure_case(entry_count: int, *, iterations: int, warmup: int) -> dict[str, Any]:
    n_qubits, edges = build_edges(entry_count)

    def construct_and_hash() -> str:
        return build_target(n_qubits, edges).semantic_fingerprint

    target = build_target(n_qubits, edges)

    def compare() -> str:
        result = compare_target_capabilities(target, target)
        assert result.compatible
        return target.semantic_fingerprint

    for _ in range(warmup):
        construct_and_hash()
        compare()
    gc.collect()
    construct_timings, identities = _time(construct_and_hash, iterations)
    comparison_timings, compared_identities = _time(compare, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        construct_and_hash()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "capability_entry_count": entry_count,
        "physical_qubit_capacity": n_qubits,
        "construct_and_fingerprint_p95_ms": round(
            percentile(construct_timings, 0.95), 6
        ),
        "compatible_comparison_p95_ms": round(percentile(comparison_timings, 0.95), 6),
        "construct_and_fingerprint_peak_host_memory_bytes": int(peak_bytes),
        "canonical_bytes": len(target.canonical_bytes),
        "deterministic_identity": len(set(identities + compared_identities)) == 1,
    }


def evaluate(*, iterations: int, warmup: int) -> dict[str, Any]:
    cases = [
        measure_case(count, iterations=iterations, warmup=warmup)
        for count in (10, 100, 1000, 10000)
    ]
    return {
        "schema_version": "1.0",
        "status": "measurement_only",
        "claim_scope": "Phase 3 Batch A private CPU baseline; not a budget or SLA",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "clock": "time.perf_counter_ns",
            "peak_memory": "tracemalloc",
            "entry_definition": "one directed topology edge",
            "operations": "construct, canonicalize, fingerprint, compatible compare",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 5 or args.warmup < 1:
        raise ValueError("iterations must be >= 5 and warmup must be >= 1")
    print(
        json.dumps(evaluate(iterations=args.iterations, warmup=args.warmup), indent=2)
    )


if __name__ == "__main__":
    main()
