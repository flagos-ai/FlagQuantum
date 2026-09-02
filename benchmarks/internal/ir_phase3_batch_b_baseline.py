"""Measure private QuantumIR-to-TargetIR legalization performance."""

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

from flagquantum._compiler.importers.circuit_ir import import_circuit_ir
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
from flagquantum._compiler.target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    TargetCapabilities,
    TargetClass,
)
from flagquantum._compiler.target_legalization import legalize_quantum_module
from flagquantum.core.ir import CircuitIR, Instruction


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_module(operation_count: int):
    templates = (
        Instruction("rx", (0,), {"theta": 0.13}),
        Instruction("ry", (1,), {"theta": -0.17}),
        Instruction("cx", (1, 2)),
        Instruction("rz", (3,), {"theta": 0.19}),
        Instruction("cx", (2, 3)),
    )
    source = CircuitIR(
        4,
        tuple(templates[index % len(templates)] for index in range(operation_count)),
        dtype="complex128",
    )
    imported = import_circuit_ir(source)
    assert imported.ok and imported.imported is not None
    return imported.imported.module


def build_target() -> TargetCapabilities:
    edges = tuple(
        (left, right) for left in range(4) for right in range(4) if left != right
    )
    return TargetCapabilities(
        target_class=TargetClass.LOCAL_RUNTIME,
        logical_qubit_capacity=4,
        physical_qubit_capacity=4,
        native_gates=(
            GateCapability("rx"),
            GateCapability("ry"),
            GateCapability("rz"),
            GateCapability("cx"),
        ),
        measurement_results=(MeasurementResult.STATE,),
        artifact_profiles=(ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0"),),
        topology=DirectedCouplingGraph(4, edges),
        maximum_program_operations=10000,
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


def measure_case(
    operation_count: int, *, iterations: int, warmup: int
) -> dict[str, Any]:
    module = build_module(operation_count)
    target = build_target()

    def legalize() -> str:
        result = legalize_quantum_module(
            module,
            target,
            required_results=(MeasurementResult.STATE,),
        )
        assert result.ok and result.target_ir is not None
        return result.target_ir.target_program_identity

    for _ in range(warmup):
        legalize()
    gc.collect()
    timings, identities = _time(legalize, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        legalize()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "operation_count": operation_count,
        "legalization_p95_ms": round(percentile(timings, 0.95), 6),
        "legalization_peak_host_memory_bytes": int(peak_bytes),
        "deterministic_identity": len(set(identities)) == 1,
    }


def evaluate(*, iterations: int, warmup: int) -> dict[str, Any]:
    cases = [
        measure_case(count, iterations=iterations, warmup=warmup)
        for count in (10, 100, 1000, 10000)
    ]
    return {
        "schema_version": "1.0",
        "status": "measurement_only",
        "claim_scope": "Phase 3 Batch B private CPU baseline; not a budget or SLA",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "clock": "time.perf_counter_ns",
            "peak_memory": "tracemalloc",
            "pipeline": "verified QuantumIR to immutable TargetIR legalization and identity",
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
