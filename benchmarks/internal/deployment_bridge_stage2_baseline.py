"""Measure the private explicit Deployment Bridge Stage 2 compile dry-run."""

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

import flagquantum as fq
from flagquantum._compiler.deployment_compatibility import (
    CompatibilityInspectionLimits,
)
from flagquantum._compiler.deployment_dry_run import (
    DeploymentDryRunPolicy,
    DeploymentDryRunStatus,
    dry_run_deployment_bridge,
)
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
from flagquantum._compiler.target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    TargetCapabilities,
    TargetClass,
)
from flagquantum.deployment.cloud import (
    CloudBackendProfile,
    create_deployment_package,
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


def _build_inputs(operation_count: int):
    circuit = fq.Circuit(4)
    for index in range(operation_count):
        selector = index % 4
        if selector == 0:
            circuit.rx(index % 4, 0.013 * (index + 1))
        elif selector == 1:
            circuit.ry(index % 4, -0.017 * (index + 1))
        elif selector == 2:
            circuit.rz(index % 4, 0.019 * (index + 1))
        else:
            circuit.cx(index % 4, (index + 1) % 4)
    backend = CloudBackendProfile(
        provider="anonymous",
        name="offline-simulator",
        n_wires=4,
        supports_openqasm=True,
        is_simulator=True,
    )
    package = create_deployment_package(
        circuit,
        backend=backend,
        name="anonymous-job",
        shots=100,
        qasm_version=2.0,
    )
    target = TargetCapabilities(
        target_class=TargetClass.QASM_TEXT,
        logical_qubit_capacity=4,
        physical_qubit_capacity=4,
        native_gates=tuple(GateCapability(name) for name in ("rx", "ry", "rz", "cx")),
        measurement_results=(MeasurementResult.COUNTS,),
        artifact_profiles=(ArtifactProfile(ArtifactFormat.OPENQASM_2, "2.0"),),
        topology=DirectedCouplingGraph(
            4,
            tuple(
                (left, right)
                for left in range(4)
                for right in range(4)
                if left != right
            ),
        ),
        maximum_shots=4096,
        maximum_program_operations=50000,
    )
    policy = DeploymentDryRunPolicy(
        CompatibilityInspectionLimits(16_777_216, 1024, 32, 1_048_576, 32),
        maximum_source_operations=20000,
        maximum_compiled_operations=50000,
        maximum_artifact_bytes=16_777_216,
        maximum_evidence_bytes=65536,
        maximum_stage_duration_ns=60_000_000_000,
        maximum_total_duration_ns=120_000_000_000,
    )
    return package, target, policy


def _time(
    operation: Callable[[], str], iterations: int
) -> tuple[list[float], list[str]]:
    timings = []
    identities = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(iterations):
            started = time.perf_counter_ns()
            identities.append(operation())
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return timings, identities


def measure_case(
    operation_count: int, *, iterations: int, warmup: int
) -> dict[str, Any]:
    package, target, policy = _build_inputs(operation_count)

    def dry_run() -> str:
        report = dry_run_deployment_bridge(package, target, policy)
        assert report.status is DeploymentDryRunStatus.READY_FOR_OPERATOR_REVIEW
        return report.evidence_identity

    for _ in range(warmup):
        dry_run()
    gc.collect()
    timings, identities = _time(dry_run, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        dry_run()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "operation_count": operation_count,
        "dry_run_p95_ms": round(percentile(timings, 0.95), 6),
        "dry_run_peak_host_memory_bytes": int(peak_bytes),
        "deterministic_evidence_identity": len(set(identities)) == 1,
    }


def evaluate(*, iterations: int, warmup: int) -> dict[str, Any]:
    cases = [
        measure_case(count, iterations=iterations, warmup=warmup)
        for count in (10, 100, 1000, 10000)
    ]
    return {
        "schema_version": "1.0",
        "status": "measurement_only",
        "claim_scope": "Deployment Bridge Stage 2 private CPU baseline; not a budget or SLA",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "clock": "time.perf_counter_ns",
            "timing_gc": "disabled after explicit collection",
            "peak_memory": "tracemalloc",
            "pipeline": "explicit offline OpenQASM2 compile dry-run",
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
