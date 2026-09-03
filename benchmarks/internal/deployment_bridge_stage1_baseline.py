"""Measure the private offline Deployment Bridge Stage 1 checker."""

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
    CompatibilityStatus,
    inspect_deployment_compatibility,
)
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


def _inputs():
    backend = CloudBackendProfile(
        provider="anonymous",
        name="offline-simulator",
        n_wires=2,
        supports_openqasm=True,
        is_simulator=True,
    )
    package = create_deployment_package(
        fq.Circuit(2).rx(0, 0.25).cx(0, 1),
        backend=backend,
        name="anonymous-job",
        shots=100,
        qasm_version=2.0,
    )
    target = TargetCapabilities(
        target_class=TargetClass.QASM_TEXT,
        logical_qubit_capacity=2,
        physical_qubit_capacity=2,
        native_gates=tuple(
            GateCapability(name)
            for name in sorted({item.name for item in package.ir.instructions})
        ),
        measurement_results=(MeasurementResult.COUNTS,),
        artifact_profiles=(ArtifactProfile(ArtifactFormat.OPENQASM_2, "2.0"),),
        maximum_shots=4096,
        maximum_program_operations=1024,
    )
    limits = CompatibilityInspectionLimits(65536, 256, 16, 262144, 16)
    return package, target, limits


def measure_case(
    inspection_count: int, *, iterations: int, warmup: int
) -> dict[str, Any]:
    package, target, limits = _inputs()

    def inspect_many() -> str:
        final_identity = ""
        for _ in range(inspection_count):
            report = inspect_deployment_compatibility(package, target, limits)
            assert report.status is CompatibilityStatus.ELIGIBLE_FOR_VERIFIED_RECOMPILE
            final_identity = report.report_identity
        return final_identity

    for _ in range(warmup):
        inspect_many()
    gc.collect()
    timings, identities = _time(inspect_many, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        inspect_many()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "inspection_count": inspection_count,
        "eligible_inspection_p95_ms": round(percentile(timings, 0.95), 6),
        "eligible_inspection_peak_host_memory_bytes": int(peak_bytes),
        "deterministic_report_identity": len(set(identities)) == 1,
    }


def evaluate(*, iterations: int, warmup: int) -> dict[str, Any]:
    cases = [
        measure_case(count, iterations=iterations, warmup=warmup)
        for count in (10, 100, 1000, 10000)
    ]
    return {
        "schema_version": "1.0",
        "status": "measurement_only",
        "claim_scope": "Deployment Bridge Stage 1 private CPU baseline; not a budget or SLA",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "clock": "time.perf_counter_ns",
            "peak_memory": "tracemalloc",
            "pipeline": "explicit offline eligible QASM2 package inspection",
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
