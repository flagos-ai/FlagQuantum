"""Measure private executable-artifact sealing and verification."""

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

from flagquantum._compiler.executable_artifact import (
    seal_executable_artifact,
    verify_executable_artifact,
)
from flagquantum._compiler.target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    TargetCapabilities,
    TargetClass,
)
from flagquantum._compiler.target_ir import TargetIR, TargetOperation

COMPILATION_IDENTITY = "c" * 64


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_case(operation_count: int):
    profile = ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0")
    target = TargetCapabilities(
        target_class=TargetClass.LOCAL_RUNTIME,
        logical_qubit_capacity=1,
        physical_qubit_capacity=1,
        native_gates=(GateCapability("rx"),),
        measurement_results=(MeasurementResult.STATE,),
        artifact_profiles=(profile,),
        maximum_program_operations=operation_count,
    )
    target_ir = TargetIR(
        "a" * 64,
        target.semantic_fingerprint,
        (0,),
        tuple(
            TargetOperation("rx", (0,), {"theta": (index % 17) / 17.0})
            for index in range(operation_count)
        ),
    )
    payload = json.dumps(
        target_ir.canonical(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return profile, target, target_ir, payload


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
    profile, target, target_ir, payload = build_case(operation_count)

    def seal_and_verify() -> str:
        sealed = seal_executable_artifact(
            target_ir,
            target,
            profile,
            payload,
            compilation_identity=COMPILATION_IDENTITY,
        )
        assert sealed.ok and sealed.artifact is not None
        verified = verify_executable_artifact(sealed.artifact, target_ir, target)
        assert verified.ok
        return sealed.artifact.artifact_identity

    for _ in range(warmup):
        seal_and_verify()
    gc.collect()
    timings, identities = _time(seal_and_verify, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        seal_and_verify()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "operation_count": operation_count,
        "payload_bytes": len(payload),
        "seal_and_verify_p95_ms": round(percentile(timings, 0.95), 6),
        "seal_and_verify_peak_host_memory_bytes": int(peak_bytes),
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
        "claim_scope": "Phase 3 Batch C private CPU baseline; not a budget or SLA",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "clock": "time.perf_counter_ns",
            "peak_memory": "tracemalloc",
            "pipeline": "canonical payload validation, sealing, identity, and verification",
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
