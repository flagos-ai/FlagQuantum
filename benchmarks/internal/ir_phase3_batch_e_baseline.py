"""Measure the private three-family provider conformance suite."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
import tracemalloc
from collections.abc import Callable, Sequence
from typing import Any

from flagquantum._compiler.executable_artifact import seal_executable_artifact
from flagquantum._compiler.provider_conformance import (
    ConformanceCase,
    ConformanceTargetFamily,
    LocalConformanceDriver,
    OfflineConformanceDriver,
    ProviderExtension,
    ProviderExtensionEntry,
    run_conformance_case,
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

ADAPTER_IDENTITY = "d" * 64
COMPILATION_IDENTITY = "c" * 64
RESULT_PAYLOAD = b"flagquantum-conformance-result-v1"


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _payload(profile: ArtifactProfile, target_ir: TargetIR) -> bytes:
    if profile.format is ArtifactFormat.RUNTIME_PLAN:
        return json.dumps(
            target_ir.canonical(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    if profile.format is ArtifactFormat.OPENQASM_2:
        return (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(0.25) q[0];\n'
        ).encode("utf-8")
    return b"Y2M Q0\nRZ Q0 0.25\nY2P Q0\n"


def build_cases() -> tuple[ConformanceCase, ...]:
    specifications = (
        (
            ConformanceTargetFamily.LOCAL_RUNTIME_PLAN,
            TargetClass.LOCAL_RUNTIME,
            ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0"),
        ),
        (
            ConformanceTargetFamily.SYNTHETIC_QASM_TEXT,
            TargetClass.QASM_TEXT,
            ArtifactProfile(ArtifactFormat.OPENQASM_2, "2.0"),
        ),
        (
            ConformanceTargetFamily.SYNTHETIC_NON_QASM_ARTIFACT,
            TargetClass.NON_QASM_ARTIFACT,
            ArtifactProfile(ArtifactFormat.QCIS_1, "1.0"),
        ),
    )
    extension = ProviderExtension(
        "org.flagquantum.benchmark",
        (ProviderExtensionEntry("fixture_note", "anonymous"),),
    )
    cases = []
    for family, target_class, profile in specifications:
        target = TargetCapabilities(
            target_class=target_class,
            logical_qubit_capacity=1,
            physical_qubit_capacity=1,
            native_gates=(GateCapability("rx"),),
            measurement_results=(MeasurementResult.STATE,),
            artifact_profiles=(profile,),
            maximum_program_operations=1,
        )
        target_ir = TargetIR(
            "a" * 64,
            target.semantic_fingerprint,
            (0,),
            (TargetOperation("rx", (0,), {"theta": 0.25}),),
        )
        artifact = seal_executable_artifact(
            target_ir,
            target,
            profile,
            _payload(profile, target_ir),
            compilation_identity=COMPILATION_IDENTITY,
        ).artifact
        assert artifact is not None
        cases.append(ConformanceCase(family, target, target_ir, artifact, (extension,)))
    return tuple(cases)


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


def measure_case(suite_count: int, *, iterations: int, warmup: int) -> dict[str, Any]:
    cases = build_cases()

    def execute_suites() -> str:
        drivers = (
            LocalConformanceDriver(ADAPTER_IDENTITY, RESULT_PAYLOAD),
            OfflineConformanceDriver(ADAPTER_IDENTITY),
            OfflineConformanceDriver(ADAPTER_IDENTITY),
        )
        final_identities = []
        for _ in range(suite_count):
            for case, driver in zip(cases, drivers, strict=True):
                result = run_conformance_case(
                    case,
                    driver,
                    result_payload=RESULT_PAYLOAD,
                )
                assert result.ok and result.report is not None
                final_identities.append(result.report.conformance_identity)
        return hashlib.sha256(":".join(final_identities[-3:]).encode()).hexdigest()

    for _ in range(warmup):
        execute_suites()
    gc.collect()
    timings, identities = _time(execute_suites, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        execute_suites()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "suite_count": suite_count,
        "target_lifecycle_count": suite_count * 3,
        "three_family_conformance_p95_ms": round(percentile(timings, 0.95), 6),
        "three_family_conformance_peak_host_memory_bytes": int(peak_bytes),
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
        "claim_scope": "Phase 3 Batch E private CPU baseline; not a budget or SLA",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "clock": "time.perf_counter_ns",
            "peak_memory": "tracemalloc",
            "pipeline": "identical conformance lifecycle across three target families",
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
