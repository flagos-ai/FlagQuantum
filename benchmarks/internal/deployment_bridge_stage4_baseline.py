"""Measure the private offline Deployment Bridge Stage 4 rehearsal engine."""

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

from benchmarks.internal.deployment_bridge_stage3_baseline import (
    _build_readiness_inputs,
)
from flagquantum._compiler.deployment_canary_readiness import (
    evaluate_deployment_canary_readiness,
)
from flagquantum._compiler.deployment_rehearsal import (
    RehearsalObservationSnapshot,
    RehearsalOutcome,
    RehearsalPolicy,
    RehearsalScenario,
    normal_rehearsal_states,
    rehearsal_requirements,
    run_offline_deployment_rehearsal,
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


def _build_rehearsal_inputs():
    readiness_inputs = _build_readiness_inputs()
    readiness_report = evaluate_deployment_canary_readiness(*readiness_inputs)
    scenario = RehearsalScenario.UNKNOWN_SUBMISSION_OUTCOME
    roles, actions = rehearsal_requirements(scenario)
    observations = RehearsalObservationSnapshot(
        normal_rehearsal_states(),
        tuple(actions),
        tuple(roles),
        10,
        20,
        30,
        40,
    )
    policy = RehearsalPolicy(1, 8, 11, 65536, 65536, 100, 100, 100, 100)
    return readiness_report, scenario, observations, policy


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
    rehearsal_count: int, *, iterations: int, warmup: int
) -> dict[str, Any]:
    readiness_report, scenario, observations, policy = _build_rehearsal_inputs()

    def rehearse_batch() -> str:
        evidence_identity = ""
        for _ in range(rehearsal_count):
            report = run_offline_deployment_rehearsal(
                readiness_report,
                scenario,
                observations,
                policy,
            )
            assert report.outcome is RehearsalOutcome.OFFLINE_REHEARSAL_PASSED
            evidence_identity = report.evidence_identity
        return evidence_identity

    for _ in range(warmup):
        rehearse_batch()
    gc.collect()
    timings, identities = _time(rehearse_batch, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        rehearse_batch()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "rehearsal_count": rehearsal_count,
        "offline_rehearsal_p95_ms": round(percentile(timings, 0.95), 6),
        "offline_rehearsal_peak_host_memory_bytes": int(peak_bytes),
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
        "claim_scope": "Deployment Bridge Stage 4 private CPU baseline; not a budget or SLA",
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
            "pipeline": "anonymous offline unknown-submission rehearsal",
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
