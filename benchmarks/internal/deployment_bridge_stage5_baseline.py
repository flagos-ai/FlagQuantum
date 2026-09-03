"""Measure the private offline Deployment Bridge Stage 5 observation evaluator."""

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

from benchmarks.internal.deployment_bridge_stage4_baseline import (
    _build_rehearsal_inputs,
)
from flagquantum._compiler.deployment_rehearsal import (
    run_offline_deployment_rehearsal,
)
from flagquantum._compiler.provider_sandbox_observation import (
    SandboxObservationDecision,
    SandboxObservationPolicy,
    SandboxObservationSnapshot,
    accepted_observation_states,
    evaluate_provider_sandbox_observation,
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


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _build_observation_inputs():
    readiness, scenario, rehearsal_observations, rehearsal_policy = (
        _build_rehearsal_inputs()
    )
    rehearsal = run_offline_deployment_rehearsal(
        readiness,
        scenario,
        rehearsal_observations,
        rehearsal_policy,
    )
    observation = SandboxObservationSnapshot(
        accepted_observation_states(),
        _identity("anonymous-sandbox-target"),
        _identity("anonymous-request"),
        _identity("anonymous-idempotency-key"),
        readiness.evidence_identity,
        rehearsal.evidence_identity,
        True,
        True,
        True,
        True,
        True,
        True,
        10,
        20,
        30,
        40,
    )
    policy = SandboxObservationPolicy(4, 65536, 65536, 100, 100, 100, 100)
    return readiness, rehearsal, observation, policy


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
    observation_count: int, *, iterations: int, warmup: int
) -> dict[str, Any]:
    readiness, rehearsal, observation, policy = _build_observation_inputs()

    def evaluate_batch() -> str:
        evidence_identity = ""
        for _ in range(observation_count):
            report = evaluate_provider_sandbox_observation(
                readiness,
                rehearsal,
                observation,
                policy,
            )
            assert report.decision is SandboxObservationDecision.OBSERVATION_ACCEPTED
            evidence_identity = report.evidence_identity
        return evidence_identity

    for _ in range(warmup):
        evaluate_batch()
    gc.collect()
    timings, identities = _time(evaluate_batch, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        evaluate_batch()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "observation_count": observation_count,
        "offline_observation_p95_ms": round(percentile(timings, 0.95), 6),
        "offline_observation_peak_host_memory_bytes": int(peak_bytes),
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
        "claim_scope": "Deployment Bridge Stage 5 private CPU baseline; not a budget or SLA",
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
            "pipeline": "anonymous offline accepted sandbox observation",
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
