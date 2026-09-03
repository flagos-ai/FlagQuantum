"""Measure the private offline Deployment Bridge Stage 3 readiness evaluator."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any

from benchmarks.internal.deployment_bridge_stage2_baseline import _build_inputs
from flagquantum._compiler.deployment_canary_readiness import (
    CanaryBudgetSnapshot,
    CanaryControlSnapshot,
    CanaryReadinessPolicy,
    CanaryReadinessStatus,
    ConformanceAttestationState,
    ProviderConformanceAttestation,
    evaluate_deployment_canary_readiness,
)
from flagquantum._compiler.deployment_dry_run import dry_run_deployment_bridge


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _build_readiness_inputs():
    package, target, dry_run_policy = _build_inputs(10)
    dry_run_report = dry_run_deployment_bridge(package, target, dry_run_policy)
    assert dry_run_report.artifact_profile is not None
    conformance = ProviderConformanceAttestation(
        True,
        ConformanceAttestationState.VALID,
        dry_run_report.target_capability_fingerprint,
        dry_run_report.artifact_profile,
        "anonymous.synthetic",
        "v1",
        "c" * 64,
    )
    controls = CanaryControlSnapshot(
        **{item.name: True for item in fields(CanaryControlSnapshot)}
    )
    budgets = CanaryBudgetSnapshot(
        **{item.name: True for item in fields(CanaryBudgetSnapshot)}
    )
    return (
        dry_run_report,
        conformance,
        controls,
        budgets,
        CanaryReadinessPolicy(65536),
    )


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
    evaluation_count: int, *, iterations: int, warmup: int
) -> dict[str, Any]:
    dry_run_report, conformance, controls, budgets, policy = _build_readiness_inputs()

    def evaluate_batch() -> str:
        evidence_identity = ""
        for _ in range(evaluation_count):
            report = evaluate_deployment_canary_readiness(
                dry_run_report,
                conformance,
                controls,
                budgets,
                policy,
            )
            assert (
                report.status
                is CanaryReadinessStatus.READY_FOR_SEPARATE_ACTIVATION_REVIEW
            )
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
        "evaluation_count": evaluation_count,
        "readiness_evaluation_p95_ms": round(percentile(timings, 0.95), 6),
        "readiness_evaluation_peak_host_memory_bytes": int(peak_bytes),
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
        "claim_scope": "Deployment Bridge Stage 3 private CPU baseline; not a budget or SLA",
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
            "pipeline": "anonymous offline canary-readiness evidence evaluation",
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
