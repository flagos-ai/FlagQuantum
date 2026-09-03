"""Measure explicit match-only shadow comparison overhead and memory."""

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

from flagquantum._compiler.runtime_abi import ExecutionState
from flagquantum._compiler.shadow_contracts import (
    ShadowObservation,
    ShadowPolicy,
    ShadowResultKind,
)
from flagquantum._compiler.shadow_harness import (
    ExplicitShadowHarness,
    ShadowKillSwitch,
)

RESULT_IDENTITY = "a" * 64


class StepClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        value = self.value
        self.value += 100
        return value


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


def measure_case(
    comparison_count: int, *, iterations: int, warmup: int
) -> dict[str, Any]:
    observation = ShadowObservation(
        ExecutionState.SUCCEEDED,
        ShadowResultKind.STATE,
        (2,),
        b"anonymous-result",
        RESULT_IDENTITY,
    )
    policy = ShadowPolicy(
        enabled=True,
        max_comparisons=comparison_count,
        max_input_bytes=1024,
        max_evidence_bytes=4096,
        max_candidate_time_ns=1000,
        max_mismatches=comparison_count,
    )

    def execute_comparisons() -> str:
        harness = ExplicitShadowHarness(
            policy,
            ShadowKillSwitch(),
            clock_ns=StepClock(),
        )
        final_identity = ""
        for _ in range(comparison_count):
            outcome = harness.compare(
                b"anonymous-input",
                lambda _input: observation,
                lambda _input: observation,
            )
            assert outcome.evidence is not None
            final_identity = outcome.evidence.evidence_identity
        return final_identity

    for _ in range(warmup):
        execute_comparisons()
    gc.collect()
    timings, identities = _time(execute_comparisons, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        execute_comparisons()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "comparison_count": comparison_count,
        "match_shadow_p95_ms": round(percentile(timings, 0.95), 6),
        "match_shadow_peak_host_memory_bytes": int(peak_bytes),
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
        "claim_scope": "Phase 3 Batch F private CPU baseline; not a budget or SLA",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "clock": "time.perf_counter_ns",
            "evidence_clock": "deterministic 100 ns test step",
            "peak_memory": "tracemalloc",
            "pipeline": "explicit legacy-authoritative match-only shadow comparison",
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
