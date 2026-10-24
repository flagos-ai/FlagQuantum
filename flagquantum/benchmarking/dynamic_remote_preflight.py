#!/usr/bin/env python3
"""Measure remote DynamicCircuit preflight and static classification overhead."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.remote_preflight import (
    evaluate_dynamic_remote_preflight,
)

from .contract import runtime_metadata, write_json_atomic

SCHEMA = "flagquantum.dynamic_remote_preflight.local_performance.v1"
DEFAULT_BUDGETS: Mapping[str, float] = {
    "preflight_fixed_p95_seconds": 0.01,
    "preflight_per_instruction_p95_seconds": 0.00055,
    "static_classifier_p95_seconds": 0.001,
}
DEFAULT_FEEDBACK_PAIRS = (1, 16, 128)
_EVALUATED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
_T = TypeVar("_T")


def build_workload(feedback_pairs: int) -> DynamicCircuit:
    """Build a bounded feedback program outside the measured region."""

    if type(feedback_pairs) is not int or feedback_pairs < 1:
        raise ValueError("feedback_pairs must be a positive integer")
    circuit = DynamicCircuit(2)
    for classical_bit in range(feedback_pairs):
        circuit.measure(0, classical_bit=classical_bit)
        circuit.conditional("x", 1, classical_bit=classical_bit)
    return circuit


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(samples: tuple[float, ...]) -> dict[str, object]:
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_seconds": median,
        "p95_seconds": _percentile(samples, 0.95),
        "relative_median_absolute_deviation": mad / median if median else 0.0,
    }


def _measure(
    function: Callable[[], _T], *, warmup: int, iterations: int
) -> tuple[dict[str, object], _T]:
    result = function()
    for _ in range(warmup):
        result = function()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - started)
    return _summary(tuple(samples)), result


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _preflight_case(
    *, feedback_pairs: int, warmup: int, iterations: int
) -> dict[str, Any]:
    circuit = build_workload(feedback_pairs)
    timing, decision = _measure(
        lambda: evaluate_dynamic_remote_preflight(
            circuit,
            target="quafu:Baihua",
            evaluated_at=_EVALUATED_AT,
        ),
        warmup=warmup,
        iterations=iterations,
    )
    if decision.compatible or not decision.blocker_codes:
        raise RuntimeError("missing provider facts did not fail closed")
    return {
        "feedback_pairs": feedback_pairs,
        "instruction_count": len(circuit.to_ir().instructions),
        "target": decision.target,
        "requirement_set_id": decision.requirement_set_id,
        "snapshot_id": decision.snapshot_id,
        "blocker_codes": decision.blocker_codes,
        "compatible": decision.compatible,
        "timing": timing,
    }


def _static_classifier_case(*, warmup: int, iterations: int) -> dict[str, Any]:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    def classify() -> bool:
        dynamic_type = import_module(
            "flagquantum.runtime.dynamic.circuit"
        ).DynamicCircuit
        return isinstance(circuit, dynamic_type)

    timing, classified_dynamic = _measure(
        classify,
        warmup=warmup,
        iterations=iterations,
    )
    if classified_dynamic:
        raise RuntimeError("static circuit was classified as dynamic")
    return {
        "classified_dynamic": classified_dynamic,
        "timing": timing,
    }


def run_benchmark(
    *,
    feedback_pairs: Sequence[int] = DEFAULT_FEEDBACK_PAIRS,
    warmup: int = 5,
    iterations: int = 101,
    static_iterations: int = 1001,
    budgets: Mapping[str, float] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Return non-release local evidence without provider or network activity."""

    pairs = tuple(feedback_pairs)
    if not pairs:
        raise ValueError("feedback_pairs must not be empty")
    for value in pairs:
        if type(value) is not int or value < 1:
            raise ValueError("feedback_pairs must contain positive integers")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    for name, value in {
        "iterations": iterations,
        "static_iterations": static_iterations,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    cases = tuple(
        _preflight_case(
            feedback_pairs=value,
            warmup=warmup,
            iterations=iterations,
        )
        for value in pairs
    )
    for case in cases:
        case["p95_budget_seconds"] = (
            budgets["preflight_fixed_p95_seconds"]
            + int(case["instruction_count"])
            * budgets["preflight_per_instruction_p95_seconds"]
        )
        case["budget_passed"] = (
            _number(case["timing"], "p95_seconds") <= case["p95_budget_seconds"]
        )
    static = _static_classifier_case(
        warmup=warmup,
        iterations=static_iterations,
    )
    checks = {
        "preflight_p95_seconds": all(bool(case["budget_passed"]) for case in cases),
        "static_classifier_p95_seconds": (
            _number(static["timing"], "p95_seconds")
            <= budgets["static_classifier_p95_seconds"]
        ),
    }
    return {
        **runtime_metadata(
            runner="dynamic_remote_preflight",
            schema=SCHEMA,
            machine=platform.machine(),
            torch=torch.__version__,
            torch_num_threads=torch.get_num_threads(),
            flagquantum=fq.__version__,
        ),
        "artifact_class": "measured_local_run",
        "benchmark": "dynamic_remote_preflight",
        "benchmark_evidence_class": "comparison_non_release",
        "claim_evidence_type": "local_performance",
        "distribution_semantics": "single_process",
        "non_release_evidence": True,
        "scalability_claim_allowed": False,
        "scalability_blockers": [
            "single-process local timing only",
            "no provider, network, accelerator, or QPU measurement",
        ],
        "release_gate_allowed": False,
        "remote_activity": {
            "credential_lookup": False,
            "provider_import": False,
            "network_access": False,
            "task_submission": False,
        },
        "methodology": {
            "warmup": warmup,
            "iterations": iterations,
            "static_iterations": static_iterations,
            "evaluated_at": _EVALUATED_AT.isoformat().replace("+00:00", "Z"),
        },
        "preflight_cases": cases,
        "static_classifier_bypass": static,
        "budgets": dict(budgets),
        "budget_checks": checks,
        "budget_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feedback-pairs",
        type=int,
        nargs="+",
        default=DEFAULT_FEEDBACK_PAIRS,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=101)
    parser.add_argument("--static-iterations", type=int, default=1001)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        feedback_pairs=args.feedback_pairs,
        warmup=args.warmup,
        iterations=args.iterations,
        static_iterations=args.static_iterations,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    print(f"dynamic remote preflight budget passed: {payload['budget_passed']}")
    return 0 if payload["budget_passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
