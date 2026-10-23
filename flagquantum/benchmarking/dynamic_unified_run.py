#!/usr/bin/env python3
"""Measure the stable local DynamicCircuit plan/run boundary."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

from .contract import runtime_metadata, write_json_atomic

SCHEMA = "flagquantum.dynamic_unified_run.local_performance.v1"
DEFAULT_BUDGETS: Mapping[str, float] = {
    "planning_p95_seconds": 0.05,
    "explicit_over_native_median": 3.0,
    "automatic_over_explicit_median": 2.0,
    "static_automatic_over_explicit_median": 2.0,
}

_T = TypeVar("_T")


def build_dynamic_workload() -> DynamicCircuit:
    """Build the bounded feedback workload used by every dynamic path."""

    circuit = DynamicCircuit(2)
    circuit.gate("h", 0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    circuit.reset(0)
    circuit.measure(0, classical_bit=1)
    return circuit


def build_static_sentinel() -> fq.Circuit:
    """Build a static sampling workload that must retain the established path."""

    circuit = fq.Circuit(2)
    circuit.gate("h", 0)
    circuit.gate("cx", (0, 1))
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


def _median(timing: Mapping[str, object]) -> float:
    return _number(timing, "median_seconds")


def _dynamic_case(
    *,
    name: str,
    shots: int,
    seed: int,
    memory_limit_bytes: int,
    expected_strategy: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    circuit = build_dynamic_workload()
    options = fq.ExecutionOptions(
        shots=shots,
        seed=seed,
        memory_limit_bytes=memory_limit_bytes,
    )
    plan = fq.plan(circuit, options=options)
    explicit_timing, explicit = _measure(
        lambda: fq.run(plan), warmup=warmup, iterations=iterations
    )
    automatic_timing, automatic = _measure(
        lambda: fq.run(circuit, options=options),
        warmup=warmup,
        iterations=iterations,
    )
    native_timing, native = _measure(
        lambda: fq.experimental.dynamic.run_dynamic(
            circuit,
            shots=shots,
            seed=seed,
            max_batched_bytes=memory_limit_bytes,
        ),
        warmup=warmup,
        iterations=iterations,
    )

    if explicit.samples is None or automatic.samples is None:
        raise RuntimeError("stable dynamic execution did not return samples")
    if not torch.equal(explicit.samples, automatic.samples):
        raise RuntimeError("automatic and explicit stable samples disagree")
    if not torch.equal(explicit.samples, native.samples):
        raise RuntimeError("stable samples disagree with the native oracle")
    if explicit.plan is None or automatic.plan is None:
        raise RuntimeError("stable dynamic result did not retain its plan")
    if explicit.plan.identity != automatic.plan.identity:
        raise RuntimeError("automatic and explicit plans have different identities")

    statistics_payload = explicit.runtime.get("dynamic_statistics")
    if not isinstance(statistics_payload, Mapping):
        raise RuntimeError("dynamic runtime statistics are unavailable")
    observed_strategy = statistics_payload.get("gate_execution_strategy")
    if observed_strategy != expected_strategy:
        raise RuntimeError(
            f"{name} selected {observed_strategy!r}, expected {expected_strategy!r}"
        )

    native_median = _median(native_timing)
    explicit_median = _median(explicit_timing)
    return {
        "name": name,
        "shots": shots,
        "memory_limit_bytes": memory_limit_bytes,
        "execution_strategy": observed_strategy,
        "plan_identity": plan.identity,
        "correctness": {
            "automatic_matches_explicit": True,
            "stable_matches_native_oracle": True,
        },
        "timing": {
            "explicit_plan_run": explicit_timing,
            "automatic_plan_and_run": automatic_timing,
            "native_oracle": native_timing,
        },
        "ratios": {
            "explicit_over_native_median": explicit_median / native_median,
            "automatic_over_explicit_median": (
                _median(automatic_timing) / explicit_median
            ),
        },
    }


def _static_case(
    *, shots: int, seed: int, warmup: int, iterations: int
) -> dict[str, Any]:
    circuit = build_static_sentinel()
    options = fq.ExecutionOptions(target="samples", shots=shots, seed=seed)
    plan = fq.plan(circuit, options=options)
    explicit_timing, explicit = _measure(
        lambda: fq.run(plan), warmup=warmup, iterations=iterations
    )
    automatic_timing, automatic = _measure(
        lambda: fq.run(circuit, options=options),
        warmup=warmup,
        iterations=iterations,
    )
    if explicit.samples is None or automatic.samples is None:
        raise RuntimeError("static sampling sentinel did not return samples")
    if not torch.equal(explicit.samples, automatic.samples):
        raise RuntimeError("static automatic and explicit samples disagree")
    if explicit.plan is None or automatic.plan is None:
        raise RuntimeError("static sampling result did not retain its plan")
    if explicit.plan.identity != automatic.plan.identity:
        raise RuntimeError("static automatic and explicit plans disagree")
    return {
        "shots": shots,
        "correctness": {"automatic_matches_explicit": True},
        "timing": {
            "explicit_plan_run": explicit_timing,
            "automatic_plan_and_run": automatic_timing,
        },
        "ratios": {
            "automatic_over_explicit_median": (
                _median(automatic_timing) / _median(explicit_timing)
            )
        },
    }


def run_benchmark(
    *,
    batched_shots: int = 4096,
    trajectory_shots: int = 64,
    static_shots: int = 4096,
    seed: int = 7,
    warmup: int = 2,
    iterations: int = 9,
    planning_iterations: int = 25,
    budgets: Mapping[str, float] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Measure stable wrapper overhead and the unchanged static sentinel."""

    for name, value in {
        "batched_shots": batched_shots,
        "trajectory_shots": trajectory_shots,
        "static_shots": static_shots,
        "iterations": iterations,
        "planning_iterations": planning_iterations,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")

    dynamic = build_dynamic_workload()
    planning_options = fq.ExecutionOptions(shots=batched_shots, seed=seed)
    planning, planned = _measure(
        lambda: fq.plan(dynamic, options=planning_options),
        warmup=warmup,
        iterations=planning_iterations,
    )
    if not isinstance(planned, fq.ExecutionPlan):
        raise RuntimeError("dynamic planning did not return ExecutionPlan")

    cases = (
        _dynamic_case(
            name="batched",
            shots=batched_shots,
            seed=seed,
            memory_limit_bytes=256 * 1024**2,
            expected_strategy="batched_statevector_kernel",
            warmup=warmup,
            iterations=iterations,
        ),
        _dynamic_case(
            name="trajectory_memory_fallback",
            shots=trajectory_shots,
            seed=seed,
            memory_limit_bytes=1,
            expected_strategy="trajectory_direct_statevector_kernel",
            warmup=warmup,
            iterations=iterations,
        ),
    )
    static = _static_case(
        shots=static_shots,
        seed=seed,
        warmup=warmup,
        iterations=iterations,
    )

    checks = {
        "planning_p95_seconds": _number(planning, "p95_seconds")
        <= budgets["planning_p95_seconds"],
        "explicit_over_native_median": all(
            float(case["ratios"]["explicit_over_native_median"])
            <= budgets["explicit_over_native_median"]
            for case in cases
        ),
        "automatic_over_explicit_median": all(
            float(case["ratios"]["automatic_over_explicit_median"])
            <= budgets["automatic_over_explicit_median"]
            for case in cases
        ),
        "static_automatic_over_explicit_median": (
            float(static["ratios"]["automatic_over_explicit_median"])
            <= budgets["static_automatic_over_explicit_median"]
        ),
    }
    return {
        **runtime_metadata(
            runner="dynamic_unified_run",
            schema=SCHEMA,
            machine=platform.machine(),
            torch=torch.__version__,
            torch_num_threads=torch.get_num_threads(),
            flagquantum=fq.__version__,
        ),
        "artifact_class": "measured_local_run",
        "benchmark": "dynamic_unified_run",
        "benchmark_evidence_class": "comparison_non_release",
        "claim_evidence_type": "local_performance",
        "distribution_semantics": "single_process",
        "non_release_evidence": True,
        "scalability_claim_allowed": False,
        "scalability_blockers": [
            "single-process local execution",
            "no cross-rank state partitioning",
            "no accelerator or QPU measurement",
        ],
        "release_gate_allowed": False,
        "seed": seed,
        "methodology": {
            "warmup": warmup,
            "iterations": iterations,
            "planning_iterations": planning_iterations,
        },
        "planning": planning,
        "dynamic_cases": cases,
        "static_sentinel": static,
        "budgets": dict(budgets),
        "budget_checks": checks,
        "budget_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batched-shots", type=int, default=4096)
    parser.add_argument("--trajectory-shots", type=int, default=64)
    parser.add_argument("--static-shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--planning-iterations", type=int, default=25)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        batched_shots=args.batched_shots,
        trajectory_shots=args.trajectory_shots,
        static_shots=args.static_shots,
        seed=args.seed,
        warmup=args.warmup,
        iterations=args.iterations,
        planning_iterations=args.planning_iterations,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    print(f"dynamic unified-run performance budget passed: {payload['budget_passed']}")
    return 0 if payload["budget_passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
