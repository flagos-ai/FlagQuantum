#!/usr/bin/env python3
"""Measure terminal-measurement normalization and counts projection overhead."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.terminal_measurement import (
    TerminalMeasurementNormalization,
    normalize_terminal_measurements,
)

from .contract import runtime_metadata, write_json_atomic

SCHEMA = "flagquantum.dynamic_terminal_measurement.local_performance.v1"
DEFAULT_WIDTHS = (2, 32, 128)
DEFAULT_BUDGETS: Mapping[str, float] = {
    "normalization_fixed_p95_seconds": 0.005,
    "normalization_per_qubit_p95_seconds": 0.0002,
    "projection_fixed_p95_seconds": 0.005,
    "projection_per_bit_p95_seconds": 0.000005,
}
_T = TypeVar("_T")


def build_workload(width: int) -> DynamicCircuit:
    """Build a reverse-mapped terminal-measurement workload."""

    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive integer")
    circuit = DynamicCircuit(width)
    for qubit in range(width):
        circuit.x(qubit)
    for qubit in range(width):
        circuit.measure(qubit, classical_bit=width - qubit - 1)
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


def _counts(width: int, entries: int) -> dict[str, int]:
    limit = min(entries, 1 << min(width, 20))
    return {format(index, f"0{width}b"): index + 1 for index in range(limit)}


def _case(
    *, width: int, count_entries: int, warmup: int, iterations: int
) -> dict[str, Any]:
    circuit = build_workload(width)
    source = circuit.to_ir()
    normalization_timing, normalization = _measure(
        lambda: normalize_terminal_measurements(circuit, outputs=fq.counts()),
        warmup=warmup,
        iterations=iterations,
    )
    if not isinstance(normalization, TerminalMeasurementNormalization):
        raise RuntimeError("terminal workload was not normalized")
    if normalization.source_ir is not source:
        raise RuntimeError("normalization did not retain the source IR")
    if len(normalization.unitary_ir.instructions) != width:
        raise RuntimeError("normalization did not remove the measurement suffix")

    counts = _counts(width, count_entries)
    projection_timing, projected = _measure(
        lambda: normalization.project_counts(counts),
        warmup=warmup,
        iterations=iterations,
    )
    if sum(projected.values()) != sum(counts.values()):
        raise RuntimeError("counts projection did not preserve the shot total")
    for outcome in projected:
        if len(outcome) != width:
            raise RuntimeError("counts projection changed the register width")

    return {
        "width": width,
        "instruction_count": len(source.instructions),
        "count_entries": len(counts),
        "normalization_id": normalization.normalization_id,
        "mapping_kind": "reverse_permutation",
        "correctness": {
            "source_ir_retained": True,
            "measurement_suffix_removed": True,
            "shot_total_preserved": True,
            "classical_width_preserved": True,
        },
        "normalization_timing": normalization_timing,
        "projection_timing": projection_timing,
    }


def run_benchmark(
    *,
    widths: Sequence[int] = DEFAULT_WIDTHS,
    count_entries: int = 64,
    warmup: int = 5,
    iterations: int = 101,
    budgets: Mapping[str, float] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Return local evidence without provider, credential, or network activity."""

    selected_widths = tuple(widths)
    if not selected_widths:
        raise ValueError("widths must not be empty")
    if any(type(width) is not int or width < 1 for width in selected_widths):
        raise ValueError("widths must contain positive integers")
    if type(count_entries) is not int or count_entries < 1:
        raise ValueError("count_entries must be a positive integer")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    cases = tuple(
        _case(
            width=width,
            count_entries=count_entries,
            warmup=warmup,
            iterations=iterations,
        )
        for width in selected_widths
    )
    for case in cases:
        width = int(case["width"])
        entries = int(case["count_entries"])
        case["normalization_p95_budget_seconds"] = (
            budgets["normalization_fixed_p95_seconds"]
            + width * budgets["normalization_per_qubit_p95_seconds"]
        )
        case["projection_p95_budget_seconds"] = (
            budgets["projection_fixed_p95_seconds"]
            + width * entries * budgets["projection_per_bit_p95_seconds"]
        )
        case["normalization_budget_passed"] = (
            _number(case["normalization_timing"], "p95_seconds")
            <= case["normalization_p95_budget_seconds"]
        )
        case["projection_budget_passed"] = (
            _number(case["projection_timing"], "p95_seconds")
            <= case["projection_p95_budget_seconds"]
        )

    checks = {
        "normalization_p95_seconds": all(
            bool(case["normalization_budget_passed"]) for case in cases
        ),
        "projection_p95_seconds": all(
            bool(case["projection_budget_passed"]) for case in cases
        ),
    }
    return {
        **runtime_metadata(
            runner="dynamic_terminal_measurement",
            schema=SCHEMA,
            machine=platform.machine(),
            torch=torch.__version__,
            torch_num_threads=torch.get_num_threads(),
            flagquantum=fq.__version__,
        ),
        "artifact_class": "measured_local_run",
        "benchmark": "dynamic_terminal_measurement",
        "benchmark_evidence_class": "comparison_non_release",
        "claim_evidence_type": "local_performance",
        "distribution_semantics": "single_process",
        "non_release_evidence": True,
        "scalability_claim_allowed": False,
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
            "count_entries": count_entries,
        },
        "cases": cases,
        "budgets": dict(budgets),
        "budget_checks": checks,
        "budget_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--count-entries", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=101)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        widths=args.widths,
        count_entries=args.count_entries,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    print(f"terminal-measurement budget passed: {payload['budget_passed']}")
    return 0 if payload["budget_passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
