"""Measure existing H-T-H CPU execution; no mixed-engine or performance claim."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import TypedDict

import torch

import flagquantum as fq

from .contract import runtime_metadata, write_json_atomic


class _Observation(TypedDict):
    warmup: bool
    construction_seconds: float
    planning_seconds: float
    explicit_execution_seconds: float
    automatic_execution_seconds: float
    construction_plan_execution_seconds: float
    explicit_probabilities: list[float]
    automatic_probabilities: list[float]
    explicit_max_absolute_error: float
    automatic_max_absolute_error: float
    explicit_normalization_error: float
    automatic_normalization_error: float


def _probabilities(result: fq.ExecutionResult) -> list[float]:
    values = result.probabilities
    if values is None or values.shape != (1, 2):
        raise ValueError("H-T-H execution must return a (1, 2) probability tensor")
    if values.dtype != torch.float64 or not bool(torch.isfinite(values).all()):
        raise ValueError("H-T-H probabilities must be finite float64 values")
    return [float(value) for value in values[0]]


def _run_once() -> _Observation:
    start = time.perf_counter()
    circuit = fq.Circuit(1).h(0).t(0).h(0)
    options = fq.ExecutionOptions(
        mode="statevector", device="cpu", precision="complex128"
    )
    constructed = time.perf_counter()
    plan = fq.plan(circuit, outputs=fq.probabilities([0]), options=options)
    planned = time.perf_counter()
    explicit = fq.run(plan)
    executed = time.perf_counter()
    automatic = fq.run(circuit, outputs=fq.probabilities([0]), options=options)
    automatic_end = time.perf_counter()
    explicit_values = _probabilities(explicit)
    automatic_values = _probabilities(automatic)
    expected = [(2 + math.sqrt(2)) / 4, (2 - math.sqrt(2)) / 4]
    return {
        "warmup": False,
        "construction_seconds": constructed - start,
        "planning_seconds": planned - constructed,
        "explicit_execution_seconds": executed - planned,
        "automatic_execution_seconds": automatic_end - executed,
        "construction_plan_execution_seconds": executed - start,
        "explicit_probabilities": explicit_values,
        "automatic_probabilities": automatic_values,
        "explicit_max_absolute_error": max(
            abs(value - target)
            for value, target in zip(explicit_values, expected, strict=True)
        ),
        "automatic_max_absolute_error": max(
            abs(value - target)
            for value, target in zip(automatic_values, expected, strict=True)
        ),
        "explicit_normalization_error": abs(sum(explicit_values) - 1),
        "automatic_normalization_error": abs(sum(automatic_values) - 1),
    }


def run_benchmark(*, warmup: int = 3, iterations: int = 10) -> dict[str, object]:
    """Observe the fixed one-qubit reference, retaining errors without new gates.

    Examples:
        Run ``flagquantum-benchmark run ftqc_exact_reference --iterations 10``.
    """
    for name, value, lower, upper in (
        ("warmup", warmup, 0, 3),
        ("iterations", iterations, 1, 10),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("this reference requires one local process")
    observations = []
    for index in range(warmup + iterations):
        row = _run_once()
        row["warmup"] = index < warmup
        observations.append(row)
    return {
        **runtime_metadata(
            runner="ftqc_exact_reference",
            schema="flagquantum.ftqc_exact_reference.local_observations.v1",
        ),
        "flagquantum_source_version": fq.__version__,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "precision": "complex128 statevector / float64 probabilities",
        "distribution_semantics": "single_process",
        "local_world_size": 1,
        "node_count": 1,
        "rank_ownership": {"0": "whole_workload"},
        "communication_bytes": 0,
        "non_release_evidence": True,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "blockers": ["one unencoded qubit; no mixed-engine or capacity evidence"],
        "workload": "W1-A: |0> -> H -> T -> H; no noise; all probabilities",
        "timing_boundaries": {
            "initial_imports_and_report_io_included": False,
            "lazy_imports_in_called_apis_included": True,
            "explicit_pipeline_includes_automatic_execution": False,
            "probability_validation_and_error_reduction_included": False,
            "automatic_execution_includes_planning": True,
            "separate_paths_share_process_and_runtime_caches": True,
        },
        "warmup": warmup,
        "iterations": iterations,
        "observations": observations,
        "explicit_pipeline_median_seconds": statistics.median(
            row["construction_plan_execution_seconds"] for row in observations[warmup:]
        ),
        "performance_targets_enabled": False,
        "new_numerical_tolerance_enabled": False,
    }


def main() -> int:
    """Run the fixed CPU reference and optionally write its atomic JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(warmup=args.warmup, iterations=args.iterations)
    if args.json_output is not None:
        write_json_atomic(args.json_output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
