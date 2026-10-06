"""Contracts for reproducible SV-013 development evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal.evidence.statevector_reversible_3q_probe import (
    BENCHMARK,
    COMPILER_LANES,
    EVIDENCE_SCHEMA,
    HOSTS,
    IMPLEMENTATION_ID,
    PERFORMANCE_FLOOR,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    _shape_record,
    aggregate_runs,
)

pytestmark = pytest.mark.unit


def _timing(seconds: float) -> dict[str, Any]:
    samples = [seconds] * 30
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": seconds,
    }


def _run(host: str, compiler_lane: str, *, revision: str = "a" * 40) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "source_revision": revision,
        "host_label": host,
        "compiler_lane": compiler_lane,
        "measurement": {
            "warmup": 10,
            "repeats": 30,
            "group_size": 10,
            "ordering": "counterbalanced by repeat parity",
            "synchronization": "before and after every timed group",
            "statistic": "median synchronized wall seconds per invocation",
        },
        "cases": [
            {
                "shape": _shape_record(*shape),
                "direct_kernel_wrapper": _timing(1.0),
                "product_reference": _timing(1.5),
                "speedup_over_product": 1.5,
                "maximum_absolute_error": 0.0,
                "relative_l2_error": 0.0,
            }
            for shape in SHAPE_MATRIX
        ],
    }


def _write_matrix(tmp_path: Path) -> list[Path]:
    paths = []
    for host in HOSTS:
        for compiler_lane in COMPILER_LANES:
            path = tmp_path / f"{host}-{compiler_lane}.json"
            path.write_text(
                json.dumps(_run(host, compiler_lane)),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def test_sv013_aggregate_requires_profitable_complete_matrix(tmp_path: Path) -> None:
    payload = aggregate_runs(_write_matrix(tmp_path))

    assert payload["benchmark"] == BENCHMARK
    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["semantic_id"] == SEMANTIC_ID
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["runner"] == RUNNER
    assert payload["performance_floor"] == PERFORMANCE_FLOOR
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    aggregate = payload["aggregate"]
    assert aggregate["case_count"] == 20
    assert aggregate["default_case_count"] == 16
    assert aggregate["excluded_boundary_case_count"] == 4
    assert aggregate["minimum_default_speedup_over_product"] == 1.5
    assert aggregate["all_default_cases_meet_performance_floor"] is True
    assert aggregate["decision"] == "eligible_for_bounded_dispatch_evaluation"


def test_sv013_aggregate_rejects_incomplete_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly four raw runs"):
        aggregate_runs(_write_matrix(tmp_path)[:-1])


def test_sv013_aggregate_rejects_performance_regression(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["cases"][1]["speedup_over_product"] = 0.99
    paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="performance floor"):
        aggregate_runs(paths)
