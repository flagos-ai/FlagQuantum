"""Contracts for reproducible SV-014 public-dispatch evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.statevector_swap_sequence_dispatch import (
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
    validate_run,
)

pytestmark = pytest.mark.unit


def _timing(seconds: float) -> dict[str, Any]:
    return {
        "samples_seconds_per_invocation": [seconds] * 30,
        "median_seconds_per_invocation": seconds,
    }


def _run(host: str, compiler_lane: str, *, revision: str = "a" * 40) -> dict[str, Any]:
    distribution, integration_path = {
        "stock_triton": ("triton", "direct"),
        "flagtree": ("flagtree", "flagtree"),
    }[compiler_lane]
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "source_revision": revision,
        "host_label": host,
        "compiler_lane": compiler_lane,
        "compiler": {
            "distribution": distribution,
            "version": "test",
            "integration_path": integration_path,
            "identity_status": "resolved",
        },
        "execution_semantics": "single_device_public_runtime_dispatch",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
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
                "public_catalog_dispatch": _timing(1.0),
                "public_pytorch_reference": _timing(1.5),
                "public_speedup_over_pytorch": 1.5,
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


def test_sv014_dispatch_run_validator_accepts_complete_measurements() -> None:
    validate_run(_run(HOSTS[0], COMPILER_LANES[0]))


def test_sv014_dispatch_run_validator_recomputes_median_and_speedup() -> None:
    payload = _run(HOSTS[0], COMPILER_LANES[0])
    payload["cases"][0]["public_catalog_dispatch"][
        "median_seconds_per_invocation"
    ] = 9.0
    with pytest.raises(ValueError, match="median is invalid"):
        validate_run(payload)

    payload = _run(HOSTS[0], COMPILER_LANES[0])
    payload["cases"][0]["public_speedup_over_pytorch"] = 9.0
    with pytest.raises(ValueError, match="speedup is invalid"):
        validate_run(payload)


def test_sv014_dispatch_aggregate_requires_profitable_complete_matrix(
    tmp_path: Path,
) -> None:
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
    assert aggregate["minimum_public_speedup_over_pytorch"] == 1.5
    assert aggregate["all_cases_meet_performance_floor"] is True
    assert aggregate["decision"] == "default_dispatch_enabled"


def test_sv014_dispatch_aggregate_rejects_incomplete_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly four raw runs"):
        aggregate_runs(_write_matrix(tmp_path)[:-1])


def test_sv014_dispatch_aggregate_rejects_performance_regression(
    tmp_path: Path,
) -> None:
    paths = _write_matrix(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    case = payload["cases"][0]
    case["public_pytorch_reference"] = _timing(0.99)
    case["public_speedup_over_pytorch"] = 0.99
    paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="performance floor"):
        aggregate_runs(paths)


def test_sv014_dispatch_aggregate_requires_exact_output(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["cases"][0]["maximum_absolute_error"] = 1.0e-7
    paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bitwise exact"):
        aggregate_runs(paths)
