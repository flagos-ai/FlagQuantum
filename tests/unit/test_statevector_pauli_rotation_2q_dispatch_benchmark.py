"""Contracts for reproducible SV-011 public-dispatch evidence."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest

from benchmarks.statevector_pauli_rotation_2q_dispatch import (
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

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = (
    _ROOT
    / "benchmarks"
    / "results"
    / "local"
    / "statevector_pauli_rotation_2q_dispatch_a800.json"
)
_EVIDENCE_REVISION = "d1a537d59ff60829d4837c600a52842dd7c9b492"


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
                "public_catalog_dispatch": _timing(1.0),
                "public_pytorch_reference": _timing(1.5),
                "public_speedup_over_pytorch": 1.5,
                "maximum_absolute_error": 1.0e-7,
                "relative_l2_error": 1.0e-8,
            }
            for shape in SHAPE_MATRIX
        ],
    }


def _write_matrix(tmp_path: Path, *, revision: str = "a" * 40) -> list[Path]:
    paths = []
    for host in HOSTS:
        for compiler_lane in COMPILER_LANES:
            path = tmp_path / f"{host}-{compiler_lane}.json"
            path.write_text(
                json.dumps(_run(host, compiler_lane, revision=revision)),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def test_sv011_dispatch_aggregate_requires_profitable_complete_matrix(
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
    assert aggregate["case_count"] == 16
    assert aggregate["minimum_public_speedup_over_pytorch"] == 1.5
    assert aggregate["all_cases_meet_performance_floor"] is True
    assert aggregate["decision"] == "default_dispatch_enabled"


def test_sv011_dispatch_aggregate_rejects_incomplete_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly four raw runs"):
        aggregate_runs(_write_matrix(tmp_path)[:-1])


def test_sv011_dispatch_aggregate_rejects_performance_regression(
    tmp_path: Path,
) -> None:
    paths = _write_matrix(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["cases"][0]["public_speedup_over_pytorch"] = 0.99
    paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="performance floor"):
        aggregate_runs(paths)


def test_checked_in_sv011_dispatch_evidence_is_profitable() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["benchmark"] == BENCHMARK
    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["semantic_id"] == SEMANTIC_ID
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["runner"] == RUNNER
    assert payload["source_revision"] == _EVIDENCE_REVISION
    assert payload["shape_matrix"] == [_shape_record(*shape) for shape in SHAPE_MATRIX]
    assert {(run["host_label"], run["compiler_lane"]) for run in payload["runs"]} == {
        (host, compiler_lane) for host in HOSTS for compiler_lane in COMPILER_LANES
    }
    aggregate = payload["aggregate"]
    assert aggregate["case_count"] == 16
    assert aggregate["minimum_public_speedup_over_pytorch"] > 1.43
    assert aggregate["maximum_public_speedup_over_pytorch"] > 7.33
    assert aggregate["maximum_absolute_error"] <= 3.4e-7
    assert aggregate["maximum_relative_l2_error"] <= 4.1e-8
    assert aggregate["all_cases_meet_performance_floor"] is True
    assert aggregate["decision"] == "default_dispatch_enabled"


def test_checked_in_sv011_dispatch_samples_reproduce_claims() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    for run in payload["runs"]:
        assert run["source_revision"] == payload["source_revision"]
        assert run["compiler"]["identity_status"] == "resolved"
        assert run["environment"]["gpu"] == "NVIDIA A800-SXM4-80GB"
        assert run["measurement"] == payload["measurement"]
        repeats = int(run["measurement"]["repeats"])
        for case in run["cases"]:
            dispatch = case["public_catalog_dispatch"]
            reference = case["public_pytorch_reference"]
            assert len(dispatch["samples_seconds_per_invocation"]) == repeats
            assert len(reference["samples_seconds_per_invocation"]) == repeats
            assert math.isclose(
                dispatch["median_seconds_per_invocation"],
                statistics.median(dispatch["samples_seconds_per_invocation"]),
            )
            assert math.isclose(
                reference["median_seconds_per_invocation"],
                statistics.median(reference["samples_seconds_per_invocation"]),
            )
            assert math.isclose(
                case["public_speedup_over_pytorch"],
                reference["median_seconds_per_invocation"]
                / dispatch["median_seconds_per_invocation"],
            )
