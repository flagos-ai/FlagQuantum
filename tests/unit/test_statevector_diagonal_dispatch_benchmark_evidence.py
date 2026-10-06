"""Contracts for the checked-in SV-010 public-dispatch evidence."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from benchmarks.statevector_local_diagonal_dispatch import (
    BENCHMARK,
    COMPILER_LANES,
    EVIDENCE_SCHEMA,
    HOSTS,
    IMPLEMENTATION_ID,
    PERFORMANCE_FLOOR,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    _shape_record,
    validate_run,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = (
    _ROOT
    / "benchmarks"
    / "results"
    / "local"
    / "statevector_local_diagonal_dispatch_a800.json"
)
_EVIDENCE_REVISION = "53467224223c1ea3766657feee4757fefc694c58"


def test_statevector_diagonal_dispatch_artifact_is_complete() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["benchmark"] == BENCHMARK
    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["semantic_id"] == SEMANTIC_ID
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["runner"] == RUNNER
    assert payload["source_revision"] == _EVIDENCE_REVISION
    assert payload["shape_matrix"] == [_shape_record(*shape) for shape in SHAPE_MATRIX]
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False

    runs = payload["runs"]
    assert {(run["host_label"], run["compiler_lane"]) for run in runs} == {
        (host, compiler_lane) for host in HOSTS for compiler_lane in COMPILER_LANES
    }
    for run in runs:
        validate_run(run)
        assert run["source_revision"] == _EVIDENCE_REVISION
        assert run["measurement"]["repeats"] == 30
        assert run["measurement"]["group_size"] == 10
        for case in run["cases"]:
            for result_name in (
                "public_catalog_dispatch",
                "public_pytorch_reference",
            ):
                result = case[result_name]
                assert len(result["samples_seconds_per_invocation"]) == 30
                assert result["median_seconds_per_invocation"] == statistics.median(
                    result["samples_seconds_per_invocation"]
                )
            assert case["public_speedup_over_pytorch"] >= PERFORMANCE_FLOOR

    aggregate = payload["aggregate"]
    assert aggregate["case_count"] == 20
    assert aggregate["minimum_public_speedup_over_pytorch"] >= PERFORMANCE_FLOOR
    assert aggregate["all_cases_meet_performance_floor"] is True
    assert aggregate["decision"] == "enable_default_dispatch"
