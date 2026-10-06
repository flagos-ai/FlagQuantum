"""Contracts for reproducible MPS-008 benchmark evidence."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal.evidence.mps_sampling_collapse_probe import (
    COMPILER_LANES,
    EVIDENCE_SCHEMA,
    HOSTS,
    IMPLEMENTATION_ID,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    aggregate_runs,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = (
    _ROOT / "benchmarks" / "results" / "local" / "mps_sampling_collapse_a800.json"
)


def _artifact() -> dict[str, Any]:
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


def test_checked_in_mps008_evidence_has_canonical_scope_and_matrix() -> None:
    payload = _artifact()

    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["semantic_id"] == SEMANTIC_ID
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["runner"] == RUNNER
    assert payload["execution_semantics"] == "single_device_fast_path"
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["shape_matrix"] == [
        {
            "batch": batch,
            "right_bond": right_dim,
            "next_right_bond": next_right_dim,
        }
        for batch, right_dim, next_right_dim in SHAPE_MATRIX
    ]
    runs = payload["runs"]
    assert {(run["host_label"], run["compiler_lane"]) for run in runs} == {
        (host, compiler_lane) for host in HOSTS for compiler_lane in COMPILER_LANES
    }


def test_checked_in_mps008_raw_measurements_reproduce_claims() -> None:
    payload = _artifact()
    source_revision = payload["source_revision"]
    for run in payload["runs"]:
        assert run["schema"] == RUN_SCHEMA
        assert run["source_revision"] == source_revision
        assert run["compiler"]["identity_status"] == "resolved"
        assert run["environment"]["gpu"] == "NVIDIA A800-SXM4-80GB"
        assert run["measurement"] == payload["measurement"]
        repeats = int(run["measurement"]["repeats"])
        for case in run["cases"]:
            direct = case["direct_kernel_wrapper"]
            reference = case["pytorch_reference"]
            assert len(direct["samples_seconds_per_invocation"]) == repeats
            assert len(reference["samples_seconds_per_invocation"]) == repeats
            assert math.isclose(
                direct["median_seconds_per_invocation"],
                statistics.median(direct["samples_seconds_per_invocation"]),
            )
            assert math.isclose(
                reference["median_seconds_per_invocation"],
                statistics.median(reference["samples_seconds_per_invocation"]),
            )
            assert math.isclose(
                case["speedup_over_pytorch"],
                reference["median_seconds_per_invocation"]
                / direct["median_seconds_per_invocation"],
            )


def test_checked_in_mps008_aggregate_records_bounded_dispatch_candidate() -> None:
    aggregate = _artifact()["aggregate"]

    assert aggregate["case_count"] == 20
    assert aggregate["all_cases_win"] is True
    assert aggregate["minimum_speedup_over_pytorch"] > 1.0
    assert aggregate["maximum_absolute_error"] <= 2e-6
    assert aggregate["maximum_relative_l2_error"] <= 3e-7
    assert aggregate["decision"] == "eligible_for_dispatch_evaluation"


def test_mps008_aggregate_rejects_an_incomplete_host_compiler_matrix(
    tmp_path: Path,
) -> None:
    paths = []
    for index, run in enumerate(_artifact()["runs"][:-1]):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run), encoding="utf-8")
        paths.append(path)

    with pytest.raises(ValueError, match="exactly four raw runs"):
        aggregate_runs(paths)
