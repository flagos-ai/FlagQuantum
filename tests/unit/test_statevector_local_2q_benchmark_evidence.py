"""Contracts for reproducible direct SV-009 evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal.evidence.statevector_local_2q_probe import (
    COMPILER_LANES,
    EVIDENCE_SCHEMA,
    HOSTS,
    IMPLEMENTATION_ID,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    _validate_run,
    aggregate_runs,
)

pytestmark = pytest.mark.unit

_REVISION = "0123456789abcdef0123456789abcdef01234567"
_EVIDENCE_REVISION = "5aef53e40f73924a718dcf866be30ecb5399037c"
_ARTIFACT = (
    Path(__file__).parents[2]
    / "benchmarks/results/local/statevector_local_2q_a800.json"
)


def _run(host: str, lane: str, *, speedup: float = 1.25) -> dict[str, Any]:
    direct_samples = [1.0e-4, 1.1e-4, 1.2e-4]
    reference_samples = [sample * speedup for sample in direct_samples]
    cases = []
    for batch, amplitudes, first_bit, second_bit in SHAPE_MATRIX:
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "amplitudes_per_batch": amplitudes,
                    "first_bit_position": first_bit,
                    "second_bit_position": second_bit,
                },
                "direct_kernel_wrapper": {
                    "samples_seconds_per_invocation": direct_samples,
                    "median_seconds_per_invocation": statistics.median(direct_samples),
                },
                "pytorch_layout_bmm_reference": {
                    "samples_seconds_per_invocation": reference_samples,
                    "median_seconds_per_invocation": statistics.median(
                        reference_samples
                    ),
                },
                "speedup_over_pytorch": speedup,
                "maximum_absolute_error": 1.0e-6,
                "relative_l2_error": 1.0e-7,
            }
        )
    distribution = "triton" if lane == "stock_triton" else "flagtree"
    integration_path = "direct" if lane == "stock_triton" else "flagtree"
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "source_revision": _REVISION,
        "host_label": host,
        "compiler_lane": lane,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler": {
            "distribution": distribution,
            "integration_path": integration_path,
            "identity_status": "resolved",
        },
        "measurement": {
            "warmup": 10,
            "repeats": len(direct_samples),
            "group_size": 10,
        },
        "cases": cases,
    }


def test_run_validator_accepts_complete_measurements() -> None:
    _validate_run(_run(HOSTS[0], COMPILER_LANES[0]))


def test_run_validator_recomputes_medians_and_compiler_identity() -> None:
    payload = _run(HOSTS[0], COMPILER_LANES[0])
    payload["cases"][0]["direct_kernel_wrapper"]["median_seconds_per_invocation"] = 9.0
    with pytest.raises(ValueError, match="median is invalid"):
        _validate_run(payload)

    payload = _run(HOSTS[0], COMPILER_LANES[0])
    payload["compiler"]["distribution"] = "flagtree"
    with pytest.raises(ValueError, match="identity does not match"):
        _validate_run(payload)


def test_aggregate_requires_full_host_compiler_matrix(tmp_path: Path) -> None:
    paths = []
    for host in HOSTS:
        for lane in COMPILER_LANES:
            path = tmp_path / f"{host}-{lane}.json"
            path.write_text(json.dumps(_run(host, lane)))
            paths.append(path)
    payload = aggregate_runs(paths)
    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["aggregate"]["all_cases_win"]
    assert payload["aggregate"]["decision"] == "eligible_for_dispatch_evaluation"

    with pytest.raises(ValueError, match="exactly four"):
        aggregate_runs(paths[:-1])


def test_checked_in_a800_evidence_is_canonical_and_profitable() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["source_revision"] == _EVIDENCE_REVISION
    assert payload["semantic_id"] == SEMANTIC_ID
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["aggregate"]["case_count"] == 20
    assert payload["aggregate"]["minimum_speedup_over_pytorch"] > 1.0
    assert payload["aggregate"]["maximum_absolute_error"] < 2.0e-5
    assert payload["aggregate"]["maximum_relative_l2_error"] < 1.0e-6
    assert payload["aggregate"]["all_cases_win"]
    for run in payload["runs"]:
        _validate_run(copy.deepcopy(run))
