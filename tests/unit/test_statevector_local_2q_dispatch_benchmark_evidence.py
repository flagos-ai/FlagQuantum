"""Contracts for reproducible public SV-009 dispatch evidence."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from benchmarks.statevector_local_2q_dispatch import (
    COMPILER_LANES,
    EVIDENCE_SCHEMA,
    HOSTS,
    IMPLEMENTATION_ID,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    merge_runs,
    validate_evidence,
    validate_run,
)

pytestmark = pytest.mark.unit

_REVISION = "0123456789abcdef0123456789abcdef01234567"
_EVIDENCE_REVISION = "9fb7fb6f5405c0c18481ebf36dcb780c2843aeb3"
_ARTIFACT = (
    Path(__file__).parents[2]
    / "benchmarks/results/local/statevector_local_2q_dispatch_a800.json"
)


def _run(host: str, lane: str, *, speedup: float = 1.25) -> dict[str, Any]:
    dispatch_samples = [1.0e-4, 1.1e-4, 1.2e-4]
    reference_samples = [sample * speedup for sample in dispatch_samples]
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
                "public_catalog_dispatch": {
                    "samples_seconds_per_invocation": dispatch_samples,
                    "median_seconds_per_invocation": statistics.median(
                        dispatch_samples
                    ),
                },
                "public_pytorch_reference": {
                    "samples_seconds_per_invocation": reference_samples,
                    "median_seconds_per_invocation": statistics.median(
                        reference_samples
                    ),
                },
                "public_speedup_over_pytorch": speedup,
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
        "command": "python benchmark.py",
        "host_label": host,
        "reported_hostname": host,
        "execution_semantics": "single_device_public_runtime_dispatch",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler_lane": lane,
        "compiler": {
            "distribution": distribution,
            "version": "test",
            "integration_path": integration_path,
            "identity_status": "resolved",
        },
        "environment": {},
        "measurement": {
            "warmup": 10,
            "repeats": len(dispatch_samples),
            "group_size": 10,
            "ordering": "alternating dispatch-first and reference-first groups",
            "synchronization": "once after warmup and once per timed group",
            "statistic": "median synchronized wall seconds per invocation",
        },
        "performance_floor": 1.0,
        "seed": 261_009,
        "cases": cases,
    }


def test_dispatch_run_validator_accepts_complete_measurements() -> None:
    validate_run(_run(HOSTS[0], COMPILER_LANES[0]))


def test_dispatch_run_validator_recomputes_median_and_enforces_floor() -> None:
    payload = _run(HOSTS[0], COMPILER_LANES[0])
    payload["cases"][0]["public_catalog_dispatch"]["median_seconds_per_invocation"] = (
        9.0
    )
    with pytest.raises(ValueError, match="median is invalid"):
        validate_run(payload)

    with pytest.raises(ValueError, match="performance floor"):
        validate_run(_run(HOSTS[0], COMPILER_LANES[0], speedup=0.99))


def test_dispatch_merge_requires_full_host_compiler_matrix() -> None:
    runs = [_run(host, lane) for host in HOSTS for lane in COMPILER_LANES]
    payload = merge_runs(runs, required_hosts=HOSTS)

    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["public_dispatch_win_on_all_runs"]
    assert payload["dispatch_selection_decision"] == "eligible_for_default"
    validate_evidence(payload)

    with pytest.raises(ValueError, match="host/compiler matrix"):
        merge_runs(runs[:-1], required_hosts=HOSTS)


def test_checked_in_dispatch_evidence_is_canonical_and_profitable() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    validate_evidence(copy.deepcopy(payload))
    assert payload["source_revision"] == _EVIDENCE_REVISION
    assert payload["public_speedup_range"][0] >= 1.0
    assert len(payload["runs"]) == 4
    assert sum(len(run["cases"]) for run in payload["runs"]) == 20
