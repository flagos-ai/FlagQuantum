"""Contracts for reproducible SV-010 benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal.evidence.statevector_local_diagonal_probe import (
    BENCHMARK,
    COMPILER_LANES,
    EVIDENCE_SCHEMA,
    HOSTS,
    IMPLEMENTATION_ID,
    RUN_SCHEMA,
    RUNNER,
    SEMANTIC_ID,
    SHAPE_MATRIX,
    _shape_record,
    aggregate_runs,
)

pytestmark = pytest.mark.unit


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
                "speedup_over_pytorch": 1.5,
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


def test_sv010_aggregate_records_canonical_identity_and_decision(
    tmp_path: Path,
) -> None:
    payload = aggregate_runs(_write_matrix(tmp_path))

    assert payload["benchmark"] == BENCHMARK
    assert payload["schema"] == EVIDENCE_SCHEMA
    assert payload["semantic_id"] == SEMANTIC_ID
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["runner"] == RUNNER
    assert payload["execution_semantics"] == "single_device_fast_path"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["release_gate_allowed"] is False
    assert payload["scalability_claim_allowed"] is False
    assert payload["aggregate"]["case_count"] == 20
    assert payload["aggregate"]["all_cases_win"] is True
    assert payload["aggregate"]["decision"] == "eligible_for_dispatch_evaluation"


def test_sv010_aggregate_rejects_incomplete_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly four raw runs"):
        aggregate_runs(_write_matrix(tmp_path)[:-1])


def test_sv010_aggregate_rejects_mixed_revisions(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["source_revision"] = "b" * 40
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="one source and measurement policy"):
        aggregate_runs(paths)
