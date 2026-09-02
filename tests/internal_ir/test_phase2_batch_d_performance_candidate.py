from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "contracts/ir-phase2-batch-d-performance-baseline.json"
CANDIDATE = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase2_batch_d_performance_budget_candidate.json"
)


def test_candidate_is_unapproved_and_has_headroom_over_observed_envelope() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    budgets = {item["gate_count"]: item for item in candidate["budgets"]}

    assert baseline["status"] == "measured_not_approved"
    assert candidate["status"] == "proposed_unapproved"
    assert candidate["decisions"]["budget_approved"] is False
    for observed in baseline["observed_maxima"]:
        budget = budgets[observed["gate_count"]]
        for emitter in ("openqasm2", "openqasm3", "qcis_v1"):
            assert budget[f"{emitter}_p95_ms_max"] >= (
                observed[f"{emitter}_p95_ms"] * 1.25
            )
            assert budget[f"{emitter}_peak_host_memory_bytes_max"] >= (
                observed[f"{emitter}_peak_host_memory_bytes"] * 1.25
            )
