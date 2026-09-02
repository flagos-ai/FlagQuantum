from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "contracts/ir-phase2-batch-b-performance-baseline.json"
BUDGET = (
    ROOT / "tests/fixtures/internal_ir/phase2_batch_b_performance_budget_candidate.json"
)


def test_batch_b_budget_candidate_is_measured_unapproved_and_internal() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))

    assert baseline["status"] == "measured"
    assert budget["status"] == "pending_owner_approval"
    assert budget["approval"]["approved"] is False
    assert "not a public SLA" in baseline["claim_boundary"]
    assert "not a public SLA" in budget["derivation"]["claim_boundary"]


def test_batch_b_budget_has_at_least_twenty_five_percent_headroom() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))
    measured = {item["gate_count"]: item for item in baseline["cases"]}

    for candidate in budget["budgets"]:
        observation = measured[candidate["gate_count"]]
        assert candidate["pipeline_p95_ms_max"] >= observation["p95_ms"] * 1.25
        assert candidate["peak_host_memory_bytes_max"] >= (
            observation["peak_host_memory_bytes"] * 1.25
        )
        assert observation["deterministic_identity"] is True


def test_batch_b_candidate_cannot_authorize_exit_or_routing() -> None:
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))

    assert budget["later_scope"]["batch_b_exit_authorized"] is False
    assert budget["later_scope"]["batch_c_routing_authorized"] is False
    assert budget["later_scope"]["public_or_default_path_change_authorized"] is False
