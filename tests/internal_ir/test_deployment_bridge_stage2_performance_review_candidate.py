from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/deployment-bridge-stage2-performance-budget-review-candidate.json"
)


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_stage2_performance_candidate_records_review_decision() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    assert candidate["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE2-PERFORMANCE-BUDGET"
    )


def test_stage2_performance_candidate_keeps_activation_closed() -> None:
    decisions = _candidate()["proposed_decisions"]

    assert decisions["stage2_performance_budget_approved"] is True
    assert decisions["stage2_machine_gate_authorized"] is True
    assert decisions["stage2_exit_authorized"] is False
    for name, value in decisions.items():
        if name not in {
            "stage2_performance_budget_approved",
            "stage2_machine_gate_authorized",
        }:
            assert value is False
