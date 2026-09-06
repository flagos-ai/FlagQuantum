from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/deployment-bridge-stage1-performance-budget-review-candidate.json"
)


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_stage1_performance_candidate_records_review_decision() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    assert candidate["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE1-PERFORMANCE-BUDGET"
    )


def test_stage1_performance_candidate_keeps_later_work_closed() -> None:
    decisions = _candidate()["proposed_decisions"]

    assert decisions["stage1_performance_budget_approved"] is True
    assert decisions["stage1_machine_gate_authorized"] is True
    assert decisions["stage1_exit_authorized"] is False
    for name, value in decisions.items():
        if name not in {
            "stage1_performance_budget_approved",
            "stage1_machine_gate_authorized",
        }:
            assert value is False
