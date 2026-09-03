from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/deployment-bridge-stage1-performance-budget-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_stage1_performance_candidate_binds_reviewed_evidence() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    assert candidate["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE1-PERFORMANCE-BUDGET"
    )
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


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
