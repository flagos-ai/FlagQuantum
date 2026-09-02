from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/ir-phase2-batch-e-performance-budget-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_e_performance_review_binds_authorization_and_artifacts() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    authorization = candidate["authorization"]

    assert candidate["status"] == "ready_for_owner_approval"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_batch_e_budget_review_keeps_exit_and_later_scope_closed() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    decisions = candidate["proposed_decisions"]

    assert decisions["batch_e_performance_budget_approved"] is True
    assert decisions["batch_e_machine_gate_authorized"] is True
    for name, value in decisions.items():
        if name not in {
            "batch_e_performance_budget_approved",
            "batch_e_machine_gate_authorized",
        }:
            assert value is False
    assert candidate["approval_command"] == (
        "approve IR-PHASE2-BATCH-E-PERFORMANCE-BUDGET"
    )
