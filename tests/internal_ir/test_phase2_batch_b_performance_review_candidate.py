from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/ir-phase2-batch-b-performance-budget-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_b_performance_review_binds_authorization_and_artifacts() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    authorization = candidate["authorization"]

    assert candidate["status"] == "ready_for_owner_approval"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_batch_b_budget_review_cannot_authorize_exit_or_routing() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    decisions = candidate["proposed_decisions"]

    assert decisions["batch_b_performance_budget_approved"] is True
    assert decisions["batch_b_machine_gate_authorized"] is True
    assert decisions["batch_b_exit_authorized"] is False
    assert decisions["batch_c_routing_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert candidate["approval_command"] == (
        "approve IR-PHASE2-BATCH-B-PERFORMANCE-BUDGET"
    )
