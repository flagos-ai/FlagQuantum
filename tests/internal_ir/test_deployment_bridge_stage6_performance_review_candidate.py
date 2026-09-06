from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REVIEW = (
    ROOT / "contracts/deployment-bridge-stage6-performance-budget-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage6_performance_review_binds_authorization() -> None:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert review["status"] == "ready_for_owner_approval"
    authorization = review["authorization"]
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]


def test_stage6_performance_review_authorizes_only_budget_and_gate() -> None:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    decisions = review["proposed_decisions"]
    allowed = {
        "stage6_performance_budget_approved",
        "stage6_machine_gate_authorized",
    }

    assert all(decisions[name] is True for name in allowed)
    assert all(
        value is False for name, value in decisions.items() if name not in allowed
    )
    assert review["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE6-PERFORMANCE-BUDGET"
    )
