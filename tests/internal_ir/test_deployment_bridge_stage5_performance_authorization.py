from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/deployment-bridge-stage5-performance-budget-authorization.json"
)
VALIDATION = (
    ROOT / "contracts/deployment-bridge-stage5-performance-budget-validation.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage5_performance_authorization_binds_review_budget_and_baseline() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE5-PERFORMANCE-BUDGET"
    )
    for field in ("review_candidate", "approved_budget", "baseline"):
        artifact = authorization[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_stage5_performance_authorization_keeps_later_work_closed() -> None:
    decisions = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))["decisions"]

    allowed = {
        "stage5_performance_budget_approved",
        "stage5_machine_gate_authorized",
    }
    assert all(decisions[name] is True for name in allowed)
    assert decisions["stage5_exit_authorized"] is False
    assert decisions["stage6_proposal_authorized"] is False
    assert all(
        value is False for name, value in decisions.items() if name not in allowed
    )


def test_stage5_budget_validation_is_bound_and_fail_closed() -> None:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))

    assert validation["status"] == "passed"
    for field in ("authorization", "budget"):
        artifact = validation[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert all(case["passed"] for case in validation["cases"])
    assert all(case["deterministic_evidence_identity"] for case in validation["cases"])
    assert validation["decisions"]["stage5_exit_authorized"] is False
    assert validation["decisions"]["stage6_proposal_authorized"] is False
    assert validation["decisions"]["public_sla_claimed"] is False
