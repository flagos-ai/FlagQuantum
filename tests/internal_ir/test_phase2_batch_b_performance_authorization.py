from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/ir-phase2-batch-b-performance-budget-authorization.json"
)
VALIDATION = ROOT / "contracts/ir-phase2-batch-b-performance-budget-validation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_b_performance_authorization_binds_review_budget_and_baseline() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve IR-PHASE2-BATCH-B-PERFORMANCE-BUDGET"
    )
    for field in ("review_candidate", "approved_budget", "baseline"):
        artifact = authorization[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_batch_b_performance_authorization_stops_before_exit_and_routing() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]

    assert decisions["batch_b_performance_budget_approved"] is True
    assert decisions["batch_b_machine_gate_authorized"] is True
    assert decisions["batch_b_exit_authorized"] is False
    assert decisions["batch_c_routing_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False


def test_batch_b_budget_validation_is_bound_and_fail_closed() -> None:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))

    assert validation["status"] == "passed"
    for field in ("authorization", "budget"):
        artifact = validation[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert all(case["passed"] for case in validation["cases"])
    assert all(case["deterministic_identity"] for case in validation["cases"])
    assert validation["decisions"]["batch_b_exit_authorized"] is False
    assert validation["decisions"]["batch_c_routing_authorized"] is False
    assert validation["decisions"]["public_sla_claimed"] is False
