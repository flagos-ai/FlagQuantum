from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/ir-phase2-batch-d-performance-budget-authorization.json"
)
VALIDATION = ROOT / "contracts/ir-phase2-batch-d-performance-budget-validation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_d_performance_authorization_binds_review_budget_and_baseline() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve IR-PHASE2-BATCH-D-PERFORMANCE-BUDGET"
    )
    for field in ("review_candidate", "approved_budget", "baseline"):
        artifact = authorization[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_batch_d_performance_authorization_keeps_later_gates_closed() -> None:
    decisions = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))["decisions"]

    assert decisions["batch_d_performance_budget_approved"] is True
    assert decisions["batch_d_machine_gate_authorized"] is True
    for name, value in decisions.items():
        if name not in {
            "batch_d_performance_budget_approved",
            "batch_d_machine_gate_authorized",
        }:
            assert value is False


def test_batch_d_budget_validation_is_bound_and_fail_closed() -> None:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))

    assert validation["status"] == "passed"
    for field in ("authorization", "budget"):
        artifact = validation[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert all(case["passed"] for case in validation["cases"])
    assert validation["decisions"]["batch_d_exit_authorized"] is False
    assert validation["decisions"]["batch_e_provider_packaging_authorized"] is False
    assert validation["decisions"]["public_sla_claimed"] is False
