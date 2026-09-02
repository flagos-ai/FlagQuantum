from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "contracts/ir-phase2-batch-a-successor-budget-authorization.json"
VALIDATION = ROOT / "contracts/ir-phase2-batch-a-successor-budget-validation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_budget_authorization_binds_review_and_both_budgets() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve IR-PHASE2-BATCH-A-SUCCESSOR-BUDGET"
    )
    for field in ("review_candidate", "approved_budget", "preserved_original_budget"):
        artifact = authorization[field]
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert authorization["preserved_original_budget"]["modified"] is False


def test_successor_budget_authorization_keeps_later_gates_closed() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]

    assert decisions["successor_budget_approved"] is True
    assert decisions["successor_machine_gate_authorized"] is True
    assert decisions["original_budget_snapshot_modified"] is False
    assert decisions["batch_a_exit_authorized"] is False
    assert decisions["batch_b_start_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False


def test_successor_machine_gate_validation_is_bound_and_fail_closed() -> None:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    authorization = validation["authorization"]
    budget = validation["budget"]

    assert validation["status"] == "passed"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    assert _sha256(ROOT / budget["path"]) == budget["sha256"]
    assert validation["method"]["growth_passed"] is True
    assert all(item["passed"] for item in validation["cases"])
    assert all(item["deterministic_identity"] for item in validation["cases"])
    assert validation["decisions"]["public_sla_claimed"] is False
    assert validation["decisions"]["batch_a_exit_authorized"] is False
    assert validation["decisions"]["batch_b_start_authorized"] is False
