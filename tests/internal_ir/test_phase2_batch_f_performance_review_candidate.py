from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT / "contracts/ir-phase2-batch-f-performance-budget-review-candidate.json"
)
SUCCESSOR = (
    ROOT / "contracts/ir-phase2-batch-f-performance-budget-artifact-successor.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_f_performance_candidate_binds_authorization() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert candidate["status"] == "ready_for_owner_approval"
    authorization = candidate["authorization"]
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]


def test_batch_f_gate_successor_is_authorized_and_scope_closed() -> None:
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    authorization = successor["authorization"]

    assert successor["status"] == "authorized_artifact_successor"
    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    assert successor["predecessor_review_semantics_changed"] is False
    assert successor["public_or_default_path_changed"] is False


def test_batch_f_performance_candidate_keeps_external_and_exit_gates_closed() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    decisions = candidate["proposed_decisions"]

    assert decisions["batch_f_performance_budget_approved"] is True
    assert decisions["budget_becomes_private_regression_gate"] is True
    assert decisions["provider_sdk_or_remote_submission_authorized"] is False
    assert decisions["adapter_repository_source_authorized"] is False
    assert decisions["credentials_or_backend_ids_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert decisions["legacy_retirement_authorized"] is False
    assert decisions["phase2_exit_authorized"] is False
    assert candidate["approval_command"] == (
        "approve IR-PHASE2-BATCH-F-PERFORMANCE-BUDGET"
    )
