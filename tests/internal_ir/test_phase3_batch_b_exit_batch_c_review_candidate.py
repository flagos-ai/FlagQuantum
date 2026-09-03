from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase3-batch-b-exit-batch-c-review-candidate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_batch_b_exit_candidate_binds_authorizations_and_artifacts() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    for authorization in candidate["authorization_chain"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_batch_b_exit_evidence_is_complete_and_has_no_known_blocker() -> None:
    evidence = _candidate()["batch_b_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["target_ir_schema_and_identity"] is True
    assert evidence["legality_and_fail_closed_diagnostics"] is True
    assert evidence["state_order_and_gradient_differential"] is True
    assert evidence["privacy_boundary_enforced"] is True
    assert evidence["approved_performance_budget_passed"] is True
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["known_blockers"] == []


def test_batch_c_candidate_scope_is_private_and_narrow() -> None:
    candidate = _candidate()
    scope = candidate["batch_c_scope_if_approved"]
    decisions = candidate["proposed_decisions"]

    assert (
        scope["feature"] == "private artifact registry and sealed executable artifact"
    )
    assert scope["new_emitter_authorized"] is False
    assert scope["runtime_adapter_or_execution_authorized"] is False
    assert scope["provider_or_remote_submission_authorized"] is False
    assert scope["real_backend_identity_authorized"] is False
    assert scope["public_api_change_authorized"] is False
    assert scope["default_path_change_authorized"] is False
    assert scope["batch_d_through_h_authorized"] is False
    assert decisions["batch_b_exit_authorized"] is True
    assert decisions["batch_c_entry_authorized"] is True
    assert decisions["batch_d_through_h_authorized"] is False
    assert candidate["approval_command"] == ("approve IR-PHASE3-BATCH-B-EXIT-BATCH-C")
