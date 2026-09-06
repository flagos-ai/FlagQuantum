from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase3-batch-d-exit-batch-e-review-candidate.json"
CANDIDATE_SHA256 = "8f9b0cb1392957e4b04074fdc111654ef73e08e3d81cb983be4f01a39b825706"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_batch_d_exit_candidate_preserves_record_and_authorizations() -> None:
    candidate = _candidate()

    assert _sha256(CANDIDATE) == CANDIDATE_SHA256
    assert candidate["status"] == "ready_for_owner_approval"
    for authorization in candidate["authorization_chain"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]


def test_batch_d_exit_evidence_is_complete_and_has_no_known_blocker() -> None:
    evidence = _candidate()["batch_d_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["runtime_protocol_and_closed_lifecycle"] is True
    assert evidence["artifact_to_result_identity_chain"] is True
    assert evidence["terminal_idempotency_and_fail_closed_transitions"] is True
    assert evidence["privacy_and_offline_boundary_enforced"] is True
    assert evidence["approved_performance_budget_passed"] is True
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["known_blockers"] == []


def test_batch_e_candidate_scope_is_private_anonymous_and_narrow() -> None:
    candidate = _candidate()
    scope = candidate["batch_e_scope_if_approved"]
    decisions = candidate["proposed_decisions"]

    assert scope["feature"] == "private provider-neutral conformance suite"
    assert scope["target_families"] == [
        "local_runtime_plan",
        "synthetic_qasm_text",
        "synthetic_non_qasm_artifact",
    ]
    assert scope["provider_sdk_or_remote_submission_authorized"] is False
    assert scope["credentials_or_secrets_authorized"] is False
    assert scope["real_backend_or_job_identity_authorized"] is False
    assert scope["shadow_or_default_integration_authorized"] is False
    assert scope["public_api_change_authorized"] is False
    assert scope["batch_f_through_h_authorized"] is False
    assert decisions["batch_d_exit_authorized"] is True
    assert decisions["batch_e_entry_authorized"] is True
    assert decisions["batch_f_through_h_authorized"] is False
    assert candidate["approval_command"] == ("approve IR-PHASE3-BATCH-D-EXIT-BATCH-E")
