from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT
    / "contracts/deployment-bridge-stage3-exit-stage4-proposal-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_stage3_exit_candidate_binds_authorizations_and_artifacts() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    for authorization in candidate["authorization_chain"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_stage3_exit_evidence_is_complete_and_non_activating() -> None:
    evidence = _candidate()["stage3_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["maximum_result_is_separate_activation_review"] is True
    assert evidence["approved_performance_budget_passed"] is True
    assert evidence["timing_variance_observation_retained"] is True
    assert evidence["runtime_execution_submission_and_activation_absent"] is True
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["known_blockers"] == []


def test_stage4_candidate_scope_is_proposal_only_and_non_activating() -> None:
    candidate = _candidate()
    scope = candidate["stage4_scope_if_approved"]
    decisions = candidate["proposed_decisions"]

    assert (
        scope["feature"] == "offline failure-injection and operator-rehearsal proposal"
    )
    assert scope["proposal_and_offline_contract_tests_only"] is True
    assert scope["rehearsal_implementation_authorized"] is False
    assert scope["runtime_adapter_access_authorized"] is False
    assert scope["canary_activation_authorized"] is False
    assert scope["public_api_change_authorized"] is False
    assert decisions["stage3_exit_authorized"] is True
    assert decisions["stage4_proposal_authorized"] is True
    assert decisions["stage4_rehearsal_implementation_authorized"] is False
    assert candidate["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE3-EXIT-STAGE4-PROPOSAL"
    )
