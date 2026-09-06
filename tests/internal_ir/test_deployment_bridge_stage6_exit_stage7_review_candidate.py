from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT
    / "contracts/deployment-bridge-stage6-exit-stage7-proposal-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_stage6_exit_candidate_binds_authorizations() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    for authorization in candidate["authorization_chain"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]


def test_stage6_exit_evidence_is_complete_and_non_activating() -> None:
    evidence = _candidate()["stage6_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["maximum_evidence_is_offline_scripted_only"] is True
    assert evidence["approved_performance_budget_passed"] is True
    assert evidence["live_transport_network_and_credential_resolution_absent"] is True
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["known_blockers"] == []


def test_stage7_candidate_scope_is_proposal_only_and_requires_provider_choice() -> None:
    candidate = _candidate()
    scope = candidate["stage7_scope_if_approved"]
    decisions = candidate["proposed_decisions"]

    assert scope["feature"] == "named-provider live-sandbox activation proposal"
    assert scope["proposal_and_offline_contract_tests_only"] is True
    assert scope["provider_selection_required_before_stage7_entry"] is True
    assert scope["live_transport_implementation_authorized"] is False
    assert scope["network_or_credential_access_authorized"] is False
    assert decisions["stage6_exit_authorized"] is True
    assert decisions["stage7_proposal_authorized"] is True
    assert decisions["stage7_entry_authorized"] is False
    assert candidate["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE6-EXIT-STAGE7-PROPOSAL"
    )
