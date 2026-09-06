from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT
    / "contracts/deployment-bridge-stage4-exit-stage5-proposal-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_stage4_exit_candidate_binds_authorizations() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    for authorization in candidate["authorization_chain"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]


def test_stage4_exit_evidence_is_complete_and_non_activating() -> None:
    evidence = _candidate()["stage4_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["maximum_result_is_not_production_readiness"] is True
    assert evidence["approved_performance_budget_passed"] is True
    assert evidence["action_side_effects_and_runtime_access_absent"] is True
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["known_blockers"] == []


def test_stage5_candidate_scope_is_proposal_only_and_non_activating() -> None:
    candidate = _candidate()
    scope = candidate["stage5_scope_if_approved"]
    decisions = candidate["proposed_decisions"]

    assert scope["feature"] == "provider-sandbox observation proposal"
    assert scope["distinct_from_product_roadmap_stage5"] is True
    assert scope["proposal_and_offline_contract_tests_only"] is True
    assert scope["sandbox_implementation_authorized"] is False
    assert scope["provider_sdk_or_network_authorized"] is False
    assert scope["execution_or_submission_authorized"] is False
    assert decisions["stage4_exit_authorized"] is True
    assert decisions["stage5_proposal_authorized"] is True
    assert decisions["stage5_sandbox_implementation_authorized"] is False
    assert candidate["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE4-EXIT-STAGE5-PROPOSAL"
    )
