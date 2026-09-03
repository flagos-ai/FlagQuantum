from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = (
    ROOT
    / "contracts/deployment-bridge-stage1-exit-stage2-proposal-review-candidate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_stage1_exit_candidate_binds_authorizations_and_artifacts() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    for authorization in candidate["authorization_chain"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_stage1_exit_evidence_is_complete_and_transparent() -> None:
    evidence = _candidate()["stage1_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["privacy_and_limits_fail_closed"] is True
    assert evidence["anonymous_deterministic_fixtures"] is True
    assert evidence["approved_performance_budget_passed"] is True
    assert evidence["timing_variance_observation_retained"] is True
    assert evidence["budget_auto_relaxed"] is False
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["known_blockers"] == []


def test_stage2_candidate_scope_is_proposal_only_and_non_activating() -> None:
    candidate = _candidate()
    scope = candidate["stage2_scope_if_approved"]
    decisions = candidate["proposed_decisions"]

    assert scope["feature"] == "explicit local dry-run proposal"
    assert scope["proposal_and_offline_contract_tests_only"] is True
    assert scope["runtime_implementation_authorized"] is False
    assert scope["package_conversion_authorized"] is False
    assert scope["execution_or_submission_authorized"] is False
    assert scope["provider_sdk_or_network_authorized"] is False
    assert scope["public_api_change_authorized"] is False
    assert decisions["stage1_exit_authorized"] is True
    assert decisions["stage2_proposal_authorized"] is True
    assert decisions["stage2_runtime_implementation_authorized"] is False
    assert candidate["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE1-EXIT-STAGE2-PROPOSAL"
    )
