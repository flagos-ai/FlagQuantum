from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase3-batch-f-exit-batch-g-review-candidate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_batch_f_exit_candidate_binds_authorizations_and_artifacts() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_owner_approval"
    for authorization in candidate["authorization_chain"].values():
        assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for relative_path, expected_hash in candidate["reviewed_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_batch_f_exit_evidence_is_complete_and_has_no_known_blocker() -> None:
    evidence = _candidate()["batch_f_exit_evidence"]

    assert evidence["authorized_scope_implemented"] is True
    assert evidence["legacy_first_and_authoritative"] is True
    assert evidence["privacy_and_failure_behavior_fail_closed"] is True
    assert evidence["bounded_overhead_and_thread_safe_kill_switch"] is True
    assert evidence["anonymous_deterministic_fixtures"] is True
    assert evidence["approved_performance_budget_passed"] is True
    assert evidence["default_path_zero_impact"] is True
    assert evidence["public_api_changed"] is False
    assert evidence["known_blockers"] == []


def test_batch_g_candidate_scope_is_proposal_only_and_non_activating() -> None:
    candidate = _candidate()
    scope = candidate["batch_g_scope_if_approved"]
    decisions = candidate["proposed_decisions"]

    assert scope["feature"] == "private deployment compatibility proposal"
    assert scope["deployment_package_mapping_required"] is True
    assert scope["rollback_and_migration_review_required"] is True
    assert scope["canary_activation_authorized"] is False
    assert scope["production_or_default_shadow_authorized"] is False
    assert scope["provider_sdk_or_network_authorized"] is False
    assert scope["public_api_change_authorized"] is False
    assert scope["legacy_retirement_authorized"] is False
    assert scope["batch_h_authorized"] is False
    assert decisions["batch_f_exit_authorized"] is True
    assert decisions["batch_g_entry_authorized"] is True
    assert decisions["batch_h_authorized"] is False
    assert candidate["approval_command"] == ("approve IR-PHASE3-BATCH-F-EXIT-BATCH-G")
