from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import flagquantum as fq
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "contracts/deployment-bridge-stage5-entry-proposal.json"
REVIEW = ROOT / "contracts/deployment-bridge-stage5-entry-review-candidate.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage5_proposal_binds_stage4_exit() -> None:
    proposal = _load(PROPOSAL)

    assert proposal["status"] == "ready_for_owner_review_implementation_not_authorized"
    for artifact in proposal["prerequisites"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert proposal["stage_name"] == "Deployment Bridge Stage 5"
    assert "product-roadmap Stage 5" in proposal["distinct_from"]


def test_stage5_lifecycle_and_decisions_are_closed_and_fail_closed() -> None:
    proposal = _load(PROPOSAL)
    states = proposal["observation_state_taxonomy"]
    decisions = proposal["decision_taxonomy"]
    lifecycle = proposal["lifecycle_contract"]

    assert len(states) == len(set(states)) == 7
    assert len(decisions) == len(set(decisions)) == 4
    assert lifecycle["acceptance_confirmed_requires_attempt_recorded"] is True
    assert lifecycle["terminal_result_requires_acceptance_confirmed"] is True
    assert lifecycle["skipped_reordered_duplicated_or_unknown_state"] == (
        "observation_rejected"
    )
    assert lifecycle["state_callback_or_side_effect_allowed"] is False


def test_identity_and_unknown_submission_facts_stay_separate() -> None:
    proposal = _load(PROPOSAL)
    identity = proposal["identity_contract"]
    unknown = proposal["unknown_submission_contract"]

    assert len(identity["required_anonymous_identities"]) == 6
    assert identity["program_identity_equals_request_identity"] is False
    assert identity["request_identity_equals_provider_job_identity"] is False
    assert identity["provider_job_identity_accepted"] is False
    assert unknown["decision"] == "reconciliation_required"
    assert unknown["same_idempotency_identity_must_be_preserved"] is True
    assert unknown["automatic_retry_allowed"] is False
    assert unknown["automatic_fallback_allowed"] is False
    assert unknown["provider_acceptance_assumed"] is False
    assert unknown["later_submission_allowed_before_reconciliation"] is False


def test_safety_budget_privacy_and_completion_are_offline_only() -> None:
    proposal = _load(PROPOSAL)
    safety = proposal["safety_evidence_contract"]
    limits = proposal["quota_cost_and_retention_contract"]
    privacy = proposal["privacy_and_isolation_contract"]
    completion = proposal["completion_rule"]

    assert safety["non_billable_synthetic_workload_required"] is True
    assert len(limits["caller_supplied_nonnegative_observations"]) == 4
    assert len(limits["explicit_nonnegative_policy_limits"]) == 4
    assert limits["provider_billing_or_quota_api_accessed"] is False
    assert privacy["credential_handle_value_accepted"] is False
    assert privacy["callback_hook_controller_plugin_or_transport_allowed"] is False
    assert privacy["environment_network_or_background_work_allowed"] is False
    assert completion["accepted_means_provider_integration_ready"] is False
    assert completion["accepted_means_production_ready"] is False
    assert completion["accepted_authorizes_execution_or_submission"] is False


def test_next_entry_authorizes_only_private_offline_evaluator_and_baseline() -> None:
    proposal = _load(PROPOSAL)
    decisions = proposal["proposed_entry_decisions"]
    allowed = {
        "stage5_private_offline_observation_evaluator_authorized",
        "stage5_anonymous_synthetic_fixtures_authorized",
        "stage5_independent_performance_baseline_authorized",
    }

    for name, value in decisions.items():
        assert value is (name in allowed)
    assert all(value is False for value in proposal["explicit_exclusions"].values())
    assert proposal["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE5-ENTRY"


def test_stage5_review_preserves_public_runtime_provider_and_default_paths() -> None:
    review = _load(REVIEW)

    assert review["status"] == "ready_for_owner_approval"
    decisions = review["proposed_decisions"]
    assert decisions["stage5_private_offline_observation_evaluator_authorized"] is True
    assert decisions["stage5_live_sandbox_integration_authorized"] is False
    assert decisions["runtime_adapter_access_authorized"] is False
    assert decisions["provider_sdk_or_network_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert review["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE5-ENTRY"
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "evaluate_provider_sandbox_observation")
