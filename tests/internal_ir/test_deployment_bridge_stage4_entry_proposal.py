from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import flagquantum as fq
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "contracts/deployment-bridge-stage4-entry-proposal.json"
REVIEW = ROOT / "contracts/deployment-bridge-stage4-entry-review-candidate.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage4_proposal_binds_stage3_exit() -> None:
    proposal = _load(PROPOSAL)

    assert proposal["status"] == "ready_for_owner_review_implementation_not_authorized"
    for artifact in proposal["prerequisites"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert proposal["stage_name"] == "Deployment Bridge Stage 4"
    assert "Unified IR Phase 4" in proposal["distinct_from"]


def test_stage4_taxonomies_are_closed_unique_and_non_activating() -> None:
    proposal = _load(PROPOSAL)

    for key in (
        "scenario_taxonomy",
        "state_taxonomy",
        "simulated_action_taxonomy",
        "outcome_taxonomy",
        "role_taxonomy",
    ):
        values = proposal[key]
        assert values
        assert len(values) == len(set(values))
    assert len(proposal["scenario_taxonomy"]) == 13
    assert not any(
        "production_ready" in item or "canary_active" in item
        for item in proposal["outcome_taxonomy"]
    )


def test_state_machine_is_deterministic_simulated_and_fail_closed() -> None:
    machine = _load(PROPOSAL)["state_machine_contract"]

    assert machine["initial_state"] == "not_started"
    assert machine["normal_transition_order"][0] == "not_started"
    assert machine["normal_transition_order"][-1] == "completed"
    assert machine["skipped_reordered_duplicated_or_unknown_transition"] == (
        "rehearsal_failed"
    )
    assert machine["all_actions_are_simulated_evidence_only"] is True
    assert machine["state_transition_callback_or_side_effect_allowed"] is False
    assert machine["production_readiness_or_activation_state_exists"] is False


def test_every_scenario_has_only_closed_roles_and_actions() -> None:
    proposal = _load(PROPOSAL)
    scenarios = {
        item["scenario"]: item for item in proposal["scenario_response_contract"]
    }
    allowed_roles = set(proposal["role_taxonomy"])
    allowed_actions = set(proposal["simulated_action_taxonomy"])

    assert set(scenarios) == set(proposal["scenario_taxonomy"])
    for scenario in scenarios.values():
        assert set(scenario["required_roles"]) <= allowed_roles
        assert set(scenario["required_actions"]) <= allowed_actions
        assert "record_detection" in scenario["required_actions"]
        assert "block_new_candidate_admission" in scenario["required_actions"]
        assert "preserve_legacy_authority" in scenario["required_actions"]
        assert "seal_anonymous_rehearsal_evidence" in scenario["required_actions"]


def test_unknown_submission_freezes_retry_and_never_falls_back() -> None:
    proposal = _load(PROPOSAL)
    contract = proposal["unknown_submission_contract"]
    scenario = next(
        item
        for item in proposal["scenario_response_contract"]
        if item["scenario"] == "unknown_submission_outcome"
    )

    assert contract["candidate_retry_decision"] == "freeze"
    assert contract["automatic_legacy_fallback_submission"] is False
    assert contract["receipt_or_provider_status_assumed"] is False
    assert contract["reconciliation_required_before_any_later_action"] is True
    assert "freeze_candidate_retry" in scenario["required_actions"]
    assert "require_submission_reconciliation" in scenario["required_actions"]


def test_objectives_resources_privacy_and_completion_are_offline_only() -> None:
    proposal = _load(PROPOSAL)
    objectives = proposal["synthetic_objective_contract"]
    privacy = proposal["privacy_identity_and_report_contract"]
    resources = proposal["resource_and_isolation_contract"]
    completion = proposal["completion_rule"]

    assert len(objectives["observations"]) == len(objectives["limits"]) == 4
    assert objectives["wall_clock_or_sleep_used"] is False
    assert objectives["objective_breach_outcome"] == "offline_rehearsal_failed"
    assert (
        privacy["person_provider_backend_job_account_user_or_credential_included"]
        is False
    )
    assert (
        privacy["raw_program_result_payload_metadata_receipt_or_diagnostic_included"]
        is False
    )
    assert resources["one_scenario_per_invocation"] is True
    assert resources["callback_hook_controller_or_plugin_allowed"] is False
    assert (
        resources["filesystem_environment_network_or_background_work_allowed"] is False
    )
    assert completion["passed_means_production_ready"] is False
    assert completion["passed_authorizes_canary_activation"] is False


def test_next_entry_authorizes_only_private_offline_rehearsal_and_baseline() -> None:
    proposal = _load(PROPOSAL)
    decisions = proposal["proposed_entry_decisions"]
    allowed = {
        "stage4_private_offline_rehearsal_authorized",
        "stage4_anonymous_synthetic_fixtures_authorized",
        "stage4_independent_performance_baseline_authorized",
    }

    for name, value in decisions.items():
        assert value is (name in allowed)
    assert all(value is False for value in proposal["explicit_exclusions"].values())
    assert proposal["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE4-ENTRY"


def test_stage4_review_preserves_public_runtime_provider_and_default_paths() -> None:
    review = _load(REVIEW)

    assert review["status"] == "ready_for_owner_approval"
    decisions = review["proposed_decisions"]
    assert decisions["stage4_private_offline_rehearsal_authorized"] is True
    assert decisions["rehearsal_action_side_effect_authorized"] is False
    assert decisions["runtime_adapter_access_authorized"] is False
    assert decisions["canary_activation_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert review["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE4-ENTRY"
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "run_offline_deployment_rehearsal")
