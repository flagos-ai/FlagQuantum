from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import flagquantum as fq
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "contracts/deployment-bridge-stage3-entry-proposal.json"
REVIEW = ROOT / "contracts/deployment-bridge-stage3-entry-review-candidate.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage3_proposal_binds_stage2_exit() -> None:
    proposal = _load(PROPOSAL)

    assert proposal["status"] == "ready_for_owner_review_implementation_not_authorized"
    for artifact in proposal["prerequisites"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert proposal["stage_name"] == "Deployment Bridge Stage 3"
    assert "provider-conformance" in proposal["distinct_from"]


def test_current_readiness_is_truthful_and_does_not_imply_activation() -> None:
    readiness = _load(PROPOSAL)["current_readiness"]

    assert readiness["stage1_offline_compatibility_complete"] is True
    assert readiness["stage2_explicit_offline_dry_run_complete"] is True
    assert readiness["private_provider_neutral_conformance_harness_exists"] is True
    assert readiness["real_provider_adapter_conformance_approved"] is False
    assert readiness["canary_ready_now"] is False
    assert readiness["canary_activation_authorized"] is False


def test_stage3_private_boundary_cannot_accept_executable_or_provider_state() -> None:
    api = _load(PROPOSAL)["proposed_private_api"]

    assert api["public_exports"] == []
    assert api["default_registration"] is False
    assert api["environment_activation"] is False
    forbidden = " ".join(api["forbidden_inputs"])
    for term in (
        "RuntimeAdapter",
        "ExecutionBinding",
        "SealedExecutableArtifact",
        "credentials",
        "real provider backend or job identifier",
        "raw program",
    ):
        assert term in forbidden


def test_readiness_taxonomies_are_closed_and_never_report_activation() -> None:
    proposal = _load(PROPOSAL)
    statuses = proposal["readiness_status_taxonomy"]
    findings = proposal["finding_taxonomy"]

    assert statuses == [
        "blocked",
        "eligible_for_offline_rehearsal",
        "ready_for_separate_activation_review",
    ]
    assert len(statuses) == len(set(statuses))
    assert len(findings) == len(set(findings))
    assert "separate_activation_approval_missing" in findings
    assert not any("active" in status or "enabled" in status for status in statuses)


def test_admission_dispatch_kill_switch_and_rollback_fail_closed() -> None:
    proposal = _load(PROPOSAL)
    admission = proposal["admission_contract"]
    dispatch = proposal["authority_and_dispatch_contract"]
    rollback = proposal["kill_switch_and_rollback_contract"]

    assert admission["explicit_per_request_opt_in_required"] is True
    assert admission["environment_variable_or_global_opt_in_allowed"] is False
    assert admission["default_requests_admitted"] is False
    assert admission["missing_unknown_stale_or_ambiguous_evidence"] == "blocked"
    assert dispatch["legacy_path_authoritative_during_readiness_stage"] is True
    assert dispatch["retry_after_unknown_or_possible_submission"] is False
    assert dispatch["at_most_one_candidate_dispatch_per_attempt_identity"] is True
    assert rollback["kill_switch_fail_closed"] is True
    assert rollback["kill_switch_precedes_sampling_and_dispatch"] is True
    assert rollback["rollback_requires_package_or_receipt_rewrite"] is False
    assert len(rollback["mandatory_triggers"]) >= 7


def test_budgets_conformance_privacy_and_operational_ownership_are_complete() -> None:
    proposal = _load(PROPOSAL)
    budgets = proposal["budget_contract"]
    conformance = proposal["provider_conformance_contract"]
    privacy = proposal["observability_identity_and_privacy_contract"]
    ownership = proposal["operational_ownership_contract"]

    assert len(budgets["budgets_required"]) == 9
    assert budgets["numeric_thresholds_authorized_now"] is False
    assert budgets["budget_requires_separate_owner_approval"] is True
    assert (
        conformance["existing_provider_neutral_harness_is_sufficient_for_real_provider"]
        is False
    )
    assert conformance["real_provider_sdk_or_network_allowed_in_stage3"] is False
    assert conformance["anonymous_synthetic_attestations_only"] is True
    assert (
        privacy[
            "real_provider_backend_job_account_or_user_identity_in_portable_evidence"
        ]
        is False
    )
    assert privacy["deterministic_anonymous_evidence_required"] is True
    assert len(ownership["named_roles_required"]) == 6
    assert (
        ownership[
            "game_day_or_offline_failure_injection_required_before_activation_review"
        ]
        is True
    )


def test_rollout_is_monotonic_and_next_entry_authorizes_only_offline_step_zero() -> (
    None
):
    proposal = _load(PROPOSAL)
    ladder = proposal["rollout_ladder"]
    decisions = proposal["proposed_entry_decisions"]

    assert [item["step"] for item in ladder] == list(range(6))
    assert [item["may_be_authorized_by_next_entry"] for item in ladder] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert [item["execution"] for item in ladder] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    allowed = {
        "stage3_private_offline_readiness_evaluator_authorized",
        "stage3_anonymous_synthetic_fixtures_authorized",
        "stage3_independent_performance_baseline_authorized",
    }
    for name, value in decisions.items():
        assert value is (name in allowed)
    assert all(value is False for value in proposal["explicit_exclusions"].values())


def test_stage3_review_preserves_public_runtime_provider_and_default_paths() -> None:
    review = _load(REVIEW)

    assert review["status"] == "ready_for_owner_approval"
    decisions = review["proposed_decisions"]
    assert decisions["stage3_private_offline_readiness_evaluator_authorized"] is True
    assert decisions["canary_activation_authorized"] is False
    assert decisions["execution_or_submission_authorized"] is False
    assert decisions["provider_sdk_or_network_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert review["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE3-ENTRY"
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "evaluate_deployment_canary_readiness")
