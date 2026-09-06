from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import flagquantum as fq
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "contracts/deployment-bridge-stage6-entry-proposal.json"
REVIEW = ROOT / "contracts/deployment-bridge-stage6-entry-review-candidate.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage6_proposal_binds_stage5_exit() -> None:
    proposal = _load(PROPOSAL)

    assert proposal["status"] == "ready_for_owner_review_implementation_not_authorized"
    for artifact in proposal["prerequisites"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert proposal["stage_name"] == "Deployment Bridge Stage 6"
    assert "product-roadmap Stage 6" in proposal["distinct_from"]


def test_operations_states_terminals_and_errors_are_closed() -> None:
    proposal = _load(PROPOSAL)

    for key in (
        "operation_taxonomy",
        "lifecycle_state_taxonomy",
        "terminal_state_taxonomy",
        "error_taxonomy",
    ):
        values = proposal[key]
        assert values
        assert len(values) == len(set(values))
    assert set(proposal["terminal_state_taxonomy"]) <= set(
        proposal["lifecycle_state_taxonomy"]
    )


def test_first_transport_is_scripted_finite_and_offline() -> None:
    transport = _load(PROPOSAL)["transport_contract"]

    assert transport["first_implementation"] == (
        "finite immutable in-memory scripted transcript only"
    )
    assert transport["allowed_operations_exactly_match_operation_taxonomy"] is True
    assert transport["one_response_consumed_per_operation"] is True
    assert transport["unexpected_missing_extra_or_reordered_operation"] == "fail_closed"
    assert (
        transport[
            "network_provider_sdk_socket_filesystem_environment_or_subprocess_allowed"
        ]
        is False
    )
    assert transport["real_transport_implementation_authorized"] is False


def test_target_credential_and_request_identities_are_separated() -> None:
    proposal = _load(PROPOSAL)
    target = proposal["target_attestation_contract"]
    credential = proposal["credential_reference_contract"]
    request = proposal["request_and_idempotency_contract"]

    assert target["non_billable_and_sandbox_only_must_be_true"] is True
    assert (
        target["target_identity_enters_execution_evidence_not_program_identity"] is True
    )
    assert (
        target["display_name_endpoint_region_account_and_real_backend_id_accepted"]
        is False
    )
    assert (
        credential[
            "secret_value_endpoint_environment_variable_name_or_store_path_accepted"
        ]
        is False
    )
    assert credential["reference_resolution_authorized"] is False
    assert credential["reference_enters_program_or_artifact_identity"] is False
    assert request["submission_count"] == 1
    assert request["automatic_retry_allowed"] is False
    assert request["automatic_fallback_allowed"] is False
    assert request["unknown_outcome_requires_reconciliation"] is True
    assert request["terminal_state_is_irreversible"] is True


def test_report_safety_and_two_step_activation_fail_closed() -> None:
    proposal = _load(PROPOSAL)
    report = proposal["normalized_report_contract"]
    safety = proposal["policy_and_safety_contract"]
    activation = proposal["two_step_activation_contract"]

    assert report["provider_job_identity_accepted_in_first_implementation"] is False
    assert (
        report["raw_provider_status_error_receipt_result_or_metadata_accepted"] is False
    )
    assert report["actual_target_must_equal_attested_requested_target"] is True
    assert report["sandbox_or_production_certification_claim_allowed"] is False
    assert safety["kill_switch_available_and_current_required"] is True
    assert safety["all_limits_explicit_immutable_and_fail_closed"] is True
    assert activation["step_one_authorizes_step_two"] is False
    assert activation["network_or_credential_access_before_step_two"] is False
    assert (
        activation["production_target_or_billable_work_allowed_in_either_step"] is False
    )


def test_next_entry_authorizes_only_private_contracts_script_and_baseline() -> None:
    proposal = _load(PROPOSAL)
    decisions = proposal["proposed_entry_decisions"]
    allowed = {
        "stage6_private_connector_contracts_authorized",
        "stage6_scripted_transport_and_anonymous_fixtures_authorized",
        "stage6_independent_performance_baseline_authorized",
    }

    for name, value in decisions.items():
        assert value is (name in allowed)
    assert all(value is False for value in proposal["explicit_exclusions"].values())
    assert proposal["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE6-ENTRY"


def test_stage6_review_preserves_public_runtime_provider_and_default_paths() -> None:
    review = _load(REVIEW)

    assert review["status"] == "ready_for_owner_approval"
    decisions = review["proposed_decisions"]
    assert decisions["stage6_private_connector_contracts_authorized"] is True
    assert (
        decisions["stage6_scripted_transport_and_anonymous_fixtures_authorized"] is True
    )
    assert decisions["stage6_live_sandbox_transport_authorized"] is False
    assert decisions["provider_sdk_or_network_authorized"] is False
    assert decisions["credential_reference_resolution_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert review["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE6-ENTRY"
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "run_scripted_sandbox_connector")
