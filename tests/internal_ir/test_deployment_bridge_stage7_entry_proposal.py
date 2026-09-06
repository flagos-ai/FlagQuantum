from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import flagquantum as fq
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "contracts/deployment-bridge-stage7-entry-proposal.json"
REVIEW = ROOT / "contracts/deployment-bridge-stage7-entry-review-candidate.json"


def _load() -> dict[str, object]:
    return json.loads(PROPOSAL.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage7_proposal_binds_stage6_exit_and_evidence() -> None:
    proposal = _load()

    assert proposal["status"] == (
        "awaiting_explicit_provider_selection_entry_not_authorized"
    )
    for artifact in proposal["prerequisites"].values():
        assert artifact["sha256"] != "TO_BE_BOUND"
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert proposal["stage_name"] == "Deployment Bridge Stage 7"
    assert "product-roadmap Stage 7" in proposal["distinct_from"]


def test_provider_and_sandbox_are_not_inferred() -> None:
    selection = _load()["provider_selection"]

    assert selection["selection_status"] == "not_selected"
    assert selection["selected_provider_namespace"] is None
    assert selection["selected_sandbox_target"] is None
    assert selection["selection_must_be_owner_supplied"] is True
    assert selection["selection_by_inference_or_repository_history_allowed"] is False
    assert selection["selection_is_not_entry_or_live_access_approval"] is True


def test_source_identity_and_network_boundaries_are_closed() -> None:
    proposal = _load()
    source = proposal["source_and_ownership_boundary"]
    target = proposal["target_discovery_and_attestation_contract"]
    network = proposal["network_policy_contract"]

    assert source["public_export_allowed"] is False
    assert source["default_registration_allowed"] is False
    assert source["external_adapter_may_contain_flagquantum_core_source"] is False
    assert (
        source["provider_specific_sdk_or_transport_must_not_enter_program_ir"] is True
    )
    assert target["target_mismatch"] == "fail_closed_before_submission"
    assert target["production_or_billable_target"] == "deny"
    assert target["dynamic_endpoint_redirect"] == "deny"
    assert network["default"] == "deny"
    assert (
        network["provider_returned_or_environment_supplied_endpoint_allowed"] is False
    )
    assert network["network_implementation_authorized"] is False


def test_credentials_are_ephemeral_redacted_and_not_authorized() -> None:
    credential = _load()["credential_resolution_contract"]

    assert credential["reference_only_in_serializable_input"] is True
    assert credential["secret_handle_visible_only_to_future_selected_transport"] is True
    assert (
        credential[
            "secret_value_environment_variable_name_store_path_and_token_metadata_serializable"
        ]
        is False
    )
    assert (
        credential[
            "plaintext_secret_logging_caching_hashing_or_evidence_capture_allowed"
        ]
        is False
    )
    assert credential["resolver_implementation_authorized"] is False


def test_first_probe_is_bounded_non_billable_and_fail_closed() -> None:
    probe = _load()["bounded_synthetic_probe_contract"]

    assert probe["maximum_submissions"] == 1
    assert probe["maximum_jobs"] == 1
    assert probe["maximum_shots"] == 100
    assert probe["maximum_estimated_cost_microunits"] == 0
    assert probe["automatic_retry_or_fallback_allowed"] is False
    assert "manual reconciliation" in probe["unknown_submission_outcome"]
    assert probe["raw_program_or_result_payload_in_audit_evidence_allowed"] is False


def test_approval_ladder_never_implies_live_or_production_access() -> None:
    proposal = _load()
    ladder = proposal["approval_ladder"]
    decisions = proposal["proposed_entry_decisions"]
    action = proposal["next_owner_action"]

    assert ladder["current_step"] == "proposal_only"
    assert ladder["next_step"] == "explicit_provider_and_sandbox_selection"
    assert ladder["entry_approval_implies_live_access"] is False
    assert ladder["implementation_readiness_implies_live_access"] is False
    assert ladder["live_sandbox_approval_implies_production_access"] is False
    assert all(value is False for value in decisions.values())
    assert action["entry_approval_available_now"] is False
    assert action["required_form"].startswith(
        "approve DEPLOYMENT-BRIDGE-STAGE7-PROVIDER-SELECTION"
    )


def test_stage7_review_preserves_runtime_public_and_default_paths() -> None:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert review["status"] == "awaiting_explicit_provider_selection"
    assert review["evidence"]["provider_selection_inferred"] is False
    assert review["evidence"]["implementation_or_live_access_authorized"] is False
    assert all(value is False for value in review["proposed_decisions"].values())
    assert review["next_owner_action"]["entry_approval_available_now"] is False
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "activate_provider_sandbox")
