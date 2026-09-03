from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "contracts/deployment-bridge-stage2-entry-proposal.json"
REVIEW = ROOT / "contracts/deployment-bridge-stage2-entry-review-candidate.json"
SUCCESS_CACHE_AUTHORIZATION = (
    ROOT / "contracts/ir-phase1-import-success-cache-successor-authorization.json"
)
SUCCESS_CACHE_ATTESTATION = (
    ROOT
    / "tests/fixtures/internal_ir/phase1_import_success_cache_successor_candidate.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage2_proposal_binds_stage1_exit_and_current_private_boundaries() -> None:
    proposal = _load(PROPOSAL)
    review = _load(REVIEW)

    assert proposal["status"] == "ready_for_owner_review_implementation_not_authorized"
    for artifact in proposal["prerequisites"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    for name, artifact in review["current_contracts"].items():
        actual_hash = _sha256(ROOT / artifact["path"])
        if actual_hash == artifact["sha256"]:
            continue
        assert name == "verified_importer"
        attestation = _load(SUCCESS_CACHE_ATTESTATION)
        implementation = attestation["implementation"]
        assert implementation["predecessor_sha256"] == artifact["sha256"]
        assert implementation["successor_sha256"] == actual_hash
        assert implementation["successor_sha256_mode"] == "exact"
    assert proposal["stage_name"] == "Deployment Bridge Stage 2"
    assert "Quafu" in proposal["distinct_from"]


def test_stage2_pipeline_recomputes_eligibility_and_never_trusts_or_executes() -> None:
    proposal = _load(PROPOSAL)
    pipeline = {item["stage"]: item for item in proposal["pipeline_contract"]}

    assert "never trust" in pipeline["compatibility"]["rule"]
    assert "do not mutate" in pipeline["legacy_validation"]["rule"]
    assert "never cast" in pipeline["verified_import"]["rule"]
    assert "no global cache" in pipeline["private_compilation"]["rule"]
    assert "shots remain outside" in pipeline["target_legalization"]["rule"]
    assert "do not duplicate" in pipeline["canonical_encoding"]["rule"]
    assert "never return" in pipeline["seal_and_verify"]["rule"]
    assert (
        proposal["differential_contract"]["runtime_result_comparison_performed"]
        is False
    )
    assert proposal["differential_contract"]["candidate_artifact_returned"] is False


def test_stage2_identity_report_and_resource_contracts_are_closed() -> None:
    proposal = _load(PROPOSAL)
    identity = proposal["identity_contract"]
    report = proposal["report_contract"]
    resources = proposal["resource_contract"]

    assert "never reused" in identity["legacy_artifact_digest"]
    assert "no ExecutionBinding" in identity["execution_options"]
    assert "excluded" in identity["provider_backend_and_package_name"]
    assert len(proposal["status_taxonomy"]) == len(set(proposal["status_taxonomy"]))
    assert len(proposal["finding_taxonomy"]) == len(set(proposal["finding_taxonomy"]))
    for name, value in report.items():
        if name.endswith("_included"):
            assert value is False
    assert report["timings_enter_evidence_identity"] is False
    assert resources["policy_is_explicit_and_immutable"] is True
    assert resources["checks_before_and_between_stages"] is True
    assert resources["shared_or_unbounded_cache_allowed"] is False
    assert resources["background_work_allowed"] is False
    assert resources["network_allowed"] is False


def test_stage2_entry_scope_is_private_ephemeral_and_non_activating() -> None:
    proposal = _load(PROPOSAL)
    decisions = proposal["proposed_entry_decisions"]
    exclusions = proposal["explicit_exclusions"]

    assert proposal["proposed_private_api"]["public_exports"] == []
    assert proposal["proposed_private_api"]["default_registration"] is False
    assert decisions["stage2_private_implementation_authorized"] is True
    assert decisions["stage2_ephemeral_target_ir_and_artifact_authorized"] is True
    assert decisions["stage2_canonical_encoder_refactor_authorized"] is True
    assert decisions["stage2_performance_baseline_authorized"] is True
    assert decisions["stage2_performance_budget_approved"] is False
    for name, value in decisions.items():
        if name not in {
            "stage2_private_implementation_authorized",
            "stage2_ephemeral_target_ir_and_artifact_authorized",
            "stage2_canonical_encoder_refactor_authorized",
            "stage2_performance_baseline_authorized",
        }:
            assert value is False
    assert all(value is False for value in exclusions.values())
    assert proposal["performance_policy"]["budget_authorized_with_entry"] is False
    assert proposal["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE2-ENTRY"


def test_stage2_review_candidate_binds_proposal_and_preserves_public_default_path() -> (
    None
):
    review = _load(REVIEW)

    assert review["status"] == "ready_for_owner_approval"
    authorization = _load(SUCCESS_CACHE_AUTHORIZATION)
    for relative_path, expected_hash in review["reviewed_artifacts"].items():
        actual_hash = _sha256(ROOT / relative_path)
        if actual_hash == expected_hash:
            continue
        transition = authorization["test_harness_transitions"][relative_path]
        assert transition["predecessor_sha256"] == expected_hash
        assert transition["successor_sha256"] == actual_hash
    decisions = review["proposed_decisions"]
    assert decisions["stage2_private_implementation_authorized"] is True
    assert decisions["stage2_performance_baseline_authorized"] is True
    assert decisions["stage2_performance_budget_approved"] is False
    assert decisions["execution_or_submission_authorized"] is False
    assert decisions["public_api_change_authorized"] is False
    assert decisions["default_path_change_authorized"] is False
    assert review["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE2-ENTRY"
