from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pytest

from flagquantum.deployment.cloud import DeploymentPackage
from flagquantum.deployment.routing_evidence import DEPLOYMENT_PACKAGE_SCHEMA

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "contracts/deployment-bridge-stage1-entry-proposal.json"
REVIEW = ROOT / "contracts/deployment-bridge-stage1-entry-review-candidate.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage1_proposal_binds_completed_phase3_and_package_shape() -> None:
    proposal = _load(PROPOSAL)

    assert proposal["status"] == "ready_for_owner_review_implementation_not_authorized"
    for artifact in proposal["prerequisites"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert [item.name for item in fields(DeploymentPackage)] == [
        "name",
        "ir",
        "qasm",
        "shots",
        "qasm_version",
        "backend",
        "metadata",
    ]
    assert DEPLOYMENT_PACKAGE_SCHEMA == "flagquantum_deployment_package_v1"


def test_stage1_contract_has_closed_status_findings_and_non_conversion_semantics() -> (
    None
):
    proposal = _load(PROPOSAL)

    assert set(proposal["status_taxonomy"]) == {
        "eligible_for_verified_recompile",
        "requires_target_enrichment",
        "unsupported",
        "invalid",
        "limit_exceeded",
    }
    assert len(proposal["finding_taxonomy"]) == len(set(proposal["finding_taxonomy"]))
    assert "never means" in proposal["inspection_contract"]["eligibility"]
    assert "never cast" in proposal["inspection_contract"]["source_ir"]
    assert "exclude" in proposal["inspection_contract"]["provider"]


def test_stage1_report_is_deterministic_bounded_and_privacy_safe() -> None:
    proposal = _load(PROPOSAL)
    report = proposal["report_contract"]
    resources = proposal["resource_contract"]

    assert report["deterministic_canonical_identity_required"] is True
    for name, value in report.items():
        if name.endswith("_included"):
            assert value is False
    assert resources["limits_are_explicit_and_immutable"] is True
    assert len(resources["required_limits"]) == 5
    assert resources["limit_breach_fails_closed"] is True
    assert resources["background_work_allowed"] is False
    assert resources["network_allowed"] is False


def test_stage1_scope_is_one_private_module_and_keeps_activation_closed() -> None:
    proposal = _load(PROPOSAL)
    exclusions = proposal["explicit_exclusions"]

    assert proposal["proposed_private_api"]["public_exports"] == []
    assert proposal["proposed_private_api"]["default_registration"] is False
    assert proposal["allowed_changes_if_approved"][0] == (
        "flagquantum/_compiler/deployment_compatibility.py"
    )
    assert all(value is False for value in exclusions.values())
    assert proposal["performance_policy"]["baseline_authorized_with_entry"] is True
    assert proposal["performance_policy"]["budget_authorized_with_entry"] is False
    assert proposal["rollback"]["user_migration_required"] is False
    assert proposal["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE1-ENTRY"


def test_stage1_review_candidate_records_decisions() -> None:
    review = _load(REVIEW)

    assert review["status"] == "ready_for_owner_approval"
    decisions = review["proposed_decisions"]
    assert decisions["stage1_implementation_authorized"] is True
    assert decisions["stage1_performance_baseline_authorized"] is True
    assert decisions["stage1_performance_budget_approved"] is False
    assert all(
        value is False
        for name, value in decisions.items()
        if name
        not in {
            "stage1_implementation_authorized",
            "stage1_performance_baseline_authorized",
        }
    )
    assert review["approval_command"] == "approve DEPLOYMENT-BRIDGE-STAGE1-ENTRY"
