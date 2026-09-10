from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum
import flagquantum.compiler as compiler
import flagquantum.core as core
import flagquantum.deployment as deployment
import flagquantum.runtime as runtime

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "contracts" / "physical-resource-allocation-v3-exit-audit.json"


def _audit() -> dict[str, object]:
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def test_exit_audit_binds_complete_approved_private_chain() -> None:
    audit = _audit()
    assert audit["schema"] == "flagquantum.physical_resource_allocation_exit_audit"
    assert audit["status"] == "private_technical_exit_complete"
    assert audit["approval_token"] == (
        "approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION"
    )
    assert audit["implemented_chain"] == [
        "compiler_sparse_allocation",
        "physical_plan_v3",
        "program_artifact_v3",
        "compilation_evidence_v3",
        "runtime_read_only_verification",
        "deployment_side_effect_free_dry_run",
    ]
    assert all(audit["acceptance"].values())


def test_exit_audit_keeps_all_v3_entry_points_out_of_stable_exports() -> None:
    audit = _audit()
    prohibited = {
        "ProgramArtifactV3",
        "CompilationEvidenceBundleV3",
        "PhysicalPlanEvidenceV3",
        "MappingTransitionEvidenceV3",
        "build_compilation_evidence_bundle",
        "verify_compilation_evidence_handoff",
        "validate_executable_result_samples",
        "prepare_artifact_deployment",
    }
    for module in (flagquantum, core, compiler, runtime, deployment):
        assert prohibited.isdisjoint(set(module.__all__))
    assert audit["stable_exports_added"] == []
    assert audit["default_path_changed"] is False


def test_exit_audit_does_not_claim_external_or_fault_tolerant_capability() -> None:
    audit = _audit()
    assert audit["production_ready"] is False
    assert audit["provider_submission_enabled"] is False
    assert audit["provider_identifier_binding_enabled"] is False
    assert audit["qec_or_ftoc_claim"] is False
    assert audit["next_requires_separate_approval"] == [
        "public_api_naming_and_lifecycle",
        "provider_qubit_identifier_binding",
        "provider_submission_and_receipts",
        "general_ancilla_lifecycle",
        "qec_or_fault_tolerant_resource_semantics",
    ]
