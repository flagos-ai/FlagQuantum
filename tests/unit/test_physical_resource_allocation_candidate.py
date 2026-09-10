from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "physical-resource-allocation-v3-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_candidate_records_exact_owner_approval_and_compiler_slice() -> None:
    candidate = _candidate()

    assert candidate["status"] == "approved_physical_plan_v3_complete"
    assert candidate["approved_on"] == "2026-09-10"
    assert candidate["approval_token"] == (
        "approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION"
    )
    assert candidate["implementation"] == {
        "authorized": True,
        "compiler_allocation_added": True,
        "physical_plan_v3_added": True,
        "program_artifact_v3_added": False,
        "core_evidence_v3_added": False,
        "runtime_verifier_added": False,
        "deployment_adapter_added": False,
    }


def test_candidate_separates_wire_domains_and_result_projection() -> None:
    candidate = _candidate()

    assert candidate["wire_domains"] == [
        "logical_wire",
        "physical_slot",
        "provider_qubit_identifier",
        "result_position",
    ]
    assert candidate["program_artifact_v3_result_fields"] == [
        "kind",
        "logical_wires",
        "physical_result_slots",
        "ordering",
        "shots_source",
    ]
    profile = candidate["semantic_profile"]
    assert profile["initial_layout"] == "injective_logical_to_physical"
    assert profile["occupancy"] == "complete_nullable_physical_to_logical"
    assert profile["result_order"] == "logical_wire_order"
    assert profile["second_target_ir"] is False


def test_candidate_preserves_prior_versions_and_excludes_qec_claims() -> None:
    candidate = _candidate()

    assert candidate["program_artifact_candidate_version"] == "3.0"
    assert candidate["compilation_evidence_candidate_version"] == "3.0"
    assert candidate["physical_plan_candidate_version"] == "3.0"
    assert candidate["compatibility"] == {
        "coupling_map_changed": False,
        "directed_coupling_map_v2_semantics_changed": False,
        "program_artifact_v1_changed": False,
        "program_artifact_v2_changed": False,
        "compilation_evidence_v1_changed": False,
        "compilation_evidence_v2_changed": False,
        "automatic_prior_version_promotion": False,
        "public_root_export": False,
        "default_path_change": False,
    }
    assert {
        "qec_syndrome_extraction",
        "fault_tolerant_resource_expansion",
        "provider_identifier_binding_or_submission",
    } <= set(candidate["excluded_claims"])
