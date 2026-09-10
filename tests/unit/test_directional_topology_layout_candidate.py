from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "directional-topology-layout-v2-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_candidate_records_approved_compiler_slice_and_exact_token() -> None:
    candidate = _candidate()

    assert candidate["status"] == "approved_compiler_implementation_complete"
    assert candidate["approved_on"] == "2026-09-10"
    assert candidate["approval_token"] == (
        "approve API_CHANGE_PROPOSAL_024_DIRECTIONAL_TOPOLOGY_LAYOUT"
    )
    assert candidate["implementation"] == {
        "authorized": True,
        "compiler_types_added": True,
        "compiler_pipeline_changed": True,
        "physical_plan_v2_added": True,
        "core_evidence_v2_added": False,
        "runtime_verifier_added": False,
        "deployment_adapter_added": False,
    }


def test_candidate_preserves_existing_contracts() -> None:
    candidate = _candidate()

    assert candidate["evidence_schema"] == ("flagquantum.compilation_evidence_bundle")
    assert candidate["candidate_version"] == "2.0"
    assert candidate["v1_policy"] == "strict_read_only_compatibility"
    assert candidate["compatibility"] == {
        "coupling_map_changed": False,
        "compilation_evidence_v1_changed": False,
        "program_artifact_v1_changed": False,
        "program_artifact_v2_changed": False,
        "automatic_v1_promotion": False,
        "public_root_export": False,
        "default_path_change": False,
    }


def test_candidate_closes_direction_and_layout_semantics() -> None:
    candidate = _candidate()
    profile = candidate["semantic_profile"]

    assert candidate["coupling_fields"] == [
        "n_wires",
        "directed_edges",
        "direction_semantics",
    ]
    assert candidate["direction_semantics"] == "directed_cx"
    assert profile == {
        "authoritative_circuit": "flagquantum.core.ir.CircuitIR",
        "topology": "directed_cx_edges",
        "initial_layout": "explicit_complete_permutation",
        "final_layout": "identity_restored",
        "physical_capacity": "equal_to_logical_capacity",
        "initial_state": "standard_all_zero",
        "reverse_cx": "h_conjugation_when_h_and_cx_are_target_native",
        "other_asymmetric_two_qubit_rewrites": "unsupported_fail_closed",
        "second_target_ir": False,
    }


def test_candidate_limits_migration_and_claims() -> None:
    candidate = _candidate()

    assert candidate["historical_migration"]["semantic_oracle"] == ("ir_phase2_batch_c")
    assert candidate["historical_migration"]["restore_removed_private_tree"] is False
    assert candidate["limits_inherited_from_v1"] is True
    assert {
        "physical_ancilla_allocation",
        "observable_remapping",
        "calibration_aware_optimization",
        "fault_tolerant_resource_expansion",
        "qpu_correctness_or_performance",
    } <= set(candidate["excluded_claims"])
