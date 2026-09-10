from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "compilation-evidence-bundle-v1-candidate.json"


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_candidate_records_approved_core_compiler_scope() -> None:
    candidate = _candidate()

    assert candidate["status"] == "approved_core_compiler_implementation_complete"
    assert candidate["approved_on"] == "2026-09-10"
    assert candidate["approval_token"] == (
        "approve API_CHANGE_PROPOSAL_023_COMPILATION_EVIDENCE_BUNDLE"
    )
    assert candidate["implementation"] == {
        "authorized": True,
        "core_value_model_added": True,
        "compiler_builder_added": True,
        "runtime_verifier_added": False,
        "deployment_adapter_added": False,
        "provider_submission_added": False,
    }


def test_candidate_has_closed_role_named_records() -> None:
    candidate = _candidate()

    assert candidate["envelope_schema"] == ("flagquantum.compilation_evidence_bundle")
    assert candidate["version"] == "1.0"
    assert candidate["top_level_fields"] == [
        "schema",
        "version",
        "producer",
        "source",
        "target",
        "physical_plan",
        "output",
        "bundle_identity",
    ]
    assert candidate["closed_records"] is True
    assert candidate["duplicate_json_keys_rejected"] is True
    assert set(candidate["source_fields"]) == {
        "source_artifact_identity",
        "circuit_artifact_identity",
        "binding_identity",
        "source_circuit_hash",
        "final_circuit_hash",
    }
    assert set(candidate["output_fields"]) == {
        "profile",
        "payload_sha256",
        "emission_identity",
        "conformance_identity",
        "executable_artifact_identity",
        "artifact_compilation_identity",
    }


def test_candidate_serializes_evidence_without_a_second_circuit_ir() -> None:
    candidate = _candidate()
    profile = candidate["semantic_profile"]

    assert profile == {
        "source": "fully_bound_circuit_artifact_or_verified_binding",
        "topology": "optional_undirected_coupling",
        "initial_layout": "identity",
        "final_layout": "identity",
        "schedule": "unit_time_dependency_layers",
        "authoritative_circuit": "flagquantum.core.ir.CircuitIR",
        "second_target_ir": False,
    }
    physical_fields = set(candidate["physical_plan_fields"])
    assert {
        "plan_identity",
        "mapping_transitions",
        "instructions",
        "schedule_identity",
        "critical_path",
    } <= physical_fields
    assert "program" not in physical_fields
    assert "payload" not in physical_fields


def test_candidate_limits_security_and_claims_are_explicit() -> None:
    candidate = _candidate()

    assert candidate["limits"] == {
        "maximum_bundle_utf8_bytes": 16777216,
        "maximum_instruction_records": 4096,
        "maximum_mapping_transitions": 4096,
        "maximum_predecessors_per_instruction": 4096,
        "maximum_nesting_depth": 8,
        "maximum_non_identity_string_utf8_bytes": 256,
    }
    prohibited = set(candidate["prohibited_content"])
    assert {
        "credentials",
        "provider_sdk_objects",
        "job_ids_or_queue_state",
        "local_file_paths",
        "binary_blobs",
        "executable_code",
        "execution_request_fields",
    } <= prohibited
    assert set(candidate["excluded_claims"]) == {
        "directed_edge_synthesis",
        "physical_ancilla_allocation",
        "gate_duration",
        "pulse_scheduling",
        "crosstalk",
        "calibration_aware_optimization",
        "fault_tolerant_resource_expansion",
        "numerical_correctness",
        "hardware_performance",
    }


def test_candidate_preserves_program_artifact_compatibility() -> None:
    compatibility = _candidate()["compatibility"]

    assert compatibility == {
        "program_artifact_v1_changed": False,
        "program_artifact_v2_changed": False,
        "existing_executable_artifact_requires_bundle": False,
        "implicit_artifact_metadata_encoding": False,
        "public_root_export": False,
        "default_path_change": False,
    }
