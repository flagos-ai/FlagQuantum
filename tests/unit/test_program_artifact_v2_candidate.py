from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flagquantum.core._artifacts import (
    ProgramArtifact,
    ProgramArtifactV2,
    read_program_artifact,
)
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "program-artifact-v2-candidate.json"
V1_FIXTURE = ROOT / "tests" / "fixtures" / "program_artifact_v1_compatibility.json"
V2_CIRCUIT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "program_artifact_v2_circuit_candidate.json"
)


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def test_candidate_records_approved_bounded_vertical_slice() -> None:
    candidate = _candidate()

    assert candidate["status"] == ("approved_symbolic_circuit_binding_complete")
    assert candidate["approved_on"] == "2026-09-10"
    assert candidate["approval_token"] == (
        "approve API_CHANGE_PROPOSAL_022_PROGRAM_ARTIFACT_V2"
    )
    assert candidate["implementation"] == {
        "authorized": True,
        "core_schema_changed": True,
        "core_v1_migrator_added": True,
        "compiler_adapter_added": True,
        "runtime_adapter_added": True,
        "deployment_adapter_added": True,
        "local_circuit_execution_added": True,
        "local_target_text_execution_added": False,
        "explicit_circuit_binding_added": True,
        "trainable_tensor_artifact_binding_added": False,
        "public_export_added": False,
    }


def test_candidate_has_closed_circuit_and_executable_profile_families() -> None:
    candidate = _candidate()

    assert candidate["envelope_schema"] == "flagquantum.program_artifact"
    assert candidate["version"] == "2.0"
    assert candidate["initial_kinds"] == ["circuit", "executable"]
    assert candidate["profiles"] == {
        "circuit-ir-1.0": {
            "kind": "circuit",
            "media_type": ("application/vnd.flagquantum.circuit-ir+json;version=1.0"),
            "encoding": "canonical-json",
            "payload": "strict_circuit_ir_to_dict",
            "requirements": None,
            "target": None,
            "compilation": None,
            "result_schema": None,
        },
        "openqasm-2.0": {
            "kind": "executable",
            "media_type": "text/x-openqasm;version=2.0;charset=utf-8",
            "encoding": "utf-8",
        },
        "openqasm-3.0": {
            "kind": "executable",
            "media_type": "text/x-openqasm;version=3.0;charset=utf-8",
            "encoding": "utf-8",
        },
        "qcis-1.0": {
            "kind": "executable",
            "media_type": "text/x-qcis;version=1.0;charset=utf-8",
            "encoding": "utf-8",
        },
    }
    assert candidate["parameter_schema"] == {
        "circuit-ir-1.0": {
            "binding": ["fully_bound", "symbolic"],
            "parameters": "sorted_names_derived_from_circuit_ir",
        },
        "executable_text_profiles": {
            "binding": "fully_bound",
            "parameters": [],
        },
    }
    assert candidate["result_schema"] == {
        "circuit-ir-1.0": None,
        "executable_text_profiles": {
            "kind": "samples",
            "wires": "dense_zero_based_full_register",
            "shots_source": "execution_request",
        },
    }


def test_candidate_separates_payload_program_compile_and_envelope_identity() -> None:
    candidate = _candidate()
    identities = candidate["identity_fields"]

    assert identities == {
        "payload": "payload_sha256",
        "final_circuit": "circuit_content_hash",
        "target": "target.snapshot_id_when_executable",
        "compilation": [
            "target_legalization_identity",
            "schedule_identity",
            "emission_identity",
            "conformance_identity",
        ],
        "envelope": "artifact_identity",
    }
    assert candidate["artifact_identity_encoding"] == {
        "algorithm": "sha256",
        "input": "all_top_level_fields_except_artifact_identity",
        "character_encoding": "utf-8",
        "sort_keys": True,
        "separators": [",", ":"],
        "ensure_ascii": True,
        "allow_nan": False,
    }


def test_candidate_keeps_request_and_sensitive_state_out_of_artifact() -> None:
    candidate = _candidate()

    assert candidate["prohibited_content_scope"] == (
        "structured_fields_outside_profile_validated_payload"
    )
    assert set(candidate["excluded_request_fields"]) == {
        "shots",
        "seed",
        "timeout",
        "priority",
        "mitigation_policy",
        "result_delivery",
    }
    prohibited = set(candidate["prohibited_content"])
    assert {
        "credentials",
        "tokens",
        "private_keys",
        "provider_sdk_objects",
        "live_handles",
        "job_ids_or_queue_state",
        "local_file_paths",
        "arbitrary_remote_urls",
    } <= prohibited


def test_candidate_sets_explicit_bounded_closed_data_limits() -> None:
    limits = _candidate()["limits"]

    assert limits == {
        "maximum_payload_utf8_bytes": 16 * 1024 * 1024,
        "maximum_envelope_utf8_bytes": 18 * 1024 * 1024,
        "maximum_nesting_depth": 8,
        "maximum_total_entries": 4096,
        "maximum_non_payload_string_utf8_bytes": 4096,
    }


def test_v1_golden_fixture_remains_accepted_with_the_exact_existing_hash() -> None:
    fixture = json.loads(V1_FIXTURE.read_text(encoding="utf-8"))
    artifact = ProgramArtifact.from_dict(fixture["artifact"])
    circuit = CircuitIR.from_dict(fixture["artifact"]["payload"])

    assert artifact.to_dict() == fixture["artifact"]
    assert artifact.content_hash == fixture["expected_content_hash"]
    encoded = json.dumps(
        fixture["artifact"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == fixture["expected_content_hash"]
    assert (
        circuit.content_hash
        == "60ed04925dc8565b3ff880b6252fe2cc6eefa0635b989497d988f1522c3bb8bb"
    )


def test_v1_circuit_migration_is_narrow_and_all_new_writes_are_v2() -> None:
    candidate = _candidate()

    assert candidate["v1_circuit_migration"] == {
        "source_kind": "circuit",
        "required_empty_fields": [
            "metadata",
            "required_capabilities",
            "parent_hashes",
        ],
        "payload_validation": "flagquantum.core.ir.CircuitIR.from_dict",
        "destination_profile": "circuit-ir-1.0",
        "other_v1_artifacts": ("readable_but_automatic_migration_fails_closed"),
    }
    assert candidate["write_policy_after_approval"] == {
        "new_artifacts": "v2_only",
        "v1": "read_only_compatibility",
    }


def test_pinned_v2_circuit_candidate_has_canonical_payload_and_envelope_hashes() -> (
    None
):
    artifact = json.loads(V2_CIRCUIT_FIXTURE.read_text(encoding="utf-8"))
    payload_bytes = json.dumps(
        artifact["payload"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    body = {key: value for key, value in artifact.items() if key != "artifact_identity"}
    body_bytes = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    assert CircuitIR.from_dict(artifact["payload"]).content_hash == payload_sha256
    assert artifact["payload_sha256"] == payload_sha256
    assert artifact["circuit_content_hash"] == payload_sha256
    assert hashlib.sha256(body_bytes).hexdigest() == artifact["artifact_identity"]


def test_candidate_requires_explicit_version_dispatch_without_conversion() -> None:
    compatibility = _candidate()["compatibility"]

    assert compatibility == {
        "v1_reader_unchanged": True,
        "v1_content_hash_unchanged": True,
        "explicit_version_dispatch": True,
        "implicit_upgrade": False,
        "implicit_downgrade": False,
        "reinterpret_v1_metadata": False,
        "old_reader_accepts_v2": False,
    }
    with pytest.raises(ValueError, match="unsupported artifact envelope version"):
        ProgramArtifact.from_dict(
            {
                "schema": "flagquantum.program_artifact",
                "version": "2.0",
                "kind": "executable",
                "producer": "must-not-be-accepted-before-approval",
                "required_capabilities": [],
                "parent_hashes": [],
                "payload": {},
                "metadata": {},
            }
        )
    v2 = read_program_artifact(
        json.loads(V2_CIRCUIT_FIXTURE.read_text(encoding="utf-8"))
    )
    assert isinstance(v2, ProgramArtifactV2)
