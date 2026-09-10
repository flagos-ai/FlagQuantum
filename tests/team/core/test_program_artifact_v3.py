from __future__ import annotations

from dataclasses import replace

import pytest

from flagquantum.core._artifacts import (
    ArtifactKind,
    ProgramArtifactV3,
    read_program_artifact,
    read_program_artifact_json,
)
from flagquantum.core.target_capabilities import RequirementSet

pytestmark = pytest.mark.unit


def _artifact(**overrides: object) -> ProgramArtifactV3:
    values: dict[str, object] = {
        "kind": ArtifactKind.EXECUTABLE,
        "producer": "phase45-core-tests",
        "profile": {
            "name": "openqasm-3.0",
            "media_type": "text/x-openqasm;version=3.0;charset=utf-8",
            "encoding": "utf-8",
        },
        "payload": (
            "OPENQASM 3.0;\n"
            'include "stdgates.inc";\n'
            "qubit[3] q;\n"
            "bit[2] c;\n"
            "c[0] = measure q[0];\n"
            "c[1] = measure q[2];"
        ),
        "circuit_content_hash": "1" * 64,
        "requirements": RequirementSet(requirements=()),
        "target": {"snapshot_id": "2" * 64},
        "compilation": {
            "target_legalization_identity": "3" * 64,
            "physical_plan_identity": "4" * 64,
            "allocation_identity": "5" * 64,
            "schedule_identity": "6" * 64,
            "emission_identity": "7" * 64,
            "conformance_identity": "8" * 64,
        },
        "parameter_schema": {"binding": "fully_bound", "parameters": []},
        "result_schema": {
            "kind": "samples",
            "logical_wires": [0, 1],
            "physical_result_slots": [0, 2],
            "ordering": "logical_wire_order",
            "shots_source": "execution_request",
        },
    }
    values.update(overrides)
    return ProgramArtifactV3(**values)  # type: ignore[arg-type]


def test_v3_allocated_executable_is_canonical_immutable_and_dispatches() -> None:
    artifact = _artifact()

    assert artifact.version == "3.0"
    assert artifact.result_schema["physical_result_slots"] == (0, 2)
    assert len(artifact.payload_sha256) == len(artifact.artifact_identity) == 64
    assert ProgramArtifactV3.from_dict(artifact.to_dict()) == artifact
    assert read_program_artifact(artifact.to_dict()) == artifact
    assert read_program_artifact_json(artifact.to_json()) == artifact


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"kind": ArtifactKind.CIRCUIT}, "kind 'executable'"),
        (
            {"parameter_schema": {"binding": "symbolic", "parameters": ["x"]}},
            "fully bound",
        ),
        (
            {
                "result_schema": {
                    "kind": "samples",
                    "logical_wires": [0, 1],
                    "physical_result_slots": [2, 2],
                    "ordering": "logical_wire_order",
                    "shots_source": "execution_request",
                }
            },
            "injective physical projection",
        ),
        (
            {
                "profile": {
                    "name": "qcis-1.0",
                    "media_type": "text/x-qcis;version=1.0;charset=utf-8",
                    "encoding": "utf-8",
                }
            },
            "ordered-result OpenQASM",
        ),
    ),
)
def test_v3_rejects_incomplete_or_ambiguous_projection(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _artifact(**overrides)


def test_v3_identity_and_compilation_bindings_reject_tampering() -> None:
    artifact = _artifact()

    with pytest.raises(ValueError, match="artifact_identity"):
        replace(artifact, artifact_identity="0" * 64)
    with pytest.raises(ValueError, match="physical_plan_identity"):
        _artifact(
            compilation={
                **dict(artifact.compilation),
                "physical_plan_identity": "not-a-digest",
            }
        )
    with pytest.raises(ValueError, match="unknown v3 program artifact field"):
        ProgramArtifactV3.from_dict({**artifact.to_dict(), "future": True})


def test_v3_duplicate_json_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate program artifact field"):
        read_program_artifact_json('{"version":"3.0","version":"3.0"}')
