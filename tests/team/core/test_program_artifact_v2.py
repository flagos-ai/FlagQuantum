from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flagquantum.core._artifacts import (
    ArtifactKind,
    ProgramArtifact,
    ProgramArtifactV2,
    migrate_v1_circuit_artifact,
    read_program_artifact,
    read_program_artifact_json,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter
from flagquantum.core.target_capabilities import RequirementSet

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
V1_FIXTURE = ROOT / "tests" / "fixtures" / "program_artifact_v1_compatibility.json"
V2_FIXTURE = ROOT / "tests" / "fixtures" / "program_artifact_v2_circuit_candidate.json"
HASHES = {
    "target_legalization_identity": "b" * 64,
    "schedule_identity": "c" * 64,
    "emission_identity": "d" * 64,
    "conformance_identity": "e" * 64,
}


def _fixture(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _circuit(*, symbolic: bool = False, metadata: object = None) -> CircuitIR:
    instruction = (
        Instruction("rx", (0,), {"theta": Parameter("theta")})
        if symbolic
        else Instruction("h", (0,))
    )
    return CircuitIR(
        n_wires=1,
        instructions=(instruction,),
        dtype="complex128",
        shape=(2,),
        metadata={} if metadata is None else metadata,
    )


def _executable(**overrides: object) -> ProgramArtifactV2:
    values: dict[str, object] = {
        "kind": ArtifactKind.EXECUTABLE,
        "producer": "phase32-core-tests",
        "profile": {
            "name": "openqasm-3.0",
            "media_type": "text/x-openqasm;version=3.0;charset=utf-8",
            "encoding": "utf-8",
        },
        "payload": "OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n",
        "circuit_content_hash": "a" * 64,
        "requirements": RequirementSet(requirements=()),
        "target": {"snapshot_id": "f" * 64},
        "compilation": HASHES,
        "parameter_schema": {"binding": "fully_bound", "parameters": []},
        "result_schema": {
            "kind": "samples",
            "wires": [0],
            "shots_source": "execution_request",
        },
    }
    values.update(overrides)
    return ProgramArtifactV2(**values)  # type: ignore[arg-type]


def test_v1_dispatch_and_hash_are_exactly_unchanged() -> None:
    fixture = _fixture(V1_FIXTURE)
    artifact = read_program_artifact(fixture["artifact"])  # type: ignore[arg-type]

    assert type(artifact) is ProgramArtifact
    assert artifact.to_dict() == fixture["artifact"]
    assert artifact.content_hash == fixture["expected_content_hash"]


def test_v2_circuit_constructor_matches_golden_fixture_and_round_trips() -> None:
    expected = _fixture(V2_FIXTURE)
    circuit = CircuitIR.from_dict(expected["payload"])  # type: ignore[arg-type]
    artifact = ProgramArtifactV2.from_circuit_ir(
        circuit,
        producer="phase31-v1-compatibility-fixture",
    )

    assert artifact.to_dict() == expected
    assert ProgramArtifactV2.from_dict(expected) == artifact
    assert read_program_artifact(expected) == artifact
    assert read_program_artifact_json(artifact.to_json()) == artifact


def test_v2_circuit_derives_symbolic_parameter_schema() -> None:
    artifact = ProgramArtifactV2.from_circuit_ir(
        _circuit(symbolic=True), producer="phase32-core-tests"
    )

    assert artifact.parameter_schema == {
        "binding": "symbolic",
        "parameters": ("theta",),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("payload_sha256", "0" * 64, "payload_sha256"),
        ("circuit_content_hash", "0" * 64, "circuit_content_hash"),
        ("artifact_identity", "0" * 64, "artifact_identity"),
    ],
)
def test_v2_circuit_rejects_identity_tampering(
    field: str, value: str, message: str
) -> None:
    payload = _fixture(V2_FIXTURE)
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ProgramArtifactV2.from_dict(payload)


def test_v2_dispatch_rejects_shape_and_duplicate_json_fields() -> None:
    payload = _fixture(V2_FIXTURE)
    with pytest.raises(ValueError, match="unknown v2 program artifact field"):
        read_program_artifact({**payload, "future": True})
    incomplete = dict(payload)
    incomplete.pop("profile")
    with pytest.raises(ValueError, match="missing v2 program artifact field"):
        read_program_artifact(incomplete)
    with pytest.raises(ValueError, match="duplicate program artifact field"):
        read_program_artifact_json('{"version":"2.0","version":"2.0"}')


def test_v2_executable_round_trip_and_separate_identities() -> None:
    artifact = _executable()
    restored = ProgramArtifactV2.from_dict(artifact.to_dict())

    assert restored == artifact
    assert restored.requirements == RequirementSet(requirements=())
    assert restored.payload_sha256 != restored.circuit_content_hash
    assert restored.artifact_identity not in {
        restored.payload_sha256,
        restored.circuit_content_hash,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": ArtifactKind.CIRCUIT}, "requires kind 'executable'"),
        (
            {"parameter_schema": {"binding": "symbolic", "parameters": ["x"]}},
            "must be fully bound",
        ),
        (
            {
                "result_schema": {
                    "kind": "samples",
                    "wires": [1],
                    "shots_source": "execution_request",
                }
            },
            "dense full-register samples",
        ),
        ({"requirements": None}, "require a Core RequirementSet"),
    ],
)
def test_v2_executable_rejects_cross_profile_or_incomplete_semantics(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _executable(**overrides)


def test_v2_limits_and_sensitive_structured_data_fail_closed() -> None:
    with pytest.raises(ValueError, match="producer exceeds"):
        ProgramArtifactV2.from_circuit_ir(_circuit(), producer="p" * 4097)

    nested: object = "leaf"
    for index in range(9):
        nested = {f"level_{index}": nested}
    with pytest.raises(ValueError, match="nesting depth"):
        ProgramArtifactV2.from_circuit_ir(
            _circuit(metadata=nested), producer="phase32-core-tests"
        )

    with pytest.raises(ValueError, match="entry count"):
        ProgramArtifactV2.from_circuit_ir(
            _circuit(metadata={f"key_{index}": index for index in range(4096)}),
            producer="phase32-core-tests",
        )

    requirements = RequirementSet(
        requirements=(),
        extensions={"org.flagquantum.test": {"access_token": "x"}},
    )
    with pytest.raises(ValueError, match="prohibited field"):
        _executable(requirements=requirements)


def test_v1_circuit_migration_is_explicit_and_narrow() -> None:
    fixture = _fixture(V1_FIXTURE)
    artifact = ProgramArtifact.from_dict(fixture["artifact"])  # type: ignore[arg-type]

    migrated = migrate_v1_circuit_artifact(artifact)
    assert migrated.to_dict() == _fixture(V2_FIXTURE)

    variants = (
        ProgramArtifact(
            kind=artifact.kind,
            payload=artifact.payload,
            producer=artifact.producer,
            metadata={"note": "ambiguous"},
        ),
        ProgramArtifact(
            kind=artifact.kind,
            payload=artifact.payload,
            producer=artifact.producer,
            required_capabilities=("statevector",),
        ),
        ProgramArtifact(
            kind=artifact.kind,
            payload=artifact.payload,
            producer=artifact.producer,
            parent_hashes=("a" * 64,),
        ),
        ProgramArtifact(
            kind=ArtifactKind.LOGICAL,
            payload=artifact.payload,
            producer=artifact.producer,
        ),
        ProgramArtifact(
            kind=artifact.kind,
            payload={"not": "a circuit"},
            producer=artifact.producer,
        ),
    )
    for variant in variants:
        with pytest.raises(ValueError, match="cannot migrate"):
            migrate_v1_circuit_artifact(variant)


def test_v2_payload_tamper_does_not_mutate_fixture() -> None:
    payload = _fixture(V2_FIXTURE)
    tampered = copy.deepcopy(payload)
    tampered["payload"]["instructions"][0]["opcode"] = "x"  # type: ignore[index]

    with pytest.raises(ValueError, match="payload_sha256"):
        ProgramArtifactV2.from_dict(tampered)
    assert payload == _fixture(V2_FIXTURE)
