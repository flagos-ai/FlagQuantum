from __future__ import annotations

import json
from dataclasses import fields

import pytest

import flagquantum.agent.service as agent_service_module
from flagquantum.agent import AgentApplicationService
from flagquantum.core._artifacts import ArtifactKind, ProgramArtifact
from flagquantum.core.ir import CircuitIR, Instruction, IRSerializationError

pytestmark = pytest.mark.unit

_PARENT_A = "a" * 64
_PARENT_B = "b" * 64


def _circuit_ir() -> CircuitIR:
    return CircuitIR(n_wires=1, instructions=(Instruction("h", (0,)),))


def _artifact(**overrides: object) -> ProgramArtifact:
    values: dict[str, object] = {
        "kind": ArtifactKind.CIRCUIT,
        "payload": _circuit_ir().to_dict(),
        "producer": "phase1-characterization",
        "required_capabilities": ("statevector", "gradient", "statevector"),
        "parent_hashes": (_PARENT_A, _PARENT_B),
        "metadata": {"namespace.example/note": "current-v1"},
    }
    values.update(overrides)
    return ProgramArtifact(**values)  # type: ignore[arg-type]


def test_envelope_shape_normalization_round_trip_and_identity_inputs() -> None:
    artifact = _artifact()
    serialized = artifact.to_dict()

    assert tuple(field.name for field in fields(ProgramArtifact)) == (
        "kind",
        "payload",
        "producer",
        "version",
        "required_capabilities",
        "parent_hashes",
        "metadata",
    )
    assert tuple(serialized) == (
        "schema",
        "version",
        "kind",
        "producer",
        "required_capabilities",
        "parent_hashes",
        "payload",
        "metadata",
    )
    assert artifact.required_capabilities == ("gradient", "statevector")
    assert artifact.parent_hashes == (_PARENT_A, _PARENT_B)
    assert ProgramArtifact.from_dict(json.loads(json.dumps(serialized))) == artifact

    reordered = ProgramArtifact(
        kind=artifact.kind,
        payload={
            key: serialized["payload"][key] for key in reversed(serialized["payload"])
        },
        producer=artifact.producer,
        required_capabilities=("statevector", "gradient"),
        parent_hashes=artifact.parent_hashes,
        metadata={"namespace.example/note": "current-v1"},
    )
    assert reordered.content_hash == artifact.content_hash

    identity_changes = (
        _artifact(kind=ArtifactKind.LOGICAL),
        _artifact(producer="another-producer"),
        _artifact(required_capabilities=("statevector",)),
        _artifact(parent_hashes=(_PARENT_B, _PARENT_A)),
        _artifact(payload={"different": True}),
        _artifact(metadata={"namespace.example/note": "different"}),
    )
    assert all(
        candidate.content_hash != artifact.content_hash
        for candidate in identity_changes
    )


def test_schema_version_and_top_level_shape_are_strict_with_current_coercions() -> None:
    serialized = _artifact().to_dict()

    for field_name in serialized:
        incomplete = dict(serialized)
        incomplete.pop(field_name)
        with pytest.raises(ValueError, match="missing program artifact field"):
            ProgramArtifact.from_dict(incomplete)
    with pytest.raises(ValueError, match="unknown program artifact field"):
        ProgramArtifact.from_dict({**serialized, "future": True})
    with pytest.raises(ValueError, match="invalid program artifact schema"):
        ProgramArtifact.from_dict(
            {**serialized, "schema": "flagquantum.program_artifact.v2"}
        )
    with pytest.raises(ValueError, match="unsupported artifact envelope version"):
        ProgramArtifact.from_dict({**serialized, "version": "2.0"})

    numeric_version = ProgramArtifact.from_dict({**serialized, "version": 1.0})
    numeric_producer = ProgramArtifact.from_dict({**serialized, "producer": 7})
    assert numeric_version.version == "1.0"
    assert numeric_producer.producer == "7"


def test_metadata_value_domain_is_json_like_but_not_yet_a_closed_algebra() -> None:
    class Projectable:
        def to_dict(self) -> dict[str, object]:
            return {"projected": [1, True, None]}

    artifact = _artifact(
        metadata={
            7: "numeric-key",
            "nested": ("value", {"item": Projectable()}),
        }
    )

    assert artifact.to_dict()["metadata"] == {
        "7": "numeric-key",
        "nested": ["value", {"item": {"projected": [1, True, None]}}],
    }
    with pytest.raises(TypeError, match="artifact value object is not serializable"):
        _artifact(metadata={"opaque": object()})

    non_finite = _artifact(metadata={"latency": float("nan")})
    with pytest.raises(ValueError, match="Out of range float values"):
        _ = non_finite.content_hash


def test_lineage_is_ordered_opaque_hashes_and_not_verified_parentage() -> None:
    unrelated = _artifact(parent_hashes=(_PARENT_B, _PARENT_B, _PARENT_A))
    reversed_lineage = _artifact(parent_hashes=(_PARENT_A, _PARENT_B, _PARENT_B))

    assert unrelated.parent_hashes == (_PARENT_B, _PARENT_B, _PARENT_A)
    assert unrelated.content_hash != reversed_lineage.content_hash
    with pytest.raises(ValueError, match="parent hashes must be SHA-256"):
        _artifact(parent_hashes=("not-a-hash",))


def test_payload_and_envelope_identities_are_distinct() -> None:
    ir = _circuit_ir()
    artifact = ProgramArtifact.from_circuit_ir(ir, producer="compiler-a")
    another_producer = ProgramArtifact.from_circuit_ir(ir, producer="compiler-b")

    assert CircuitIR.from_dict(artifact.payload).content_hash == ir.content_hash
    assert artifact.content_hash != ir.content_hash
    assert another_producer.content_hash != artifact.content_hash
    assert another_producer.payload == artifact.payload
    with pytest.raises(TypeError, match="artifact value bytes is not serializable"):
        _artifact(kind=ArtifactKind.EXECUTABLE, payload={"blob": b"opaque"})


def test_per_kind_payload_validation_and_capability_check_belong_to_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = _artifact(payload={"kind": "not-circuit-ir"})
    assert ProgramArtifact.from_dict(malformed.to_dict()) == malformed
    with pytest.raises(IRSerializationError, match="not a FlagQuantum CircuitIR"):
        AgentApplicationService().validate_program(malformed.to_dict())

    required = _artifact(required_capabilities=("qpu", "statevector"))
    manifest = {
        "schema": "flagquantum_agent_capabilities_v1",
        "version": "characterization",
        "contracts": {"ir_validation": True},
        "backends": {
            "cpu": {
                "name": "cpu",
                "available": True,
                "supports_statevector": True,
                "devices": ["cpu"],
            }
        },
    }
    monkeypatch.setattr(
        agent_service_module, "_agent_capabilities", lambda **_: manifest
    )

    assert (
        AgentApplicationService().validate_program(required.to_dict())["valid"] is True
    )
    plan = AgentApplicationService().plan_execution(required.to_dict())
    assert plan["executable"] is False
    assert plan["blockers"][0]["code"] == "REQUIRED_CAPABILITY_UNAVAILABLE"
    assert plan["blockers"][0]["context"]["missing_capabilities"] == ["qpu"]


@pytest.mark.parametrize("proposed_field", ["provenance", "requirements", "extensions"])
def test_proposed_top_level_fields_are_not_part_of_v1(proposed_field: str) -> None:
    serialized = _artifact().to_dict()

    with pytest.raises(ValueError, match="unknown program artifact field"):
        ProgramArtifact.from_dict({**serialized, proposed_field: {}})
