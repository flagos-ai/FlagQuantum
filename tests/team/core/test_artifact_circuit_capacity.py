from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import flagquantum as fq
from flagquantum.core import _artifacts as artifacts
from flagquantum.core._artifacts import (
    ArtifactKind,
    ProgramArtifactV2,
    ProgramArtifactV3,
    read_program_artifact_json,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.target_capabilities import RequirementSet

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/program_artifact_v2_circuit_candidate.json"
)


def _entries(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value) + sum(_entries(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return len(value) + sum(_entries(item) for item in value)
    return 0


def _circuit_with_entries(count: int) -> CircuitIR:
    circuit = fq.Circuit(2).to_ir().to_dict()
    circuit["metadata"] = {"padding": []}
    circuit["metadata"]["padding"] = [0] * (count - _entries(circuit))
    assert _entries(circuit) == count
    return CircuitIR.from_dict(circuit)


def _envelope(**changes: Any) -> dict[str, Any]:
    value = json.loads(FIXTURE.read_text())
    value.update(changes)
    value["artifact_identity"] = ""
    value["payload_sha256"] = ""
    return value


def _executable(
    version: str, padding: list[Any]
) -> ProgramArtifactV2 | ProgramArtifactV3:
    allocated = version == "3.0"
    compilation = {
        "target_legalization_identity": "b" * 64,
        "schedule_identity": "c" * 64,
        "emission_identity": "d" * 64,
        "conformance_identity": "e" * 64,
    }
    if allocated:
        compilation.update(
            physical_plan_identity="1" * 64, allocation_identity="2" * 64
        )
    result = (
        {
            "kind": "samples",
            "logical_wires": [0],
            "physical_result_slots": [0],
            "ordering": "logical_wire_order",
            "shots_source": "execution_request",
        }
        if allocated
        else {"kind": "samples", "wires": [0], "shots_source": "execution_request"}
    )
    cls = ProgramArtifactV3 if allocated else ProgramArtifactV2
    return cls(
        kind=ArtifactKind.EXECUTABLE,
        producer="capacity-tests",
        profile={
            "name": "openqasm-3.0",
            "media_type": "text/x-openqasm;version=3.0;charset=utf-8",
            "encoding": "utf-8",
        },
        payload="OPENQASM 3.0; qubit[1] q; h q[0];",
        circuit_content_hash="a" * 64,
        requirements=RequirementSet(
            requirements=(),
            extensions={"org.flagquantum.capacity": {"padding": padding}},
        ),
        target={"snapshot_id": "f" * 64},
        compilation=compilation,
        parameter_schema={"binding": "fully_bound", "parameters": []},
        result_schema=result,
    )


def test_expanded_circuit_round_trip_uses_independent_payload_budget() -> None:
    circuit = fq.Circuit(2)
    for _ in range(600):
        circuit.h(0)
    artifact = ProgramArtifactV2.from_circuit_ir(
        circuit.to_ir(), producer="capacity-tests"
    )
    assert len(artifact.payload["instructions"]) == 600
    assert ProgramArtifactV2.from_dict(artifact.to_dict()) == artifact
    assert read_program_artifact_json(artifact.to_json()) == artifact


def test_circuit_accepts_exact_payload_entry_limit() -> None:
    artifact = ProgramArtifactV2.from_circuit_ir(
        _circuit_with_entries(65_536), producer="capacity-tests"
    )
    assert _entries(artifact.payload) == 65_536
    assert _entries(artifact.to_dict()) > 65_536
    assert read_program_artifact_json(artifact.to_json()) == artifact


@pytest.mark.parametrize("shape", ["cumulative", "wide"])
def test_payload_entry_overflow_precedes_ir_reconstruction(
    monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    payload = (
        _circuit_with_entries(65_537).to_dict()
        if shape == "cumulative"
        else {str(index): 0 for index in range(65_537)}
    )

    def unexpected_reconstruction(*args: Any, **kwargs: Any) -> None:
        pytest.fail("over-limit payload reached CircuitIR reconstruction")

    monkeypatch.setattr(CircuitIR, "from_dict", unexpected_reconstruction)
    with pytest.raises(ValueError, match="maximum entry count 65536"):
        ProgramArtifactV2.from_dict(_envelope(payload=payload))


@pytest.mark.parametrize("version", ["2.0", "3.0"])
def test_executable_envelope_retains_exact_4096_entry_budget(version: str) -> None:
    empty = _executable(version, [])
    count = 4096 - _entries(empty._identity_payload())
    accepted = _executable(version, [0] * count)
    assert _entries(accepted._identity_payload()) == 4096
    assert read_program_artifact_json(accepted.to_json()) == accepted
    with pytest.raises(ValueError, match="maximum entry count 4096"):
        _executable(version, [0] * (count + 1))


@pytest.mark.parametrize("value", ["x" * (16 * 1024**2 + 1), "\u2603" * (3 * 1024**2)])
def test_payload_bytes_rejected_before_ir_reconstruction(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    payload = json.loads(FIXTURE.read_text())["payload"]
    payload["metadata"] = {"padding": value}

    def unexpected_reconstruction(*args: Any, **kwargs: Any) -> None:
        pytest.fail("oversized payload reached CircuitIR reconstruction")

    monkeypatch.setattr(CircuitIR, "from_dict", unexpected_reconstruction)
    with pytest.raises(ValueError, match="payload exceeds maximum UTF-8 bytes"):
        ProgramArtifactV2.from_dict(_envelope(payload=payload))


def test_payload_byte_limit_is_exact() -> None:
    payload = fq.Circuit(2).to_ir().to_dict()
    payload["metadata"] = {"padding": ""}
    overhead = len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    payload["metadata"]["padding"] = "x" * (16 * 1024**2 - overhead)
    circuit = CircuitIR.from_dict(payload)
    artifact = ProgramArtifactV2.from_circuit_ir(circuit, producer="capacity-tests")
    assert len(artifacts._canonical_json_bytes(payload)) == 16 * 1024**2
    assert read_program_artifact_json(artifact.to_json()) == artifact
    payload["metadata"]["padding"] += "x"
    with pytest.raises(ValueError, match="payload exceeds maximum UTF-8 bytes"):
        ProgramArtifactV2.from_dict(_envelope(payload=payload))


def test_depth_limit_precedes_ir_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload: Any = []
    for _ in range(10):
        payload = [payload]

    def unexpected_reconstruction(*args: Any, **kwargs: Any) -> None:
        pytest.fail("deep payload reached CircuitIR reconstruction")

    monkeypatch.setattr(CircuitIR, "from_dict", unexpected_reconstruction)
    with pytest.raises(ValueError, match="maximum nesting depth 8"):
        ProgramArtifactV2.from_dict(_envelope(payload={"nested": payload}))


def test_envelope_byte_limit_is_independent_of_payload_and_entries() -> None:
    base = _executable("2.0", ["x" * 4096] * 512)
    envelope = base.to_dict()
    overhead = len(artifacts._canonical_json_bytes(envelope)) - len(base.payload)
    text_size = 18 * 1024**2 - overhead
    assert 0 < text_size < 16 * 1024**2
    envelope.update(payload="x" * text_size, payload_sha256="", artifact_identity="")
    artifact = ProgramArtifactV2.from_dict(envelope)
    assert len(artifact.to_json().encode()) == 18 * 1024**2
    with pytest.raises(ValueError, match="envelope exceeds maximum UTF-8 bytes"):
        envelope["payload"] += "x"
        ProgramArtifactV2.from_dict(envelope)


@pytest.mark.parametrize("version", ["2.0", "3.0"])
def test_structured_depth_and_string_limits_remain_unchanged(version: str) -> None:
    with pytest.raises(ValueError, match="string exceeds maximum UTF-8 bytes 4096"):
        _executable(version, ["x" * 4097])
    nested: Any = []
    for _ in range(10):
        nested = [nested]
    with pytest.raises(ValueError, match="maximum nesting depth 8"):
        _executable(version, nested)


def test_streamed_payload_encoding_matches_canonical_bytes() -> None:
    value = {"non_ascii": "\u2603", "float": -0.0, "nested": [True, None, 1.25]}
    expected = artifacts._canonical_json_bytes(value)
    assert (
        artifacts._canonical_json_bytes(value, maximum_bytes=len(expected)) == expected
    )
    with pytest.raises(ValueError, match="payload exceeds"):
        artifacts._canonical_json_bytes(value, maximum_bytes=len(expected) - 1)
