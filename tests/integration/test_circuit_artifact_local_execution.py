from __future__ import annotations

import pytest
import torch

from flagquantum.core._artifacts import (
    ArtifactKind,
    ProgramArtifact,
    ProgramArtifactV2,
    read_program_artifact_json,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter
from flagquantum.core.target_capabilities import RequirementSet
from flagquantum.errors import ExecutionError
from flagquantum.runtime.artifact_execution import execute_circuit_artifact
from flagquantum.runtime.options import ExecutionOptions

pytestmark = pytest.mark.integration


def _bell_artifact() -> ProgramArtifactV2:
    circuit = CircuitIR(
        2,
        (Instruction("h", (0,)), Instruction("cx", (0, 1))),
        dtype="complex128",
    )
    return ProgramArtifactV2.from_circuit_ir(
        circuit,
        producer="flagquantum.integration-test",
    )


def test_circuit_artifact_runs_through_canonical_local_runtime() -> None:
    artifact = read_program_artifact_json(_bell_artifact().to_json())
    assert isinstance(artifact, ProgramArtifactV2)

    result = execute_circuit_artifact(
        artifact,
        options=ExecutionOptions(
            mode="statevector",
            device="cpu",
            target="samples",
            precision="complex128",
            shots=256,
            seed=1234,
        ),
    )

    samples = result.require_samples()
    assert samples.shape == (1, 256, 2)
    assert torch.all(samples[..., 0] == samples[..., 1])
    assert result.provenance["program_artifact_identity"] == artifact.artifact_identity
    assert result.provenance["program_artifact_payload_sha256"] == (
        artifact.payload_sha256
    )
    assert result.provenance["program_artifact_circuit_content_hash"] == (
        artifact.circuit_content_hash
    )
    assert result.provenance["program_artifact_profile"] == "circuit-ir-1.0"
    assert result.compatibility["program_artifact_version"] == "2.0"
    assert result.compatibility["program_artifact_verified"] is True


def test_circuit_artifact_execution_preserves_request_identity_boundary() -> None:
    artifact = _bell_artifact()
    first = execute_circuit_artifact(
        artifact,
        options=ExecutionOptions(target="samples", shots=32, seed=7),
    )
    second = execute_circuit_artifact(
        artifact,
        options=ExecutionOptions(target="samples", shots=64, seed=7),
    )

    assert first.require_samples().shape == (1, 32, 2)
    assert second.require_samples().shape == (1, 64, 2)
    assert first.provenance["program_artifact_identity"] == (
        second.provenance["program_artifact_identity"]
    )
    assert artifact == _bell_artifact()


def test_local_artifact_execution_rejects_v1_and_target_text() -> None:
    with pytest.raises(TypeError, match="ProgramArtifactV2"):
        execute_circuit_artifact(  # type: ignore[arg-type]
            ProgramArtifact(
                kind=ArtifactKind.CIRCUIT,
                payload=_bell_artifact().payload,
                producer="legacy",
            )
        )

    circuit = _bell_artifact()
    executable = ProgramArtifactV2(
        kind=ArtifactKind.EXECUTABLE,
        producer="flagquantum.integration-test",
        profile={
            "name": "openqasm-3.0",
            "media_type": "text/x-openqasm;version=3.0;charset=utf-8",
            "encoding": "utf-8",
        },
        payload="OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nh q[0];\nc = measure q;\n",
        circuit_content_hash=circuit.circuit_content_hash,
        requirements=RequirementSet(requirements=()),
        target={"snapshot_id": "a" * 64},
        compilation={
            "target_legalization_identity": "b" * 64,
            "schedule_identity": "c" * 64,
            "emission_identity": "d" * 64,
            "conformance_identity": "e" * 64,
        },
        parameter_schema={"binding": "fully_bound", "parameters": []},
        result_schema={
            "kind": "samples",
            "wires": [0],
            "shots_source": "execution_request",
        },
    )
    with pytest.raises(ExecutionError, match="target adapter"):
        execute_circuit_artifact(executable)


def test_local_artifact_execution_rejects_symbolic_circuits() -> None:
    circuit = CircuitIR(
        1,
        (Instruction("rx", (0,), {"theta": Parameter("theta")}),),
    )
    artifact = ProgramArtifactV2.from_circuit_ir(
        circuit,
        producer="flagquantum.integration-test",
    )

    with pytest.raises(ExecutionError, match="fully bound"):
        execute_circuit_artifact(artifact)
