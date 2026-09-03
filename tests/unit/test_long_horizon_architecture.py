from __future__ import annotations

import json
from pathlib import Path

from flagquantum.core._artifacts import ArtifactKind, ProgramArtifact
from flagquantum.core.ir import CircuitIR, Instruction


def test_architecture_contract_declares_independent_domains() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "contracts" / "long-horizon-architecture-v1.json").read_text()
    )
    assert payload["status"] == "candidate"
    assert payload["domains"]["core"]["may_depend_on"] == []
    assert payload["domains"]["gateways"]["may_depend_on"] == ["agent_services"]
    assert set(payload["provider_kinds"]) == {
        "accelerator",
        "simulation",
        "qpu",
        "service",
    }


def test_program_artifact_wraps_circuit_without_replacing_circuit_ir() -> None:
    circuit = CircuitIR(n_wires=1, instructions=(Instruction("h", (0,)),))
    artifact = ProgramArtifact.from_circuit_ir(circuit, producer="test-compiler")
    equivalent = ProgramArtifact.from_circuit_ir(circuit, producer="test-compiler")
    assert artifact.kind is ArtifactKind.CIRCUIT
    assert artifact.payload["kind"] == "flagquantum.circuit_ir"
    assert artifact.content_hash == equivalent.content_hash
    assert len(artifact.content_hash) == 64
    restored = ProgramArtifact.from_dict(artifact.to_dict())
    assert restored == artifact
    assert restored.content_hash == artifact.content_hash
