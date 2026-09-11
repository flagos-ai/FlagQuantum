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
    assert payload["version"] == "1.1"
    assert payload["shared_contract_owner"] == "core"
    assert payload["domains"]["core"]["may_depend_on"] == []
    assert payload["domains"]["runtime"]["may_depend_on"] == ["core"]
    assert payload["domains"]["gateways"]["may_depend_on"] == [
        "services",
        "public_api",
    ]
    assert set(payload["compute_boundaries"]["remote"]) == {
        "qpu",
        "accelerator_service",
        "hpc_service",
        "cloud_service",
    }
    assert set(payload["compute_boundaries"]["direct"]) == {
        "cpu",
        "accelerator",
        "communication",
    }
    migration_tracks = payload["migration_tracks"]
    assert len(migration_tracks) == 8
    assert all(track["completion_evidence"] for track in migration_tracks)
    assert all(track["retirement_condition"] for track in migration_tracks)
    simulation = next(
        track for track in migration_tracks if track["name"] == "simulation_extraction"
    )
    assert simulation["status"] == "complete"
    assert simulation["current_authority"] == [simulation["target_authority"]]
    compiler = next(
        track for track in migration_tracks if track["name"] == "compiler_convergence"
    )
    assert compiler["status"] == "complete"
    assert compiler["current_authority"] == [compiler["target_authority"]]


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
