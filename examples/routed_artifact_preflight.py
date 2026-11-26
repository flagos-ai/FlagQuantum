"""Save, reload, and verify a routed executable with a local target snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.compiler.artifacts.artifact_compilation import (
    compile_circuit_artifact_for_target,
)
from flagquantum.compiler.artifacts.compilation_evidence import (
    build_compilation_evidence_bundle,
)
from flagquantum.compiler.topology.directed_topology import DirectedCouplingMap
from flagquantum.core._artifacts import (
    ProgramArtifactV2,
    ProgramArtifactV3,
    read_program_artifact_json,
)
from flagquantum.core._compilation_evidence import (
    read_compilation_evidence_bundle_json,
)
from flagquantum.core._compilation_evidence_v3 import CompilationEvidenceBundleV3
from flagquantum.core.ir import MeasurementNode
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.compute.remote.validation import prepare_artifact_deployment


def _target_snapshot(evaluated_at: datetime) -> TargetCapabilitySnapshot:
    # These capabilities describe this offline exercise's synthetic target.
    values = {
        "qubits.logical_capacity": 3,
        "limits.maximum_program_operations": 128,
        "precision.effective_dtype": "complex128",
        "gates.native": ("h", "rx", "ry", "cx", "swap"),
        "measurements.results": ("samples",),
        "artifacts.profiles": ("openqasm-3.0",),
        "limits.maximum_shots": 256,
    }
    scope = CapabilityScope(device_ids=("offline-line:0",))
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="offline-line",
            target_class="test",
            provider="flagquantum.example",
            provider_version="1",
            target_revision="1",
            environment_id="local",
        ),
        scope=scope,
        captured_at=evaluated_at.isoformat(),
        valid_until=(evaluated_at + timedelta(hours=1)).isoformat(),
        facts=tuple(
            CapabilityFact(
                name=name,
                value=value,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=(
                    FactExposure.OBSERVED
                    if name == "precision.effective_dtype"
                    else FactExposure.DECLARED
                ),
                source=FactSource(
                    kind="example_fixture", ref="routed_artifact_preflight"
                ),
            )
            for name, value in values.items()
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="routed_artifact_preflight",
                sha256=hashlib.sha256(
                    json.dumps(values, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                level=EvidenceLevel.OBSERVABLE,
                scope=scope,
            ),
        ),
    )


def run_example(output_dir: Path) -> None:
    """Persist the handoff, preflight it, and verify its state on CPU."""
    circuit = (
        fq.Circuit(2, dtype="complex128").rx(0, theta=0.63).ry(1, theta=0.37).cx(0, 1)
    )
    source = ProgramArtifactV2.from_circuit_ir(
        replace(
            circuit.to_ir(),
            measurements=(MeasurementNode("samples", (0, 1)),),
        ),
        producer="routed-example",
    )
    evaluated_at = datetime.now(timezone.utc)
    snapshot = _target_snapshot(evaluated_at)
    topology = DirectedCouplingMap(3, ((0, 1), (1, 2)))
    result = compile_circuit_artifact_for_target(
        source,
        backend="qasm",
        profile="openqasm-3.0",
        snapshot=snapshot,
        producer="flagquantum.compiler",
        evaluated_at=evaluated_at,
        coupling_map=topology,
        initial_layout=(2, 0),
        max_routing_added_operations=2,
    )
    evidence = build_compilation_evidence_bundle(
        result, snapshot=snapshot, producer="routed-example"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("source.json", source.to_json()),
        ("snapshot.json", snapshot.to_json()),
        ("program.json", result.executable_artifact.to_json()),
        ("evidence.json", evidence.to_json()),
    ):
        (output_dir / filename).write_text(payload, encoding="utf-8")
    restored_source = read_program_artifact_json(
        (output_dir / "source.json").read_text(encoding="utf-8")
    )
    snapshot = TargetCapabilitySnapshot.from_dict(
        json.loads((output_dir / "snapshot.json").read_text(encoding="utf-8"))
    )
    artifact = read_program_artifact_json(
        (output_dir / "program.json").read_text(encoding="utf-8")
    )
    evidence = read_compilation_evidence_bundle_json(
        (output_dir / "evidence.json").read_text(encoding="utf-8")
    )
    assert isinstance(restored_source, ProgramArtifactV2)
    assert isinstance(artifact, ProgramArtifactV3)
    assert isinstance(evidence, CompilationEvidenceBundleV3)
    prepared = prepare_artifact_deployment(
        artifact,
        snapshot=snapshot,
        source=restored_source,
        compilation_evidence=evidence,
        provider=snapshot.target_identity.provider,
        target_id=snapshot.target_identity.target_id,
        shots=32,
        evaluated_at=evaluated_at,
    )
    assert prepared.capability_match.executable
    assert prepared.physical_result_slots == (2, 0)
    imported = fq.from_openqasm(prepared.program)
    assert imported.measurement_qubits == (2, 0)
    swaps = tuple(item for item in imported.instructions if item.name == "swap")
    assert len(swaps) == len(evidence.physical_plan.mapping_transitions) == 2
    for item in imported.instructions:
        if len(item.wires) == 2:
            assert (
                topology.has_weak_edge(*item.wires)
                if item.name == "swap"
                else topology.has_edge(*item.wires)
            )

    options = fq.ExecutionOptions(
        backend="pytorch", device="cpu", mode="statevector", precision="complex128"
    )
    physical = (
        fq.run(imported.to_circuit(), options=options).statevector().reshape(2, 2, 2)
    )
    logical = fq.run(circuit, options=options).statevector().reshape(2, 2)
    # Physical axes 2 and 0 hold logical qubits 0 and 1.
    torch.testing.assert_close(physical[:, 0, :].T, logical, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        physical[:, 1, :], torch.zeros_like(logical), atol=1e-12, rtol=0
    )
    print(f"Target topology {topology.edges}")
    print(f"Inserted SWAPs {len(swaps)}")
    print(f"Logical result slots {prepared.physical_result_slots}")
    print(f"Saved and reloaded JSON handoff in {output_dir}")
    print("Offline preflight and CPU state verification passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_example(args.output_dir)


if __name__ == "__main__":
    main()
