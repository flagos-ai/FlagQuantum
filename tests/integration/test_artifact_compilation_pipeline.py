from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.synthesis_offline_deployment import bell
from flagquantum.compiler.artifacts.artifact_compilation import (
    ArtifactCompilationError,
    ArtifactCompilationResult,
    compile_circuit_artifact_for_target,
)
from flagquantum.compiler.artifacts.compilation_evidence import (
    CompilationEvidenceError,
    build_compilation_evidence_bundle,
    verify_compilation_evidence_bundle,
)
from flagquantum.compiler.routing import CouplingMap
from flagquantum.compiler.topology.directed_topology import DirectedCouplingMap
from flagquantum.compute.remote.validation import prepare_artifact_deployment
from flagquantum.core._artifacts import (
    ArtifactKind,
    ProgramArtifactV2,
    bind_circuit_artifact,
    read_program_artifact_json,
)
from flagquantum.core._compilation_evidence import (
    read_compilation_evidence_bundle_json,
)
from flagquantum.core._compilation_evidence_v2 import CompilationEvidenceBundleV2
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.core.parameters import Parameter
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
from flagquantum.errors import ExecutionError
from flagquantum.runtime.artifact_preflight import preflight_executable_artifact
from flagquantum.runtime.compilation_evidence import (
    verify_compilation_evidence_handoff,
)
from flagquantum.synth import CX, H, Output, QBit, allocate, qfunc, synthesize

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 20, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("target:0",))


@qfunc
def _bell_at_line_ends(
    left: Output[QBit], middle: Output[QBit], right: Output[QBit]
) -> None:
    allocate(left)
    allocate(middle)
    allocate(right)
    H(left)
    CX(left, right)


def _snapshot(
    *,
    native_gates: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("h", ()),
        ("cx", ()),
        ("swap", ()),
        ("rx", ("theta",)),
    ),
) -> TargetCapabilitySnapshot:
    source = FactSource(kind="artifact_compilation_test", ref="target-evidence")
    values = {
        "qubits.logical_capacity": 3,
        "limits.maximum_program_operations": 128,
        "precision.effective_dtype": "complex128",
        "measurements.results": ("samples",),
        "limits.maximum_shots": 4096,
        "artifacts.profiles": ("openqasm-3.0",),
        "gates.native": tuple(
            {"name": name, "parameters": parameters}
            for name, parameters in native_gates
        ),
    }
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase35-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase35-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
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
                source=source,
            )
            for name, value in values.items()
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="target-evidence",
                sha256="a" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _artifact(*, shots: int | None = None) -> ProgramArtifactV2:
    return ProgramArtifactV2.from_circuit_ir(
        CircuitIR(
            3,
            (Instruction("h", (0,)), Instruction("cx", (0, 2))),
            dtype="complex128",
            measurements=(MeasurementNode("samples", (0, 1, 2), shots=shots),),
        ),
        producer="phase35-source",
    )


def _compile(artifact: object) -> ArtifactCompilationResult:
    return compile_circuit_artifact_for_target(  # type: ignore[arg-type]
        artifact,
        backend="qasm",
        profile="openqasm-3.0",
        snapshot=_snapshot(),
        producer="flagquantum.compiler",
        evaluated_at=_NOW,
        coupling_map=CouplingMap(3, ((0, 1), (1, 2))),
    )


def test_artifact_compilation_closes_verified_target_handoff() -> None:
    source = read_program_artifact_json(_artifact().to_json())
    assert isinstance(source, ProgramArtifactV2)

    first = _compile(source)
    second = _compile(source)
    executable = first.executable_artifact

    assert first == second
    assert first.source_artifact_identity == source.artifact_identity
    assert first.circuit_artifact_identity == source.artifact_identity
    assert first.binding_identity is None
    assert first.legalization.topology_legalization is not None
    assert first.legalization.topology_legalization.inserted_swap_count > 0
    assert first.physical_plan.legalization is first.legalization
    assert first.physical_plan.program is first.legalization.program
    assert first.physical_plan.plan_identity == second.physical_plan.plan_identity
    assert first.physical_plan.mapping_transitions
    assert first.emission.text == executable.payload
    assert first.conformance.conformance_identity == (
        executable.compilation["conformance_identity"]
    )
    assert len(first.compilation_identity) == 64

    match = preflight_executable_artifact(
        executable,
        snapshot=_snapshot(),
        shots=128,
        evaluated_at=_NOW,
    )
    prepared = prepare_artifact_deployment(
        executable,
        snapshot=_snapshot(),
        provider="flagquantum.test",
        target_id="phase35-target",
        shots=128,
        evaluated_at=_NOW,
    )
    assert match.executable
    assert prepared.program == first.emission.text
    assert prepared.artifact_identity == executable.artifact_identity
    with pytest.raises(ValueError, match="input lineage"):
        replace(first, source_artifact_identity="0" * 64)
    with pytest.raises(ValueError, match="compilation_identity"):
        replace(first, compilation_identity="0" * 64)


def test_binding_lineage_is_preserved_through_target_compilation() -> None:
    symbolic = ProgramArtifactV2.from_circuit_ir(
        CircuitIR(
            1,
            (Instruction("rx", (0,), {"theta": Parameter("theta")}),),
            dtype="complex128",
            measurements=(MeasurementNode("samples", (0,)),),
        ),
        producer="phase35-symbolic",
    )
    snapshot = _snapshot()
    prepare = partial(
        prepare_artifact_deployment,
        snapshot=snapshot,
        provider=snapshot.target_identity.provider,
        target_id=snapshot.target_identity.target_id,
        shots=128,
        evaluated_at=_NOW,
    )
    options = fq.ExecutionOptions(
        backend="pytorch", device="cpu", mode="statevector", precision="complex128"
    )
    handoffs = []
    for angle, expected in ((0.0, (1, 0)), (math.pi, (0, -1j))):
        binding = bind_circuit_artifact(
            symbolic, {"theta": angle}, producer="phase35-binding"
        )
        result = _compile(binding)
        bundle = build_compilation_evidence_bundle(
            result, snapshot=snapshot, producer="flagquantum.compiler"
        )
        evidence = read_compilation_evidence_bundle_json(bundle.to_json())
        artifact = result.executable_artifact
        assert result.source_artifact_identity == symbolic.artifact_identity
        assert (
            result.circuit_artifact_identity == binding.bound_artifact.artifact_identity
        )
        assert result.binding_identity == binding.binding_identity
        assert artifact.circuit_content_hash == result.legalization.program.content_hash
        assert evidence.source["binding_identity"] == binding.binding_identity

        prepared = prepare(artifact, source=binding, compilation_evidence=evidence)
        assert prepared.capability_match.executable
        assert prepared.compilation_evidence_identity == evidence.bundle_identity

        imported = fq.from_openqasm(prepared.program)
        assert imported.version == 3.0
        assert imported.measurement_qubits == (0,)
        state = fq.run(imported.to_circuit(), options=options).statevector()
        torch.testing.assert_close(
            state.reshape(-1),
            torch.tensor(expected, dtype=torch.complex128),
            atol=1e-12,
            rtol=0,
        )

        handoffs.append((binding, artifact, evidence))

    (_, _, zero_evidence), (pi_binding, pi_artifact, pi_evidence) = handoffs
    assert (
        zero_evidence.source["binding_identity"]
        != pi_evidence.source["binding_identity"]
    )
    with pytest.raises(ExecutionError, match="source lineage"):
        prepare(pi_artifact, source=pi_binding, compilation_evidence=zero_evidence)


def test_artifact_compilation_rejects_symbolic_and_embedded_shots() -> None:
    symbolic = ProgramArtifactV2.from_circuit_ir(
        CircuitIR(
            1,
            (Instruction("rx", (0,), {"theta": Parameter("theta")}),),
            dtype="complex128",
            measurements=(MeasurementNode("samples", (0,)),),
        ),
        producer="phase35-symbolic",
    )
    with pytest.raises(ArtifactCompilationError, match="fully bound"):
        _compile(symbolic)
    with pytest.raises(ArtifactCompilationError, match="execution request"):
        _compile(_artifact(shots=100))


def test_artifact_compilation_rejects_executable_as_source() -> None:
    executable = _compile(_artifact()).executable_artifact

    with pytest.raises(ArtifactCompilationError, match="circuit-ir-1.0"):
        _compile(executable)


def test_compilation_evidence_survives_canonical_cross_process_handoff() -> None:
    result = _compile(_artifact())
    bundle = build_compilation_evidence_bundle(
        result,
        snapshot=_snapshot(),
        producer="flagquantum.compiler",
    )
    restored = read_compilation_evidence_bundle_json(bundle.to_json())

    verify_compilation_evidence_bundle(
        restored,
        result,
        snapshot=_snapshot(),
    )
    verify_compilation_evidence_handoff(
        restored,
        result.input_artifact,
        result.executable_artifact,
        snapshot=_snapshot(),
    )
    assert restored == bundle
    assert restored.physical_plan.plan_identity == result.physical_plan.plan_identity
    assert restored.output["executable_artifact_identity"] == (
        result.executable_artifact.artifact_identity
    )
    assert restored.output["artifact_compilation_identity"] == (
        result.compilation_identity
    )

    mismatched_snapshot = replace(
        _snapshot(),
        captured_at=(_NOW + timedelta(minutes=1)).isoformat(),
        snapshot_id="",
    )
    with pytest.raises(CompilationEvidenceError, match="snapshot"):
        verify_compilation_evidence_bundle(
            restored,
            result,
            snapshot=mismatched_snapshot,
        )


def test_synthesized_bell_reaches_verified_offline_preflight() -> None:
    circuit = replace(
        synthesize(bell).circuit, measurements=(MeasurementNode("samples", (0, 1)),)
    )
    source = ProgramArtifactV2.from_circuit_ir(circuit, producer="bell-source")
    assert source.kind is ArtifactKind.CIRCUIT

    snapshot = _snapshot()
    result = compile_circuit_artifact_for_target(
        source,
        backend="qasm",
        profile="openqasm-3.0",
        snapshot=snapshot,
        producer="flagquantum.compiler",
        evaluated_at=_NOW,
        coupling_map=CouplingMap.line(2),
    )
    executable = result.executable_artifact
    assert isinstance(executable, ProgramArtifactV2)
    assert executable.kind is ArtifactKind.EXECUTABLE

    evidence = build_compilation_evidence_bundle(
        result, snapshot=snapshot, producer="flagquantum.compiler"
    )
    restored = read_compilation_evidence_bundle_json(evidence.to_json())
    assert restored.source["source_artifact_identity"] == source.artifact_identity
    assert restored.source["source_circuit_hash"] == circuit.content_hash
    assert restored.target["snapshot_id"] == snapshot.snapshot_id
    assert (
        restored.output["executable_artifact_identity"] == executable.artifact_identity
    )
    assert (
        restored.output["artifact_compilation_identity"] == result.compilation_identity
    )

    identity = snapshot.target_identity
    prepared = prepare_artifact_deployment(
        executable,
        snapshot=snapshot,
        source=source,
        compilation_evidence=restored,
        provider=identity.provider,
        target_id=identity.target_id,
        shots=256,
        evaluated_at=_NOW,
    )
    assert prepared.capability_match.executable
    assert prepared.compilation_evidence_identity == restored.bundle_identity
    assert prepared.artifact_identity == executable.artifact_identity
    assert prepared.snapshot_id == snapshot.snapshot_id
    assert prepared.profile == "openqasm-3.0"
    assert prepared.program == executable.payload == result.emission.text

    imported = fq.from_openqasm(prepared.program)
    assert imported.version == 3.0
    assert imported.measurement_qubits == (0, 1)
    options = fq.ExecutionOptions(
        backend="pytorch", device="cpu", mode="statevector", precision="complex128"
    )
    state = fq.run(imported.to_circuit(), options=options).statevector()
    amplitude = 1 / math.sqrt(2)
    expected = torch.tensor([amplitude, 0, 0, amplitude], dtype=torch.complex128)
    torch.testing.assert_close(state.reshape(-1), expected, atol=1e-12, rtol=0)

    # A refresh for the same backend has a different snapshot identity.
    refreshed = replace(
        snapshot, captured_at=(_NOW - timedelta(minutes=1)).isoformat(), snapshot_id=""
    )
    assert refreshed.snapshot_id != snapshot.snapshot_id
    with pytest.raises(
        ExecutionError, match="does not match the supplied capability snapshot"
    ):
        prepare_artifact_deployment(
            executable,
            snapshot=refreshed,
            source=source,
            compilation_evidence=restored,
            provider=identity.provider,
            target_id=identity.target_id,
            shots=256,
            evaluated_at=_NOW,
        )


def test_synthesized_bell_routes_on_line_with_native_gates() -> None:
    circuit = replace(
        synthesize(_bell_at_line_ends).circuit,
        measurements=(MeasurementNode("samples", (0, 1, 2)),),
    )
    assert any(
        instruction.name == "cx" and instruction.wires == (0, 2)
        for instruction in circuit.instructions
    )
    source = ProgramArtifactV2.from_circuit_ir(circuit, producer="routed-bell-source")
    snapshot = _snapshot(native_gates=(("h", ()), ("cz", ()), ("swap", ())))
    coupling = CouplingMap.line(3)

    def compile_with_budget(budget: int) -> ArtifactCompilationResult:
        return compile_circuit_artifact_for_target(
            source,
            backend="qasm",
            profile="openqasm-3.0",
            snapshot=snapshot,
            producer="flagquantum.compiler",
            evaluated_at=_NOW,
            coupling_map=coupling,
            routing_strategy="restore_after_each_gate",
            max_routing_added_operations=budget,
        )

    result = compile_with_budget(2)
    bundle = build_compilation_evidence_bundle(
        result, snapshot=snapshot, producer="flagquantum.compiler"
    )
    restored = read_compilation_evidence_bundle_json(bundle.to_json())
    verify_compilation_evidence_bundle(restored, result, snapshot=snapshot)
    plan = restored.physical_plan
    swaps = tuple(
        instruction.physical_wires
        for instruction in plan.instructions
        if instruction.opcode == "swap"
    )
    assert len(swaps) == 2
    assert swaps == tuple(
        transition.physical_wires for transition in plan.mapping_transitions
    )

    payload = result.executable_artifact.payload
    assert isinstance(payload, str)
    imported = fq.from_openqasm(payload)
    assert imported.version == 3.0
    assert imported.measurement_qubits == (0, 1, 2)
    assert {instruction.name for instruction in imported.instructions} == {
        "h",
        "cz",
        "swap",
    }
    assert all(
        coupling.has_edge(*instruction.wires)
        for instruction in imported.instructions
        if len(instruction.wires) == 2
    )
    assert (
        tuple(
            instruction.wires
            for instruction in imported.instructions
            if instruction.name == "swap"
        )
        == swaps
    )

    options = fq.ExecutionOptions(
        backend="pytorch", device="cpu", mode="statevector", precision="complex128"
    )
    state = fq.run(imported.to_circuit(), options=options).statevector()
    expected = torch.zeros(8, dtype=torch.complex128)
    expected[[0, 5]] = 1 / math.sqrt(2)  # |000> and |101>
    torch.testing.assert_close(state.reshape(-1), expected, atol=1e-12, rtol=0)

    with pytest.raises(
        ArtifactCompilationError, match="topology routing exceeds max_added_operations"
    ):
        compile_with_budget(0)


def test_synthesized_bell_legalizes_reverse_cx_for_offline_preflight() -> None:
    circuit = replace(
        synthesize(bell).circuit, measurements=(MeasurementNode("samples", (0, 1)),)
    )
    source = ProgramArtifactV2.from_circuit_ir(circuit, producer="bell-source")
    snapshot = _snapshot(native_gates=(("h", ()), ("cx", ())))
    coupling = DirectedCouplingMap(2, ((1, 0),))

    def compile_with_direction_budget(budget: int) -> ArtifactCompilationResult:
        return compile_circuit_artifact_for_target(
            source,
            backend="qasm",
            profile="openqasm-3.0",
            snapshot=snapshot,
            producer="flagquantum.compiler",
            evaluated_at=_NOW,
            coupling_map=coupling,
            initial_layout=(0, 1),
            max_routing_added_operations=0,
            max_direction_added_operations=budget,
        )

    result = compile_with_direction_budget(4)
    bundle = build_compilation_evidence_bundle(
        result, snapshot=snapshot, producer="flagquantum.compiler"
    )
    restored = read_compilation_evidence_bundle_json(bundle.to_json())
    assert isinstance(restored, CompilationEvidenceBundleV2)
    assert restored.physical_plan.initial_logical_to_physical == (0, 1)
    assert restored.physical_plan.reversed_cx_count == 1

    identity = snapshot.target_identity
    prepared = prepare_artifact_deployment(
        result.executable_artifact,
        snapshot=snapshot,
        source=source,
        compilation_evidence=restored,
        provider=identity.provider,
        target_id=identity.target_id,
        shots=256,
        evaluated_at=_NOW,
    )
    assert prepared.capability_match.executable
    assert prepared.compilation_evidence_identity == restored.bundle_identity

    imported = fq.from_openqasm(prepared.program)
    assert imported.version == 3.0
    assert imported.measurement_qubits == (0, 1)
    assert {instruction.name for instruction in imported.instructions} == {"h", "cx"}
    emitted_cx_wires = tuple(
        instruction.wires
        for instruction in imported.instructions
        if instruction.name == "cx"
    )
    assert emitted_cx_wires == ((1, 0),)
    options = fq.ExecutionOptions(
        backend="pytorch", device="cpu", mode="statevector", precision="complex128"
    )
    state = fq.run(imported.to_circuit(), options=options).statevector()
    amplitude = 1 / math.sqrt(2)
    expected = torch.tensor([amplitude, 0, 0, amplitude], dtype=torch.complex128)
    torch.testing.assert_close(state.reshape(-1), expected, atol=1e-12, rtol=0)

    with pytest.raises(
        ArtifactCompilationError,
        match="directed-CX legalization exceeds max_added_operations",
    ):
        compile_with_direction_budget(0)


def test_qmod_bell_reaches_verified_offline_preflight() -> None:
    qmod_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures/classiq_qmod/prepare_bell_state.qmod"
    )
    circuit = replace(
        synthesize(qmod_path).circuit,
        measurements=(MeasurementNode("samples", (0, 1)),),
    )
    source = ProgramArtifactV2.from_circuit_ir(circuit, producer="qmod-bell-source")
    snapshot = _snapshot(native_gates=(("h", ()), ("cx", ()), ("x", ())))

    result = compile_circuit_artifact_for_target(
        source,
        backend="qasm",
        profile="openqasm-3.0",
        snapshot=snapshot,
        producer="flagquantum.compiler",
        evaluated_at=_NOW,
        coupling_map=CouplingMap.line(2),
    )
    bundle = build_compilation_evidence_bundle(
        result, snapshot=snapshot, producer="flagquantum.compiler"
    )
    restored = read_compilation_evidence_bundle_json(bundle.to_json())
    assert restored.source["source_artifact_identity"] == source.artifact_identity

    executable = result.executable_artifact
    prepared = prepare_artifact_deployment(
        executable,
        snapshot=snapshot,
        source=source,
        compilation_evidence=restored,
        provider=snapshot.target_identity.provider,
        target_id=snapshot.target_identity.target_id,
        shots=256,
        evaluated_at=_NOW,
    )
    assert prepared.capability_match.executable
    assert prepared.compilation_evidence_identity == restored.bundle_identity
    assert prepared.program == executable.payload

    imported = fq.from_openqasm(prepared.program)
    assert imported.version == 3.0
    assert imported.measurement_qubits == (0, 1)
    options = fq.ExecutionOptions(
        backend="pytorch", device="cpu", mode="statevector", precision="complex128"
    )
    state = fq.run(imported.to_circuit(), options=options).statevector()
    amplitude = 1 / math.sqrt(2)
    expected = torch.tensor([0, amplitude, amplitude, 0], dtype=torch.complex128)
    torch.testing.assert_close(state.reshape(-1), expected, atol=1e-12, rtol=0)
