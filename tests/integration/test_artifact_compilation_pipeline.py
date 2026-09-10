from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from flagquantum.compiler.artifact_compilation import (
    ArtifactCompilationError,
    ArtifactCompilationResult,
    compile_circuit_artifact_for_target,
)
from flagquantum.compiler.compilation_evidence import (
    CompilationEvidenceError,
    build_compilation_evidence_bundle,
    verify_compilation_evidence_bundle,
)
from flagquantum.compiler.routing import CouplingMap
from flagquantum.core._artifacts import (
    ProgramArtifactV2,
    bind_circuit_artifact,
    read_program_artifact_json,
)
from flagquantum.core._compilation_evidence import (
    read_compilation_evidence_bundle_json,
)
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
from flagquantum.deployment.artifact_dry_run import prepare_artifact_deployment
from flagquantum.runtime.artifact_preflight import preflight_executable_artifact

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 20, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("target:0",))


def _snapshot() -> TargetCapabilitySnapshot:
    source = FactSource(kind="artifact_compilation_test", ref="target-evidence")
    values = {
        "qubits.logical_capacity": 3,
        "limits.maximum_program_operations": 128,
        "precision.effective_dtype": "complex128",
        "measurements.results": ("samples",),
        "limits.maximum_shots": 4096,
        "artifacts.profiles": ("openqasm-3.0",),
        "gates.native": (
            {"name": "h", "parameters": ()},
            {"name": "cx", "parameters": ()},
            {"name": "swap", "parameters": ()},
            {"name": "rx", "parameters": ("theta",)},
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
    binding = bind_circuit_artifact(
        symbolic,
        {"theta": 0.25},
        producer="phase35-binding",
    )
    result = _compile(binding)

    assert result.source_artifact_identity == symbolic.artifact_identity
    assert result.circuit_artifact_identity == binding.bound_artifact.artifact_identity
    assert result.binding_identity == binding.binding_identity
    assert result.executable_artifact.circuit_content_hash == (
        result.legalization.program.content_hash
    )


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
