from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from flagquantum.compiler.target_artifact import (
    TargetArtifactError,
    build_target_artifact,
)
from flagquantum.compiler.target_conformance import verify_target_emission
from flagquantum.compiler.target_emission import emit_legalized_target
from flagquantum.compiler.target_legalization import legalize_circuit_for_target
from flagquantum.core._artifacts import read_program_artifact_json
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
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

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 16, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("target:0",))


def _snapshot(
    *,
    profiles: tuple[str, ...] = ("openqasm-3.0",),
    maximum_shots: int = 4096,
) -> TargetCapabilitySnapshot:
    values = {
        "qubits.logical_capacity": 2,
        "limits.maximum_program_operations": 128,
        "precision.effective_dtype": "complex128",
        "measurements.results": ("samples",),
        "limits.maximum_shots": maximum_shots,
        "artifacts.profiles": profiles,
        "gates.native": (
            {"name": "h", "parameters": ()},
            {"name": "cx", "parameters": ()},
        ),
    }
    source = FactSource(kind="artifact_preflight_test", ref="artifact-evidence")
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="artifact-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="artifact-environment",
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
                evidence_id="artifact-evidence",
                sha256="a" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _compiled_artifact(snapshot: TargetCapabilitySnapshot):
    circuit = CircuitIR(
        2,
        (Instruction("h", (0,)), Instruction("cx", (0, 1))),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0, 1)),),
    )
    legalization = legalize_circuit_for_target(
        circuit,
        backend="qasm",
        snapshot=snapshot,
        evaluated_at=_NOW,
    )
    emission = emit_legalized_target(legalization, profile="openqasm-3.0")
    conformance = verify_target_emission(emission, legalization)
    return build_target_artifact(
        legalization,
        emission,
        conformance,
        producer="flagquantum.compiler",
    )


def test_verified_compilation_forms_artifact_and_passes_runtime_preflight() -> None:
    snapshot = _snapshot()
    artifact = _compiled_artifact(snapshot)

    match = preflight_executable_artifact(
        artifact,
        snapshot=snapshot,
        shots=1024,
        evaluated_at=_NOW,
    )

    assert match.executable is True
    assert artifact.target == {"snapshot_id": snapshot.snapshot_id}
    assert artifact.profile["name"] == "openqasm-3.0"
    assert artifact.compilation is not None
    assert artifact.compilation["conformance_identity"]
    assert read_program_artifact_json(artifact.to_json()) == artifact


@pytest.mark.parametrize(
    ("snapshot", "message"),
    (
        (_snapshot(profiles=("qcis-1.0",)), "artifacts.profiles"),
        (_snapshot(maximum_shots=100), "limits.maximum_shots"),
    ),
)
def test_runtime_preflight_rejects_incompatible_request(
    snapshot: TargetCapabilitySnapshot,
    message: str,
) -> None:
    artifact = _compiled_artifact(snapshot)

    with pytest.raises(ExecutionError, match=message):
        preflight_executable_artifact(
            artifact,
            snapshot=snapshot,
            shots=1024,
            evaluated_at=_NOW,
        )


def test_runtime_preflight_rejects_a_different_target_snapshot() -> None:
    snapshot = _snapshot()
    artifact = _compiled_artifact(snapshot)
    other = replace(
        snapshot,
        target_identity=replace(snapshot.target_identity, target_revision="2"),
        snapshot_id="",
    )

    with pytest.raises(ExecutionError, match="does not match"):
        preflight_executable_artifact(
            artifact,
            snapshot=other,
            shots=1024,
            evaluated_at=_NOW,
        )


def test_compiler_rejects_embedded_shots_before_artifact_construction() -> None:
    snapshot = _snapshot()
    circuit = CircuitIR(
        1,
        (Instruction("h", (0,)),),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0,), shots=100),),
    )
    legalization = legalize_circuit_for_target(
        circuit,
        backend="qasm",
        snapshot=snapshot,
        evaluated_at=_NOW,
    )
    emission = emit_legalized_target(legalization, profile="openqasm-3.0")
    conformance = verify_target_emission(emission, legalization)

    with pytest.raises(TargetArtifactError, match="execution request"):
        build_target_artifact(
            legalization,
            emission,
            conformance,
            producer="flagquantum.compiler",
        )


def test_compiler_rejects_conformance_from_another_evidence_chain() -> None:
    snapshot = _snapshot()
    circuit = CircuitIR(
        1,
        (Instruction("h", (0,)),),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0,)),),
    )
    legalization = legalize_circuit_for_target(
        circuit,
        backend="qasm",
        snapshot=snapshot,
        evaluated_at=_NOW,
    )
    emission = emit_legalized_target(legalization, profile="openqasm-3.0")
    conformance = verify_target_emission(emission, legalization)

    with pytest.raises(TargetArtifactError, match="does not match"):
        build_target_artifact(
            legalization,
            emission,
            replace(conformance, conformance_identity="0" * 64),
            producer="flagquantum.compiler",
        )
