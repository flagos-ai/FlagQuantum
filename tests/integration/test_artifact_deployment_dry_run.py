from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flagquantum.compiler.target_artifact import build_target_artifact
from flagquantum.compiler.target_conformance import verify_target_emission
from flagquantum.compiler.target_emission import emit_legalized_target
from flagquantum.compiler.target_legalization import legalize_circuit_for_target
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
from flagquantum.deployment.artifact_dry_run import prepare_artifact_deployment
from flagquantum.errors import ExecutionError

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)


def _snapshot(*, maximum_shots: int = 4096) -> TargetCapabilitySnapshot:
    scope = CapabilityScope(device_ids=("qpu:0",))
    source = FactSource(kind="deployment_dry_run_test", ref="target-evidence")
    values = {
        "qubits.logical_capacity": 2,
        "limits.maximum_program_operations": 128,
        "precision.effective_dtype": "complex128",
        "measurements.results": ("samples",),
        "limits.maximum_shots": maximum_shots,
        "artifacts.profiles": ("openqasm-3.0",),
        "gates.native": (
            {"name": "h", "parameters": ()},
            {"name": "cx", "parameters": ()},
        ),
    }
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="ScQ-P10",
            target_class="qpu",
            provider="quafu",
            provider_version="1",
            target_revision="calibration-1",
            environment_id="quafu-production",
        ),
        scope=scope,
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
                sha256="b" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=scope,
            ),
        ),
    )


def _artifact(snapshot: TargetCapabilitySnapshot):
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


def test_dry_run_prepares_exact_verified_program_without_submission() -> None:
    snapshot = _snapshot()
    artifact = _artifact(snapshot)

    prepared = prepare_artifact_deployment(
        artifact,
        snapshot=snapshot,
        provider="quafu",
        target_id="ScQ-P10",
        shots=1024,
        evaluated_at=_NOW,
    )

    assert prepared.capability_match.executable
    assert prepared.program == artifact.payload
    assert prepared.profile == "openqasm-3.0"
    assert prepared.artifact_identity == artifact.artifact_identity
    assert prepared.snapshot_id == snapshot.snapshot_id
    assert prepared.shots == 1024


@pytest.mark.parametrize(
    ("provider", "target_id", "message"),
    (
        ("other", "ScQ-P10", "provider"),
        ("quafu", "other", "target"),
    ),
)
def test_dry_run_rejects_locator_snapshot_mismatch(
    provider: str,
    target_id: str,
    message: str,
) -> None:
    snapshot = _snapshot()

    with pytest.raises(ExecutionError, match=message):
        prepare_artifact_deployment(
            _artifact(snapshot),
            snapshot=snapshot,
            provider=provider,
            target_id=target_id,
            shots=1024,
            evaluated_at=_NOW,
        )


def test_dry_run_rejects_request_that_exceeds_target_shots() -> None:
    snapshot = _snapshot(maximum_shots=100)

    with pytest.raises(ExecutionError, match="limits.maximum_shots"):
        prepare_artifact_deployment(
            _artifact(snapshot),
            snapshot=snapshot,
            provider="quafu",
            target_id="ScQ-P10",
            shots=1024,
            evaluated_at=_NOW,
        )


def test_execution_options_do_not_change_program_artifact_identity() -> None:
    snapshot = _snapshot()
    artifact = _artifact(snapshot)

    first = prepare_artifact_deployment(
        artifact,
        snapshot=snapshot,
        provider="quafu",
        target_id="ScQ-P10",
        shots=100,
        evaluated_at=_NOW,
    )
    second = prepare_artifact_deployment(
        artifact,
        snapshot=snapshot,
        provider="quafu",
        target_id="ScQ-P10",
        shots=200,
        evaluated_at=_NOW,
    )

    assert first.artifact_identity == second.artifact_identity
    assert first.program == second.program
    assert first.shots != second.shots
    assert not hasattr(first, "credentials")
    assert not hasattr(first, "task_id")
