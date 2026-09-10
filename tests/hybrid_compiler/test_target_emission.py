from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flagquantum.compiler.target_emission import (
    TargetEmissionError,
    emit_legalized_target,
)
from flagquantum.compiler.target_legalization import legalize_circuit_for_target
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
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

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("target:0",))


def _snapshot() -> TargetCapabilitySnapshot:
    facts = (
        CapabilityFact(
            name="qubits.logical_capacity",
            value=4,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=FactSource(kind="phase29_test", ref="phase29-evidence"),
        ),
        CapabilityFact(
            name="limits.maximum_program_operations",
            value=128,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=FactSource(kind="phase29_test", ref="phase29-evidence"),
        ),
        CapabilityFact(
            name="precision.effective_dtype",
            value="complex128",
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.OBSERVED,
            source=FactSource(kind="phase29_test", ref="phase29-evidence"),
        ),
        CapabilityFact(
            name="measurements.results",
            value=("expectation", "samples"),
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=FactSource(kind="phase29_test", ref="phase29-evidence"),
        ),
        CapabilityFact(
            name="limits.maximum_shots",
            value=4096,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=FactSource(kind="phase29_test", ref="phase29-evidence"),
        ),
        CapabilityFact(
            name="gates.native",
            value=(
                {"name": "h", "parameters": ()},
                {"name": "cx", "parameters": ()},
                {"name": "rx", "parameters": ("theta",)},
            ),
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=FactSource(kind="phase29_test", ref="phase29-evidence"),
        ),
    )
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase29-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase29-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=facts,
        evidence_refs=(
            EvidenceReference(
                evidence_id="phase29-evidence",
                sha256="b" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _circuit(*instructions: Instruction) -> CircuitIR:
    return CircuitIR(
        2,
        instructions,
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0, 1), shots=100),),
    )


def _legalize(circuit: CircuitIR, backend: str = "qasm"):
    return legalize_circuit_for_target(
        circuit,
        backend=backend,
        snapshot=_snapshot(),
        evaluated_at=_NOW,
    )


def test_openqasm_emission_is_deterministic_and_binds_all_identities() -> None:
    legalized = _legalize(_circuit(Instruction("h", (0,)), Instruction("cx", (0, 1))))

    first = emit_legalized_target(legalized, profile="openqasm-3.0")
    second = emit_legalized_target(legalized, profile="openqasm-3.0")

    assert first == second
    assert first.text.startswith("OPENQASM 3.0;")
    assert first.source_circuit_hash == legalized.program.content_hash
    assert first.target_snapshot_id == legalized.target_snapshot_id
    assert first.target_legalization_identity == legalized.legalization_identity
    assert first.schedule_identity == legalized.schedule.schedule_identity
    assert len(first.payload_sha256) == len(first.emission_identity) == 64


def test_profile_version_and_backend_are_part_of_emission_legality() -> None:
    legalized = _legalize(_circuit(Instruction("h", (0,))))
    qasm2 = emit_legalized_target(legalized, profile="openqasm-2.0")
    qasm3 = emit_legalized_target(legalized, profile="openqasm-3.0")

    assert qasm2.payload_sha256 != qasm3.payload_sha256
    assert qasm2.emission_identity != qasm3.emission_identity
    with pytest.raises(TargetEmissionError, match="requires legalized backend 'qcis'"):
        emit_legalized_target(legalized, profile="qcis-1.0")
    with pytest.raises(
        TargetEmissionError, match="unsupported target emission profile"
    ):
        emit_legalized_target(legalized, profile="openqasm-4.0")


def test_qcis_reuses_existing_deterministic_emitter() -> None:
    legalized = _legalize(_circuit(Instruction("h", (0,))), backend="qcis")
    emitted = emit_legalized_target(legalized, profile="qcis-1.0")

    assert emitted.text == "Y2M Q0\nRZ Q0 3.141593"
    assert emitted.media_type.startswith("text/x-qcis")


def test_static_profiles_reject_semantics_the_emitters_cannot_preserve() -> None:
    expectation = CircuitIR(
        2,
        (Instruction("h", (0,)),),
        dtype="complex128",
        observables=(ObservableNode("z", (0,)),),
    )
    partial = CircuitIR(
        2,
        (Instruction("h", (0,)),),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0,), shots=100),),
    )

    with pytest.raises(TargetEmissionError, match="expectation semantics"):
        emit_legalized_target(_legalize(expectation), profile="openqasm-3.0")
    with pytest.raises(TargetEmissionError, match="full-register samples"):
        emit_legalized_target(_legalize(partial), profile="openqasm-3.0")

    dynamic = _circuit(Instruction("h", (0,), metadata={"is_dynamic": True}))
    with pytest.raises(TargetEmissionError, match="cannot be represented"):
        emit_legalized_target(_legalize(dynamic), profile="openqasm-3.0")


def test_unbound_parameters_fail_closed_at_emission() -> None:
    legalized = _legalize(
        _circuit(Instruction("rx", (0,), params={"theta": Parameter("theta")}))
    )

    with pytest.raises(TargetEmissionError, match="Unbound circuit parameter"):
        emit_legalized_target(legalized, profile="openqasm-3.0")


def test_mutation_after_legalization_invalidates_schedule_evidence() -> None:
    legalized = _legalize(_circuit(Instruction("h", (0,))))
    legalized.program.instructions[0].metadata["changed"] = True

    with pytest.raises(TargetEmissionError, match="no longer matches"):
        emit_legalized_target(legalized, profile="openqasm-3.0")
