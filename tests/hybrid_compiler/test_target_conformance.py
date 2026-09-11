from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum import Circuit
from flagquantum.compiler.target_conformance import (
    TargetConformanceError,
    verify_target_emission,
)
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

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("target:0",))


def _snapshot() -> TargetCapabilitySnapshot:
    values = {
        "qubits.logical_capacity": 4,
        "limits.maximum_program_operations": 128,
        "precision.effective_dtype": "complex128",
        "measurements.results": ("samples",),
        "limits.maximum_shots": 4096,
        "gates.native": (
            {"name": "h", "parameters": ()},
            {"name": "cx", "parameters": ()},
            {"name": "rx", "parameters": ("theta",)},
        ),
    }
    source = FactSource(kind="phase30_test", ref="phase30-evidence")
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase30-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase30-environment",
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
                evidence_id="phase30-evidence",
                sha256="c" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _legalized(backend: str):
    circuit = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("rx", (1,), params={"theta": 0.37}),
        ),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0, 1), shots=100),),
    )
    return legalize_circuit_for_target(
        circuit,
        backend=backend,
        snapshot=_snapshot(),
        evaluated_at=_NOW,
    )


def _assert_state_equivalent(left: CircuitIR, right: CircuitIR, atol: float) -> None:
    left_state = Circuit.from_ir(left).state()[0]
    right_state = Circuit.from_ir(right).state()[0]
    overlap = torch.vdot(left_state, right_state).abs()
    torch.testing.assert_close(
        overlap,
        torch.ones((), dtype=overlap.dtype),
        atol=atol,
        rtol=0,
    )


@pytest.mark.parametrize("profile", ("openqasm-2.0", "openqasm-3.0"))
def test_openqasm_round_trip_is_strict_and_statevector_equivalent(profile: str) -> None:
    legalized = _legalized("qasm")
    emitted = emit_legalized_target(legalized, profile=profile)

    first = verify_target_emission(emitted, legalized)
    second = verify_target_emission(emitted, legalized)

    measurement = first.reconstructed_program.measurements[0]
    assert measurement.kind == "samples"
    assert measurement.wires == (0, 1)
    assert measurement.shots is None
    assert first.parsed_operation_count == len(legalized.program.instructions)
    assert first.conformance_identity == second.conformance_identity
    assert len(first.conformance_identity) == 64
    _assert_state_equivalent(legalized.program, first.reconstructed_program, 1e-12)


def test_qcis_native_text_reconstructs_an_equivalent_core_circuit() -> None:
    legalized = _legalized("qcis")
    emitted = emit_legalized_target(legalized, profile="qcis-1.0")
    conformance = verify_target_emission(emitted, legalized)

    assert conformance.parsed_operation_count > len(legalized.program.instructions)
    _assert_state_equivalent(
        legalized.program,
        conformance.reconstructed_program,
        1e-6,
    )


@pytest.mark.parametrize(
    "change",
    (
        {"text": "OPENQASM 3.0;\n"},
        {"payload_sha256": "0" * 64},
        {"target_snapshot_id": "wrong-snapshot"},
        {"emission_identity": "0" * 64},
    ),
)
def test_payload_and_identity_tampering_fail_closed(change: dict[str, str]) -> None:
    legalized = _legalized("qasm")
    emitted = emit_legalized_target(legalized, profile="openqasm-3.0")

    with pytest.raises(TargetConformanceError, match="does not match legalization"):
        verify_target_emission(replace(emitted, **change), legalized)


def test_emission_from_another_legalization_fails_closed() -> None:
    qasm = _legalized("qasm")
    other_program = replace(
        qasm.program,
        instructions=(Instruction("h", (1,)),),
    )
    other = legalize_circuit_for_target(
        other_program,
        backend="qasm",
        snapshot=_snapshot(),
        evaluated_at=_NOW,
    )
    emitted = emit_legalized_target(qasm, profile="openqasm-3.0")

    with pytest.raises(TargetConformanceError, match="does not match legalization"):
        verify_target_emission(emitted, other)
