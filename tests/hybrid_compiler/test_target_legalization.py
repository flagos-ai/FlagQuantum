from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum.compiler._hybrid import (
    capture_source,
    scalar_type,
    specialize_and_lower,
)
from flagquantum.compiler.routing import CouplingMap
from flagquantum.compiler.target_legalization import (
    TargetLegalizationError,
    circuit_target_requirements,
    legalize_circuit_for_target,
)
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

_NOW = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("target:0",))
_SOURCE = FactSource(kind="phase25_test", ref="phase25-evidence")
_EVIDENCE = EvidenceReference(
    evidence_id="phase25-evidence",
    sha256="a" * 64,
    level=EvidenceLevel.OBSERVABLE,
    scope=_SCOPE,
)


def _snapshot(**changes: object) -> TargetCapabilitySnapshot:
    values: dict[str, object] = {
        "qubits.logical_capacity": 8,
        "limits.maximum_program_operations": 128,
        "precision.effective_dtype": "complex128",
        "measurements.results": ("expectation", "samples"),
        "limits.maximum_shots": 4096,
        "gates.native": (
            {"name": "h", "parameters": ()},
            {"name": "cx", "parameters": ()},
            {"name": "rx", "parameters": ("theta",)},
            {"name": "amplitude_damping", "parameters": ()},
        ),
    }
    values.update(changes)
    facts = tuple(
        CapabilityFact(
            name=name,
            value=value,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=(
                FactExposure.OBSERVED
                if name == "precision.effective_dtype"
                else FactExposure.DECLARED
            ),
            source=_SOURCE,
        )
        for name, value in values.items()
    )
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase25-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase25-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=facts,
        evidence_refs=(_EVIDENCE,),
    )


def _measured_circuit() -> CircuitIR:
    return CircuitIR(
        n_wires=2,
        instructions=(Instruction("h", (0,)), Instruction("cx", (0, 1))),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0, 1), shots=100),),
    )


def test_requirements_are_derived_only_from_circuit_semantics() -> None:
    requirements = circuit_target_requirements(_measured_circuit())
    by_name = {item.name: item for item in requirements.requirements}

    assert by_name["qubits.logical_capacity"].value == 2
    assert by_name["limits.maximum_program_operations"].value == 2
    assert by_name["precision.effective_dtype"].value == "complex128"
    assert by_name["measurements.results"].value == ("samples",)
    assert by_name["limits.maximum_shots"].value == 100
    assert requirements.fallback_authorizations.to_dict() == {
        "backend": False,
        "device": False,
        "cpu": False,
        "precision": False,
        "algorithm": False,
        "approximation": False,
    }


def test_legalization_preserves_circuit_and_is_deterministic() -> None:
    circuit = _measured_circuit()
    snapshot = _snapshot()
    first = legalize_circuit_for_target(
        circuit, backend="pytorch", snapshot=snapshot, evaluated_at=_NOW
    )
    second = legalize_circuit_for_target(
        circuit, backend="pytorch", snapshot=snapshot, evaluated_at=_NOW
    )

    assert first.program is circuit
    assert first.program.content_hash == circuit.content_hash
    assert first.capability_match.executable is True
    assert {item.opcode for item in first.lowering_capabilities} == {"h", "cx"}
    assert first.legalization_identity == second.legalization_identity
    assert len(first.legalization_identity) == 64


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"qubits.logical_capacity": 1}, "qubits.logical_capacity"),
        ({"limits.maximum_program_operations": 1}, "maximum_program_operations"),
        ({"precision.effective_dtype": "complex64"}, "effective_dtype"),
        ({"measurements.results": ("expectation",)}, "measurements.results"),
        ({"limits.maximum_shots": 99}, "maximum_shots"),
    ),
)
def test_target_mismatch_fails_before_emission(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(TargetLegalizationError, match=message):
        legalize_circuit_for_target(
            _measured_circuit(),
            backend="pytorch",
            snapshot=_snapshot(**changes),
            evaluated_at=_NOW,
        )


def test_missing_backend_lowering_fails_before_capability_acceptance() -> None:
    circuit = CircuitIR(
        n_wires=1,
        instructions=(Instruction("amplitude_damping", (0,)),),
        dtype="complex128",
    )
    with pytest.raises(
        TargetLegalizationError, match="cannot lower.*amplitude_damping"
    ):
        legalize_circuit_for_target(
            circuit,
            backend="qasm",
            snapshot=_snapshot(),
            evaluated_at=_NOW,
        )


def test_hybrid_lowering_enters_target_legality_without_another_ir() -> None:
    program = capture_source(
        """
def cost(theta):
    qp.RX(theta, wires=0)
    return qp.expval(qp.PauliZ(0))
""",
        (scalar_type("float64"),),
    )
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    circuit = specialize_and_lower(
        program,
        (theta,),
        circuit_dtype="complex128",
        require_smooth_gradients=True,
    ).bind()
    result = legalize_circuit_for_target(
        circuit,
        backend="pytorch",
        snapshot=_snapshot(),
        evaluated_at=_NOW,
    )

    assert result.program is circuit
    assert type(result.program) is CircuitIR
    assert result.requirements.requirements
    value = result.program.instructions[0].params["theta"]
    assert value is theta


def test_unknown_circuit_precision_fails_closed() -> None:
    circuit = CircuitIR(1, (Instruction("h", (0,)),), dtype="complex256")
    with pytest.raises(TargetLegalizationError, match="complex256"):
        circuit_target_requirements(circuit)


def test_decomposition_operation_count_is_checked_against_target_limit() -> None:
    circuit = CircuitIR(
        1,
        (Instruction("rx", (0,), params={"theta": 0.2}),),
        dtype="complex128",
    )
    native_basis = (
        {"name": "h", "parameters": ()},
        {"name": "rz", "parameters": ("theta",)},
    )
    with pytest.raises(TargetLegalizationError, match="maximum_program_operations"):
        legalize_circuit_for_target(
            circuit,
            backend="pytorch",
            snapshot=_snapshot(
                **{
                    "gates.native": native_basis,
                    "limits.maximum_program_operations": 2,
                }
            ),
            evaluated_at=_NOW,
        )


def test_topology_routing_precedes_native_gate_and_resource_legalization() -> None:
    circuit = CircuitIR(
        3,
        (Instruction("cx", (0, 2)),),
        dtype="complex128",
    )
    result = legalize_circuit_for_target(
        circuit,
        backend="pytorch",
        snapshot=_snapshot(
            **{
                "gates.native": ({"name": "cx", "parameters": ()},),
            }
        ),
        evaluated_at=_NOW,
        coupling_map=CouplingMap.line(3),
        routing_strategy="restore_after_each_gate",
    )

    assert result.topology_legalization is not None
    assert result.topology_legalization.inserted_swap_count == 2
    assert tuple(item.name for item in result.program.instructions) == ("cx",) * 7
    assert tuple(
        item.source_opcode for item in result.native_gate_legalization.decompositions
    ) == ("swap", "swap")
    assert all(
        CouplingMap.line(3).has_edge(*item.wires)
        for item in result.program.instructions
    )
