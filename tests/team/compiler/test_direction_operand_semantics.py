"""Operand semantics of native two-wire instructions on an ordered topology."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum.compiler.directed_topology import DirectedCouplingMap
from flagquantum.compiler.direction_legalization import (
    DirectionLegalizationError,
    _validate_direction_legal_program,
    legalize_directed_cx,
)
from flagquantum.compiler.target_legalization import legalize_circuit_for_target
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
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
from flagquantum.simulation.statevector.local import run_local_statevector

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 10, 19, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("direction-operands:0",))
_SOURCE = FactSource(
    kind="direction_operand_semantics", ref="direction-operands-evidence"
)

# The authoritative set of two-wire opcodes, read from Core rather than
# restated here, so a new Core opcode is exercised by these tests on the run
# that adds it.
_TWO_WIRE_OPCODES = tuple(
    sorted(name for name, schema in OPERATOR_SCHEMAS.items() if schema.arity == 2)
)
# A native set wide enough to reverse CX and to carry every two-wire opcode
# this module classifies.
_NATIVE = ("h", "ry", *_TWO_WIRE_OPCODES)


def _snapshot(native_gates: tuple[str, ...] = _NATIVE) -> TargetCapabilitySnapshot:
    values: dict[str, object] = {
        "qubits.logical_capacity": 8,
        "limits.maximum_program_operations": 4096,
        "precision.effective_dtype": "complex128",
        "gates.native": native_gates,
        "artifacts.profiles": ("openqasm-3.0",),
        "measurements.results": ("samples",),
        "limits.maximum_shots": 4096,
    }
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="direction-operands-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="direction-operands-environment",
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
                source=_SOURCE,
            )
            for name, value in values.items()
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="direction-operands-evidence",
                sha256="e" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _state(program: CircuitIR) -> torch.Tensor:
    return run_local_statevector(
        program,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )


def _operands_commute(opcode: str) -> bool:
    """Report whether the opcode's unitary ignores operand order, by measurement."""

    def program(wires: tuple[int, int]) -> CircuitIR:
        return CircuitIR(
            n_wires=2,
            dtype="complex128",
            instructions=(
                Instruction("ry", (0,), params={"theta": 0.71}),
                Instruction("ry", (1,), params={"theta": -0.43}),
                Instruction(
                    opcode,
                    wires,
                    params=dict.fromkeys(OPERATOR_SCHEMAS[opcode].parameters, 0.31),
                ),
            ),
        )

    difference = torch.max(torch.abs(_state(program((0, 1))) - _state(program((1, 0)))))
    return bool(float(difference) < 1e-12)


def _prepared_program(opcode: str, wires: tuple[int, int] = (0, 1)) -> CircuitIR:
    """A non-invariant three-wire input state followed by the two-wire opcode."""

    return CircuitIR(
        n_wires=3,
        dtype="complex128",
        instructions=(
            Instruction("ry", (0,), params={"theta": 0.71}),
            Instruction("ry", (1,), params={"theta": -0.43}),
            Instruction("ry", (2,), params={"theta": 1.19}),
            Instruction(
                opcode,
                wires,
                params=dict.fromkeys(OPERATOR_SCHEMAS[opcode].parameters, 0.31),
            ),
        ),
    )


def _legalize(program: CircuitIR, graph: DirectedCouplingMap):
    return legalize_directed_cx(
        program,
        coupling_map=graph,
        snapshot=_snapshot(),
        native_opcodes=_NATIVE,
    )


def test_every_two_wire_opcode_is_classified() -> None:
    """No Core two-wire opcode may become unusable on a directed device."""

    graph = DirectedCouplingMap(3, ((0, 1),))
    refused: dict[str, str] = {}
    for opcode in _TWO_WIRE_OPCODES:
        try:
            _legalize(_prepared_program(opcode), graph)
        except DirectionLegalizationError as error:
            refused[opcode] = str(error)

    assert _TWO_WIRE_OPCODES  # the schema lookup must not be vacuous
    assert refused == {}


def test_classification_matches_the_operand_algebra() -> None:
    """A gate is direction-free exactly when its unitary ignores operand order."""

    reverse_only = DirectedCouplingMap(3, ((1, 0),))
    disagreements: dict[str, bool] = {}
    for opcode in _TWO_WIRE_OPCODES:
        try:
            result = _legalize(_prepared_program(opcode), reverse_only)
            direction_free = result.reversed_cx_count == 0
        except DirectionLegalizationError:
            direction_free = False
        if direction_free != _operands_commute(opcode):
            disagreements[opcode] = direction_free

    assert _TWO_WIRE_OPCODES
    assert disagreements == {}


def test_unlinked_two_wire_opcode_is_refused() -> None:
    """Operand symmetry is not a licence to ignore physical connectivity."""

    detached = DirectedCouplingMap(3, ())
    refused = 0
    for opcode in _TWO_WIRE_OPCODES:
        with pytest.raises(
            DirectionLegalizationError,
            match="no physical link|requires the physical edge|unavailable",
        ):
            _legalize(_prepared_program(opcode, (0, 2)), detached)
        refused += 1

    assert refused == len(_TWO_WIRE_OPCODES)


def test_unclassified_two_wire_opcode_is_refused() -> None:
    """An opcode with no recorded operand semantics must not guess a rule."""

    program = CircuitIR(
        n_wires=3,
        dtype="complex128",
        instructions=(
            Instruction("unitary", (0, 1), matrix=torch.eye(4, dtype=torch.complex128)),
        ),
    )

    with pytest.raises(
        DirectionLegalizationError, match="does not define operand semantics"
    ):
        legalize_directed_cx(
            program,
            coupling_map=DirectedCouplingMap(3, ((0, 1),)),
            snapshot=_snapshot(("unitary",)),
            native_opcodes=("unitary",),
        )


def test_commuting_opcode_compiles_on_a_single_direction_device() -> None:
    """A ``cz``-native device that offers only 1->0 must still compile CZ and RZZ."""

    graph = DirectedCouplingMap(2, ((1, 0),))
    source = CircuitIR(
        n_wires=2,
        dtype="complex128",
        instructions=(
            Instruction("ry", (0,), params={"theta": 0.37}),
            Instruction("h", (1,)),
            Instruction("cz", (0, 1)),
            Instruction("rzz", (0, 1), params={"theta": 0.4}),
        ),
    )
    result = legalize_circuit_for_target(
        source,
        backend="qasm",
        snapshot=_snapshot(),
        evaluated_at=_NOW,
        coupling_map=graph,
    )

    assert result.direction_legalization is not None
    assert result.direction_legalization.reversed_cx_count == 0
    assert tuple(item.name for item in result.program.instructions) == (
        "ry",
        "h",
        "cz",
        "rzz",
    )
    assert not graph.has_edge(0, 1)  # the emitted order is legal only by symmetry
    torch.testing.assert_close(_state(result.program), _state(source))


def test_ordered_opcode_requires_the_declared_direction() -> None:
    """CY keeps control/target meaning, so only the declared edge satisfies it."""

    source = _prepared_program("cy")
    accepted = _legalize(source, DirectedCouplingMap(3, ((0, 1),)))

    assert accepted.reversed_cx_count == 0
    assert accepted.program.instructions[-1].name == "cy"
    torch.testing.assert_close(_state(accepted.program), _state(source))

    with pytest.raises(DirectionLegalizationError, match="only CX operand reversal"):
        _legalize(source, DirectedCouplingMap(3, ((1, 0),)))


def test_direction_validator_refuses_an_unlinked_commuting_opcode() -> None:
    """The guard behind the pass must apply connectivity to symmetric opcodes."""

    graph = DirectedCouplingMap(3, ((0, 1),))
    native = frozenset(_NATIVE)
    linked = CircuitIR(3, (Instruction("cz", (0, 1)),), dtype="complex128")
    reverse_linked = CircuitIR(3, (Instruction("cz", (1, 0)),), dtype="complex128")
    ordered_violation = CircuitIR(3, (Instruction("cy", (1, 0)),), dtype="complex128")
    unlinked = CircuitIR(3, (Instruction("cz", (0, 2)),), dtype="complex128")

    _validate_direction_legal_program(linked, native, graph)
    _validate_direction_legal_program(reverse_linked, native, graph)
    with pytest.raises(DirectionLegalizationError, match="violates ordered topology"):
        _validate_direction_legal_program(ordered_violation, native, graph)
    with pytest.raises(DirectionLegalizationError, match="violates topology"):
        _validate_direction_legal_program(unlinked, native, graph)
