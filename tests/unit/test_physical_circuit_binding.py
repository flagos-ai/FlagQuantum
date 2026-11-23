"""Close complete physical circuit bindings through existing native owners."""

import json
from copy import copy, deepcopy
from pathlib import Path

import pytest

from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.artifacts.reaction_depth import reaction_depth_for_operations
from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from flagquantum.compiler.formats._physical_operation_statistics import (
    PhysicalOperationStatistics,
)
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_circuit import (
    KmxCircuit,
    concatenate_kmx,
    read_kmx_circuit,
    repeat_kmx,
)
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.core.ir import Instruction
from flagquantum.drawer.reversible_phase_diagram import render_circuit_text
from flagquantum.ecosystem._physical_basis_session import PhysicalBasisSession

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "physical_circuit_binding.json").read_text()
)
_CIRCUITS = tuple(read_kmx_circuit(row["text"]) for row in _DATA["circuits"])


def observe(circuit: KmxCircuit) -> dict[str, object]:
    simulator = PhysicalBasisSession(
        3, seed=0, debug_print=False, counting_operations=True
    )
    for wire in range(circuit.qubits):
        simulator.write_across_shots(wire, (wire * 5 + 3) & 7)
    simulator.do(circuit)
    return {
        "text": circuit.to_kmx(),
        "records": len(circuit),
        "qubits": circuit.qubits,
        "bits": circuit.bits,
        "registers": len(circuit.registers),
        "register_data": [
            [
                (
                    ["index", key.value]
                    if isinstance(key, OperandIdentity)
                    else ["name", key]
                ),
                [value.diagnostic_text for value in values],
            ]
            for key, values in circuit.register_data.items()
        ],
        "reaction_depth": reaction_depth_for_operations(
            ReversiblePhaseOperation(
                record.opcode.lower(),
                record.qubits,
                record.result_bit,
                record.condition_bit,
                record.angle,
            )
            for record in circuit
        ),
        "text_diagram": render_circuit_text(circuit),
        "static_counts": dict(PhysicalOperationStatistics.from_circuit(circuit)),
        "executed_counts": dict(simulator.operation_statistics()),
        "quantum": [
            simulator.read_across_shots(wire, out=int) for wire in range(circuit.qubits)
        ],
        "classical": [
            simulator.read_across_shots(
                OperandIdentity(OperandKind.CLASSICAL, bit), out=int
            )
            for bit in range(circuit.bits)
        ],
        "phases": [str(simulator.read_shot_phase(shot)) for shot in range(3)],
    }


@pytest.mark.parametrize("index", range(len(_CIRCUITS)))
def test_complete_value_projections_and_copy(index: int) -> None:
    circuit = _CIRCUITS[index]
    assert observe(circuit) == _DATA["circuits"][index]
    assert observe(copy(circuit)) == observe(deepcopy(circuit)) == observe(circuit)
    assert [circuit == other for other in _CIRCUITS] == _DATA["equalities"][index]
    assert [circuit != other for other in _CIRCUITS] == [
        not equal for equal in _DATA["equalities"][index]
    ]
    registers = circuit.register_data
    registers.clear()
    assert observe(circuit) == _DATA["circuits"][index]


@pytest.mark.parametrize("left,right,expected", _DATA["additions"])
def test_complete_source_concatenation(
    left: int, right: int, expected: dict[str, object]
) -> None:
    first, second = _CIRCUITS[left], _CIRCUITS[right]
    combined = first + second
    assert observe(combined) == expected
    assert concatenate_kmx(first.to_kmx(), second.to_kmx()) == combined.to_kmx()
    assert combined.registers is (first.registers or second.registers)
    assert observe(first) == _DATA["circuits"][left]
    assert observe(second) == _DATA["circuits"][right]


@pytest.mark.parametrize("index,count,expected", _DATA["repetitions"])
def test_complete_source_signed_repetition(
    index: int, count: int, expected: dict[str, object]
) -> None:
    circuit = _CIRCUITS[index]
    repeated = circuit * count
    assert observe(repeated) == observe(count * circuit) == expected
    assert repeated.registers is circuit.registers
    assert repeat_kmx(circuit.to_kmx(), max(0, count)) == repeated.to_kmx()
    assert observe(circuit) == _DATA["circuits"][index]


def test_object_integer_protocol_and_memory_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.formats import kmx_circuit as owner

    class Two:
        def __index__(self) -> int:
            return 2

    circuit = read_kmx_circuit('REGISTER r0 "kept"\nX q0')
    assert circuit * Two() == Two() * circuit == circuit + circuit
    for value in (-(1 << 63) - 1, 1 << 63, 1.5, "2", None):
        with pytest.raises(TypeError):
            circuit * value
        with pytest.raises(TypeError):
            value * circuit
    assert circuit.__add__(object()) is NotImplemented
    assert circuit.__mul__(object()) is NotImplemented
    assert circuit.__rmul__(object()) is NotImplemented
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        repeat_kmx("X q0", -1)
    empty = read_kmx_circuit('REGISTER r0 "kept"')
    assert empty * ((1 << 63) - 1) == empty
    assert repeat_kmx(empty.to_kmx(), (1 << 64) - 1) == empty.to_kmx()
    monkeypatch.setattr(owner, "_MAX_RECORDS", 1)
    for count in (2, (1 << 63) - 1):
        with pytest.raises(ValueError, match="record limit"):
            circuit * count
    with pytest.raises(ValueError, match="record limit"):
        circuit + circuit
    assert circuit * 1 == circuit
    assert len(circuit * 0) == 0


def test_value_composition_reuses_normal_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.algorithms import unary_address_traversal as consumer

    monkeypatch.setattr(
        consumer, "bind_reversible_phase_execution", _composed_instructions
    )
    consumer.main()


def _composed_instructions(program: ReversiblePhaseProgram) -> tuple[Instruction, ...]:
    circuit = read_kmx_circuit(encode_kmx(program))
    cut = len(circuit) // 2
    first = KmxCircuit(circuit.registers, circuit.records[:cut])
    second = KmxCircuit((), circuit.records[cut:])
    restored = decode_kmx(((first + second) * 1 + circuit * -1).to_kmx())
    assert restored == program
    return bind_reversible_phase_execution(restored)


@pytest.mark.parametrize("n", (0, 1, 2))
@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("park", (False, True))
def test_composition_preserves_all_small_complex_branches(
    n: int, controlled: bool, park: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit import test_kickmix_unary_iteration as accepted

    monkeypatch.setattr(
        accepted, "bind_reversible_phase_execution", _composed_instructions
    )
    accepted.test_all_reset_and_erasure_branches_preserve_complex_amplitudes(
        n, controlled, park
    )
