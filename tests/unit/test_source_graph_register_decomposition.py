"""Import the actual source graph while preserving aliases and clean lifetimes."""

from collections import Counter

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._qubit_manager import InteropQubitManager
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
from flagquantum.ecosystem.cirq._source_graph_operations import source_graph_operations
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,modulus", ((2, 1), (2, 2), (3, 3), (3, 5)))
def test_modneg_source_graph_full_residue_channel_and_complex_append(
    width: int, modulus: int
) -> None:
    import cirq
    from qualtran import QUInt
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.mod_arithmetic import ModNeg
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = ModNeg(QUInt(width), modulus)
    quregs = get_named_qubits(source.signature)
    operation, outgoing = source.as_cirq_op(
        cirq.GreedyQubitManager(prefix="source_root"), **quregs
    )
    program = import_cirq_register_optree(
        operation,
        signature=Signature(tuple(native_signature(source.signature))),
        in_quregs=quregs,
        out_quregs=outgoing,
    )
    wires = tuple(reversed(range(width)))
    clean = tuple(range(width, width + 2 * width))
    bound = bind_register_program(program, (wires,), clean=clean)
    assert bound.outputs == (wires,)
    ir = (
        DynamicCircuit(3 * width)
        .append(FieldOperation(3 * width, bound.circuit.instructions))
        .to_ir()
    )
    assert CircuitIR.from_json(ir.to_json()).to_json() == ir.to_json()
    counts = Counter(g.name for g in ir.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["measure"] == model.measurement == 3 * (width - 1)
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert all(not event.dirty for event in bound.events)
    assert (
        Counter(event.kind for event in bound.events)["allocate"]
        == Counter(event.kind for event in bound.events)["free"]
    )

    def basis(value: int) -> int:
        return sum(
            ((value >> (width - index - 1)) & 1) << (3 * width - wire - 1)
            for index, wire in enumerate(wires)
        )

    before = {basis(value): complex(value + 1, 2 - value) for value in range(modulus)}
    norm = sum(abs(value) ** 2 for value in before.values()) ** 0.5
    before = {key: value / norm for key, value in before.items()}
    expected = {
        basis(source.on_classical_vals(x=value)["x"]): before[basis(value)]
        for value in range(modulus)
    }
    bits = tuple(
        tuple((value >> shift) & 1 for shift in reversed(range(3 * width)))
        for value in before
    )
    mapped, measurements = certified_and_channel_map(ir.instructions, bits)
    assert measurements == counts["measure"]
    assert tuple(
        sum(
            bit << shift
            for bit, shift in zip(row, reversed(range(3 * width)), strict=True)
        )
        for row in mapped
    ) == tuple(expected)
    for seed in (0, 1, 17):
        actual = execute_sparse_register(ir, before, seed=seed)
        for basis_value in actual.keys() | expected.keys():
            assert actual.get(basis_value, 0j) == pytest.approx(
                expected.get(basis_value, 0j), abs=2e-12
            )


def test_source_graph_alias_permutation_and_atomic_decline() -> None:
    import cirq
    from qualtran import BloqBuilder, QBit, Register
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import XGate

    builder, words = BloqBuilder.from_signature(
        SourceSignature([Register("left", QBit()), Register("right", QBit())])
    )
    graph = builder.finalize(left=words["right"], right=words["left"])
    qubits = tuple(cirq.LineQubit.range(2))
    from qualtran.cirq_interop import BloqAsCirqGate

    gate = BloqAsCirqGate(graph).on(*qubits)
    manager = InteropQubitManager(cirq.GreedyQubitManager(prefix="views"))
    assert source_graph_operations(XGate().on(qubits[0]), manager) is None
    assert source_graph_operations(cirq.H(qubits[0]), manager) is None
    operations = source_graph_operations(gate, manager)
    assert operations is not None
    assert cirq.Circuit(operations).unitary(qubit_order=qubits) == pytest.approx(
        cirq.unitary(cirq.SWAP)
    )
