"""Concrete partitions preserve source port order without emitting gates."""

from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
)
from flagquantum.compiler.registers.layout.partition import (
    join_integer_pair,
    join_register_pair,
    partition_register,
    partition_width,
    split_integer_pair,
    split_register_pair,
    unpartition_register,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_original_notebook_shaped_partition_and_join_keep_port_order() -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import QAny, Register
    from qualtran.bloqs.bookkeeping.partition import Partition
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.layout.component_partition import (
        partition_component_program,
        partition_signature,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register as NativeRegister

    original_ports = (Register("xx", QAny(2), shape=(2, 3)), Register("yy", QAny(37)))
    ports = (
        NativeRegister("xx", opaque_register_type(2), shape=(2, 3)),
        NativeRegister("yy", opaque_register_type(37)),
    )
    source = Partition(49, original_ports)
    wires = tuple(reversed(range(49)))
    manager = cirq.SimpleQubitManager()
    operation, source_parts = source.as_cirq_op(
        manager, x=np.array([cirq.LineQubit(wire) for wire in wires])
    )
    assert operation is None
    parts = tuple(
        tuple(q.x for q in array[index].flat)
        for name, array in source_parts.items()
        for index in source.signature.get_right(name).all_idxs()
    )

    for reverse, locked, inputs, outputs in (
        (False, source, (wires,), parts),
        (True, source.adjoint(), parts, (wires,)),
    ):
        metadata = partition_signature(ports, reverse=reverse)
        assert metadata.n_qubits() == get_cost_value(locked, QubitCount()) == 49
        assert get_cost_value(locked, QECGatesCost()).asdict() == {}
        assert tuple(
            (port.name, port.shape, port.bitsize, port.side.value) for port in metadata
        ) == tuple(
            (port.name, port.shape, port.bitsize, port.side.value)
            for port in locked.signature
        )
        program = partition_component_program(ports, "main", reverse)
        result = bind_register_program(program, inputs)
        assert result.outputs == outputs
        assert result.events == ()
        assert result.circuit.instructions == ()


@pytest.mark.ftqc_reference
def test_original_twelve_wire_wrapper_preserves_the_complex_tensor_map() -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import BloqBuilder, Signature
    from qualtran.bloqs.bookkeeping import Partition
    from qualtran.bloqs.for_testing import TestMultiRegister
    from qualtran.simulation.tensor import cbloq_to_quimb
    from quimb.tensor import Tensor

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.layout.component_partition import (
        partition_component_program,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir import Instruction
    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register

    child = TestMultiRegister()
    partition = Partition(12, tuple(child.signature))
    builder, inputs = BloqBuilder.from_signature(Signature.build(test_regs=12))
    child_inputs = builder.add_d(partition, x=inputs["test_regs"])
    child_outputs = builder.add_d(child, **child_inputs)
    composite = builder.finalize(
        test_regs=builder.add(partition.adjoint(), **child_outputs)
    )
    network = cbloq_to_quimb(composite.flatten(), friendly_indices=True)
    assert network.shape == (2,) * 24
    assert all(tensor.ndim <= 4 for tensor in network.rank_simplify().tensors)
    wires = tuple(range(12))
    external_wires = cirq.NamedQubit.range(12, prefix="system")
    foreign = composite.to_cirq_circuit(
        qubit_manager=cirq.SimpleQubitManager(),
        cirq_quregs={"test_regs": external_wires},
    )
    (foreign_child,) = foreign.all_operations()
    assert foreign_child.qubits == tuple(external_wires)

    registers = (
        Register("xx", opaque_register_type(1)),
        Register("yy", opaque_register_type(2), shape=(2, 2)),
        Register("zz", opaque_register_type(3)),
    )
    split = bind_register_program(
        partition_component_program(registers, "main", False), (wires,)
    )
    join = bind_register_program(
        partition_component_program(registers, "main", True), split.outputs
    )
    assert join.outputs == (wires,)
    assert split.circuit.instructions == join.circuit.instructions == ()
    first, *pairs, spectator = split.outputs
    assert (first, pairs, spectator) == (
        (0,),
        [(1, 2), (3, 4), (5, 6), (7, 8)],
        (9, 10, 11),
    )
    # The source testing atoms define X and CNOT tensors, not gate decompositions.
    instructions = (
        Instruction("x", first),
        *(Instruction("cx", pair) for pair in pairs),
    )
    values = torch.arange(1, 4097, dtype=torch.float64)
    initial = torch.complex(values, (values % 13) - 6)
    initial /= initial.norm()
    left = tuple(f"test_regs_{i}l" for i in wires)
    right = tuple(f"test_regs_{i}r" for i in wires)
    assert set(network.outer_inds()) == set(left + right)
    ket = Tensor(initial.numpy().reshape((2,) * 12), inds=left)
    expected = torch.as_tensor((network & ket).to_dense(right).reshape(-1))
    circuit = fq.Circuit(12, inputs=initial, dtype=initial.dtype)
    circuit.append(FieldOperation(12, instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)


@pytest.mark.ftqc_reference
def test_source_partition_shaped_ports_and_join() -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import QAny, QBit, Register
    from qualtran.bloqs.bookkeeping.partition import Partition

    order = (9, 4, 5, 3, 0)
    ports = (Register("cells", QBit(), shape=(2, 2)), Register("flag", QAny(1)))
    source = Partition(5, ports)
    manager = cirq.SimpleQubitManager()
    qubits = np.array([cirq.LineQubit(q) for q in order])
    op, reference = source.as_cirq_op(manager, x=qubits)
    assert op is None
    assert partition_register(order, (("cells", 4), ("flag", 1))) == {
        name: tuple(q.x for q in array.flat) for name, array in reference.items()
    }
    op, joined = source.adjoint().as_cirq_op(manager, **reference)
    assert op is None and tuple(q.x for q in joined["x"]) == order


@pytest.mark.parametrize(
    "packed,ports,message",
    [
        ((0,), (("a", 1), ("a", 0)), "distinct nonempty names"),
        ((0,), (("a", -1),), "nonnegative integer widths"),
        ((0,), (("a", 2),), "sum of partition widths"),
        ((0, 0), (("a", 2),), "distinct nonnegative wires"),
    ],
)
def test_partition_rejects_bad_layout(
    packed: tuple[int, ...], ports: tuple[tuple[str, int], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        partition_register(packed, ports)


def test_zero_width_named_partition() -> None:
    assert partition_register((), (("empty", 0),)) == {"empty": ()}


def test_unpartition_checks_names_widths_and_aliases() -> None:
    layout = (("high", 1), ("low", 1))
    with pytest.raises(ValueError, match="port names"):
        unpartition_register({"high": (2,), "typo": (3,)}, layout)
    with pytest.raises(ValueError, match="port widths"):
        unpartition_register({"high": (2, 3), "low": ()}, layout)
    with pytest.raises(ValueError, match="distinct"):
        unpartition_register({"high": (2,), "low": (2,)}, layout)
    assert unpartition_register({"low": (3,), "high": (2,)}, layout) == (2, 3)
    assert unpartition_register({"empty": ()}, (("empty", 0),)) == ()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("widths", [(0, 0), (0, 3), (3, 0), (2, 3), (5, 2)])
def test_split2_join2_match_source_physical_bindings_and_resources(
    widths: tuple[int, int],
) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.bookkeeping import Join2, Split2
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    order = tuple(reversed(range(sum(widths))))
    source = Split2(*widths)
    assert source.adjoint() == Join2(*widths)
    assert source.adjoint().adjoint() == source
    manager = cirq.SimpleQubitManager()
    op, reference = source.as_cirq_op(
        manager, x=np.array([cirq.LineQubit(q) for q in order], dtype=object)
    )
    assert op is None
    words = split_register_pair(order, *widths)
    assert words == tuple(tuple(q.x for q in port.flat) for port in reference.values())
    op, joined = source.adjoint().as_cirq_op(manager, **reference)
    assert op is None
    assert join_register_pair(*words) == tuple(q.x for q in joined["x"]) == order
    layout = (("y1", widths[0]), ("y2", widths[1]))
    opcodes: tuple[Literal["split2", "join2"], ...] = ("split2", "join2")
    for opcode, leaf in zip(opcodes, (source, source.adjoint()), strict=True):
        assert partition_width(layout) == get_cost_value(leaf, QubitCount())
        assert get_cost_value(leaf, QECGatesCost()).asdict() == {}
        assert query_qec_gate_cost(BookkeepingCall(opcode, widths)) == QECGateCost()


@pytest.mark.ftqc_reference
def test_split2_join2_symbolic_width_and_integer_boundaries() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.bookkeeping import Join2, Split2
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    a, b = Symbol("a", integer=True), Symbol("b", integer=True)
    width = partition_width((("y1", a), ("y2", b)))
    for source in (Split2(a, b), Join2(a, b)):
        assert width == get_cost_value(source, QubitCount()) == a + b
        assert get_cost_value(source, QECGatesCost()).asdict() == {}
    with pytest.raises(ValueError, match="concrete"):
        split_register_pair((), a, b)

    for value in range(32):
        words = split_integer_pair(value, 2, 3)
        assert words == Split2(2, 3).call_classically(x=value)
        assert (
            (join_integer_pair(*words, 2, 3),)
            == Join2(2, 3).call_classically(y1=words[0], y2=words[1])
            == (value,)
        )
    for widths in ((0, 3), (3, 0), (0, 0)):
        with pytest.raises(ValueError, match="invalid literal"):
            Split2(*widths).call_classically(x=0)
        with pytest.raises(ValueError, match="positive"):
            split_integer_pair(0, *widths)
    with pytest.raises(ValueError, match="nonempty"):
        join_integer_pair(0, 0, 0, 0)
    for a_width, b_width, left, right in (
        (0, 3, 1, 5),
        (3, 0, 3, 1),
        (2, 3, 4, 9),
        (65, 2, 1 << 64, 3),
    ):
        assert (join_integer_pair(left, right, a_width, b_width),) == Join2(
            a_width, b_width
        ).call_classically(y1=left, y2=right)
    with pytest.raises(ValueError):
        Join2(64, 1).call_classically(y1=1 << 64, y2=1)
    with pytest.raises(ValueError, match="word range"):
        join_integer_pair(1 << 64, 1, 64, 1)


@pytest.mark.ftqc_reference
def test_partition_inverse_preserves_spectator_complex_amplitudes() -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import BloqBuilder, QAny, QBit, Register, Side, Signature
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.bookkeeping import Partition

    words = (Register("cells", QBit(), shape=(2,)), Register("flag", QAny(1)))
    part = Partition(3, words)
    builder, inputs = BloqBuilder.from_signature(
        Signature([Register("x", QAny(3), side=Side.THRU)])
    )
    cells, flag = builder.add(part, x=inputs["x"])
    flag, cells[0] = builder.add(CNOT(), ctrl=flag, target=cells[0])
    output = builder.add(part.adjoint(), cells=cells, flag=flag)
    expected = torch.as_tensor(
        builder.finalize(x=output).tensor_contract(), dtype=torch.complex128
    ).reshape(8, 8)

    wires = (3, 1, 0)
    layout = (("cells", 2), ("flag", 1))
    ports = partition_register(wires, layout)
    # Direct source Cirq calls flatten keyword insertion order. The composite
    # tensor and named native layout use declared order.
    raw = {
        name: np.array([cirq.LineQubit(q) for q in ws]) for name, ws in ports.items()
    }
    op, reversed_keywords = part.adjoint().as_cirq_op(
        cirq.SimpleQubitManager(), flag=raw["flag"], cells=raw["cells"]
    )
    assert op is None
    assert tuple(q.x for q in reversed_keywords["x"]) == (0, 3, 1)
    output_wires = unpartition_register(
        {"flag": ports["flag"], "cells": ports["cells"]}, layout
    )
    assert output_wires == wires

    rows = torch.arange(1, 17, dtype=torch.float64)
    initial = torch.complex(rows, rows.remainder(5) - 2)
    initial /= initial.norm()
    circuit = fq.Circuit(4, inputs=initial, dtype=initial.dtype)
    circuit.cx(ports["flag"][0], ports["cells"][0])
    native = circuit.state().reshape((2,) * 4).permute(2, *output_wires).reshape(2, 8)
    original = initial.reshape((2,) * 4).permute(2, *wires).reshape(2, 8)
    torch.testing.assert_close(native, original @ expected.T, atol=2e-12, rtol=0)
    assert tuple(gate.name for gate in circuit.to_ir().instructions) == ("cx",)


@pytest.mark.ftqc_reference
def test_admitted_mixed_partition_preserves_word_types_and_physical_bindings() -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import QGF, QBit, QUInt
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.bookkeeping.partition import Partition

    from flagquantum.compiler.registers.layout.component_partition import (
        partition_component_program,
        partition_signature,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program.register_types import (
        quantum_bit_type,
        unsigned_register_type,
    )
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    field = FiniteFieldEncoding(2, 2)
    native = (
        Register("index", unsigned_register_type(3)),
        Register("field", field.register_type(), (2,)),
        Register("flag", quantum_bit_type()),
    )
    foreign = (
        SourceRegister("index", QUInt(3)),
        SourceRegister("field", QGF(2, 2), (2,)),
        SourceRegister("flag", QBit()),
    )
    source = Partition(8, foreign)
    wires = (7, 2, 5, 0, 3, 4, 1, 6)
    manager = cirq.SimpleQubitManager()
    operation, parts = source.as_cirq_op(
        manager, x=np.asarray([cirq.LineQubit(wire) for wire in wires])
    )
    expected = tuple(
        tuple(qubit.x for qubit in parts[register.name][idx].flat)
        for register in foreign
        for idx in register.all_idxs()
    )
    assert operation is None
    operation, packed = source.adjoint().as_cirq_op(manager, **parts)
    assert operation is None
    assert tuple(qubit.x for qubit in packed["x"]) == wires

    for reverse, inputs, outputs in (
        (False, (wires,), expected),
        (True, expected, (wires,)),
    ):
        declaration = partition_signature(native, reverse=reverse)
        assert tuple(part.dtype for part in declaration[1:]) == tuple(
            part.dtype for part in native
        )
        program = partition_component_program(native, "mixed", reverse)
        result = bind_register_program(program, inputs, entry="mixed")
        assert result.outputs == outputs
        assert result.events == ()
        assert result.circuit.instructions == ()


@pytest.mark.ftqc_reference
def test_partition_retains_signed_and_fixed_point_metadata_at_execution_boundary() -> (
    None
):
    from qualtran import BQUInt, QFxp, QInt
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.bookkeeping.partition import Partition

    from flagquantum.compiler.registers.layout.component_partition import (
        partition_component_program,
        partition_signature,
    )
    from flagquantum.core.ir.program.register_types import (
        bounded_unsigned_register_type,
        fixed_point_register_type,
        signed_register_type,
    )
    from flagquantum.core.ir.program.registers import Register

    native = (
        Register("selection", bounded_unsigned_register_type(2, 3)),
        Register("signed", signed_register_type(3), (2,)),
        Register("point", fixed_point_register_type(4, 2, signed=True)),
    )
    source = Partition(
        12,
        (
            SourceRegister("selection", BQUInt(2, 3)),
            SourceRegister("signed", QInt(3), (2,)),
            SourceRegister("point", QFxp(4, 2, signed=True)),
        ),
    )
    for reverse, locked in ((False, source), (True, source.adjoint())):
        declared = partition_signature(native, reverse=reverse)
        assert tuple(
            (p.name, p.shape, p.bitsize, p.side.value) for p in declared
        ) == tuple((p.name, p.shape, p.bitsize, p.side.value) for p in locked.signature)
        assert tuple(p.dtype for p in declared[1:]) == tuple(p.dtype for p in native)
        program = partition_component_program(native, "admitted", reverse)
        from flagquantum.compiler.registers.register_binding import (
            bind_register_program,
        )

        parts = ((0, 1), (2, 3, 4), (5, 6, 7), (8, 9, 10, 11))
        packed = tuple(range(12))
        bound = bind_register_program(
            program, parts if reverse else (packed,), entry="admitted"
        )
        assert bound.outputs == ((packed,) if reverse else parts)
        assert bound.circuit.instructions == () and bound.events == ()
