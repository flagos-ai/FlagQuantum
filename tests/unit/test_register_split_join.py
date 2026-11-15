"""Check pinned register reshapes and independent classical encodings."""

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
from flagquantum.compiler.registers.layout.register_bits import (
    join_integer,
    join_register,
    split_integer,
    split_register,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_original_ten_wire_allocate_split_join_transfers_the_zero_word() -> None:
    from qualtran import BloqBuilder, QAny
    from qualtran.bloqs.bookkeeping import Allocate, Join, Split

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register, RegisterSide
    from flagquantum.core.ir.program.signatures import Signature

    width = 10
    original = BloqBuilder()
    source_dtype = QAny(width)
    word = original.add(Allocate(source_dtype))
    source_bits = original.add(Split(source_dtype), reg=word)
    source_output = original.add(Join(source_dtype), reg=source_bits)
    tensor = original.finalize(out=source_output).tensor_contract()
    expected = torch.as_tensor(tensor, dtype=torch.complex128)
    dtype = opaque_register_type(width)
    builder = ComponentProgramBuilder(
        Signature((Register("out", dtype, side=RegisterSide.RIGHT),))
    )
    temporary = builder.allocate(dtype=dtype)
    output = builder.join(builder.split(temporary), dtype)
    module = builder.finalize({"out": output})
    wires = tuple(reversed(range(width)))
    bound = bind_register_program(module, (), clean=wires, allow_workspace_outputs=True)
    assert bound.outputs == (wires,) and bound.circuit.instructions == ()
    (allocation,) = bound.events
    assert (allocation.kind, allocation.wires, allocation.dirty) == (
        "allocate",
        wires,
        False,
    )
    circuit = fq.Circuit(width, dtype=expected.dtype)
    circuit.append(FieldOperation(width, bound.circuit.instructions))
    actual = circuit.state().reshape((2,) * width).permute(wires).reshape(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)


@pytest.mark.ftqc_reference
def test_original_four_wire_split_flip_join_values_and_zero_gate_rebinding() -> None:
    from functools import partial

    from qualtran import BloqBuilder, QAny
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.bookkeeping import Allocate, Join, Split

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.core.ir import Instruction
    from flagquantum.simulation.classical_bit_binding import (
        allocate_classical_value,
        join_classical_value,
        split_classical_value,
    )
    from flagquantum.simulation.integer_encoding import (
        integer_from_bits,
        integer_to_bits,
    )

    width = 4
    dtype = QAny(width)
    reference = BloqBuilder()
    word = reference.add(Allocate(dtype))
    parts = reference.add(Split(dtype), reg=word)
    parts[1] = reference.add(XGate(), q=parts[1])
    source = reference.finalize(y=reference.add(Join(dtype), reg=parts))
    encode = partial(integer_to_bits, width=width)
    decode = partial(integer_from_bits, width=width)
    value = allocate_classical_value(width, decode)
    values = split_classical_value(value, encode)
    assert values.tolist() == [0, 0, 0, 0]
    values[1] ^= 1
    assert source.call_classically() == (join_classical_value(values, decode),) == (4,)
    wires = tuple(range(width))
    native_parts = split_register(wires)
    assert join_register(native_parts) == wires
    circuit = fq.Circuit(width, dtype=torch.complex128)
    circuit.append(FieldOperation(width, (Instruction("x", native_parts[1]),)))
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )
    assert circuit.to_ir().instructions == (Instruction("x", (1,)),)


@pytest.mark.ftqc_reference
def test_original_split_keeps_numpy_scalar_values_and_range_failures() -> None:
    from functools import partial

    import numpy as np
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Split

    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.simulation.classical_bit_binding import split_classical_value
    from flagquantum.simulation.classical_typed_values import (
        collect_typed_register_value,
    )
    from flagquantum.simulation.integer_encoding import integer_to_bits

    source = Split(QAny(8))
    port = Register("reg", quantum_bit_type(), shape=(8,))
    encode = partial(integer_to_bits, width=8)
    for value in (255, np.uint8(255)):
        (expected,) = source.call_classically(reg=value)
        actual = split_classical_value(value, encode)
        assert actual.dtype == source.on_classical_vals(value)["reg"].dtype
        assert actual.tolist() == expected.tolist() == [1] * 8
        collected = collect_typed_register_value(
            port, {(i,): bit for i, bit in enumerate(actual)}
        )
        assert isinstance(collected, np.ndarray)
        assert collected.dtype == expected.dtype
        np.testing.assert_array_equal(collected, expected)
    for value in (256, np.uint16(256)):
        with pytest.raises(ValueError):
            source.call_classically(reg=value)
        with pytest.raises(ValueError):
            split_classical_value(value, encode)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "wires", [(), (2,), (3, 0, 2), (4, 0, 2, 1, 3), tuple(reversed(range(123)))]
)
def test_split_join_wire_binding_and_resources(wires: tuple[int, ...]) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.component_rebindings import (
        join_word_function,
        split_word_function,
    )
    from flagquantum.core.ir.program.register_types import opaque_register_type

    width = len(wires)
    split, join = Split(QAny(width)), Join(QAny(width))
    assert split.adjoint() == join and join.adjoint() == split
    source_reg = np.array([cirq.LineQubit(wire) for wire in wires], dtype=object)
    op, outputs = split.as_cirq_op(qubit_manager=None, reg=source_reg)
    assert op is None and outputs["reg"].shape == (width, 1)
    parts = split_register(wires)
    assert parts == tuple(tuple(q.x for q in port) for port in outputs["reg"])
    op, outputs = join.as_cirq_op(qubit_manager=None, **outputs)
    assert op is None and outputs["reg"].shape == (width,)
    assert join_register(parts) == tuple(q.x for q in outputs["reg"]) == wires
    for opcode, source, helper in (
        ("split", split, split_word_function),
        ("join", join, join_word_function),
    ):
        registers = helper(opcode, opaque_register_type(width)).registers
        assert registers is not None
        assert tuple(
            (port.name, port.shape, port.bitsize, port.side.value) for port in registers
        ) == tuple(
            (port.name, port.shape, port.bitsize, port.side.value)
            for port in source.signature
        )
        assert get_cost_value(source, QECGatesCost()).asdict() == {}
        assert get_cost_value(source, QubitCount()) == width
        assert query_qec_gate_cost(BookkeepingCall(opcode, width)) == QECGateCost()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 5])
@pytest.mark.parametrize("signed", [False, True])
def test_split_join_integer_parity(width: int, signed: bool) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import QInt, QUInt
    from qualtran.bloqs.bookkeeping import Join, Split

    dtype = QInt(width) if signed else QUInt(width)
    split, join = Split(dtype), Join(dtype)
    for value in dtype.get_classical_domain():
        bits = split_integer(value, width, signed=signed)
        source_bits = split.call_classically(reg=value)[0]
        assert bits == tuple(source_bits)
        assert (
            join_integer(bits, signed=signed)
            == join.call_classically(reg=np.asarray(bits))[0]
            == value
        )


@pytest.mark.ftqc_reference
def test_reordered_join_keeps_complex_action_including_spectator() -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, QAny, Register, Side, Signature
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.bookkeeping import Join, Split

    width, wires = 3, (3, 1, 0)
    dtype = QAny(width)
    builder, inputs = BloqBuilder.from_signature(
        Signature(
            [
                Register("data", dtype, side=Side.LEFT),
                Register("out", dtype, side=Side.RIGHT),
            ]
        )
    )
    parts = builder.add(Split(dtype), reg=inputs["data"])
    parts[1] = builder.add(XGate(), q=parts[1])
    permutation = (2, 0, 1)
    joined = builder.add(Join(dtype), reg=parts[list(permutation)])
    source = torch.as_tensor(
        builder.finalize(out=joined).tensor_contract(), dtype=torch.complex128
    ).reshape(8, 8)

    rows = torch.arange(1, 17, dtype=torch.float64)
    initial = torch.complex(rows, rows.remainder(7) - 3)
    initial /= initial.norm()
    native_parts = split_register(wires)
    output = join_register(tuple(native_parts[i] for i in permutation))
    circuit = fq.Circuit(4, inputs=initial, dtype=initial.dtype).x(native_parts[1][0])
    actual = circuit.state().reshape((2,) * 4).permute(2, *output).reshape(2, 8)
    by_input = initial.reshape((2,) * 4).permute(2, *wires).reshape(2, 8)
    torch.testing.assert_close(actual, by_input @ source.T, atol=2e-12, rtol=0)
    assert tuple(gate.name for gate in circuit.to_ir().instructions) == ("x",)


def test_bit_ports_reject_aliases_and_integer_encodings_check_bounds() -> None:
    with pytest.raises(ValueError, match="distinct"):
        split_register((1, 1))
    with pytest.raises(ValueError, match="distinct"):
        join_register(((1,), (1,)))
    with pytest.raises(ValueError, match="one wire"):
        join_register(((1, 2),))
    with pytest.raises(ValueError, match="fit"):
        split_integer(-2, 1, signed=True)
    with pytest.raises(ValueError, match="fit"):
        split_integer(1, 1, signed=True)
    with pytest.raises(ValueError, match="binary"):
        join_integer((0, 2))


@pytest.mark.ftqc_reference
def test_symbolic_and_empty_integer_encoding_boundaries() -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Join, Split
    from sympy import Symbol

    for op in (Split, Join):
        with pytest.raises(ValueError, match="symbolic"):
            op(QAny(Symbol("n")))
    with pytest.raises(ValueError, match="Incorrect shape"):
        Split(QAny(0)).call_classically(reg=0)
    with pytest.raises(ValueError, match="invalid literal"):
        Join(QAny(0)).call_classically(reg=np.asarray([], dtype=int))
    with pytest.raises(ValueError, match="positive concrete width"):
        split_integer(0, 0)
    with pytest.raises(ValueError, match="nonempty binary"):
        join_integer(())
