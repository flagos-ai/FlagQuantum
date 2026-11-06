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
from flagquantum.compiler.registers.register_bits import (
    join_integer,
    join_register,
    split_integer,
    split_register,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("wires", [(), (2,), (3, 0, 2), tuple(reversed(range(123)))])
def test_split_join_wire_binding_and_resources(wires: tuple[int, ...]) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

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
    for opcode, source in (("split", split), ("join", join)):
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
