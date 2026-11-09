"""Concrete partitions preserve source port order without emitting gates."""

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
def test_source_partition_shaped_ports_and_join():
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
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
def test_partition_rejects_bad_layout(packed, ports, message):
    with pytest.raises(ValueError, match=message):
        partition_register(packed, ports)


def test_zero_width_named_partition():
    assert partition_register((), (("empty", 0),)) == {"empty": ()}


def test_unpartition_checks_names_widths_and_aliases():
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
def test_split2_join2_match_source_physical_bindings_and_resources(widths):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
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
    for opcode, leaf in (("split2", source), ("join2", source.adjoint())):
        assert partition_width(layout) == get_cost_value(leaf, QubitCount())
        assert get_cost_value(leaf, QECGatesCost()).asdict() == {}
        assert query_qec_gate_cost(BookkeepingCall(opcode, widths)) == QECGateCost()


@pytest.mark.ftqc_reference
def test_split2_join2_symbolic_width_and_integer_boundaries():
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
def test_partition_inverse_preserves_spectator_complex_amplitudes():
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
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
