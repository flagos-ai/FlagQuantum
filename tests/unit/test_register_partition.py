"""Concrete partitions preserve source port order without emitting gates."""

import pytest

from flagquantum.compiler.registers.partition import partition_register

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
