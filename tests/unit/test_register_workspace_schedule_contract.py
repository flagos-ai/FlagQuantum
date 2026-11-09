"""Pinned physical-lifetime witnesses for the next workspace lowering slice."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("width", [1, 3])
def test_source_allocation_can_be_split_and_freed_one_bit_at_a_time(width, dirty):
    import cirq
    from qualtran import BloqBuilder, QAny, Signature

    events = []
    wires = tuple(cirq.LineQubit.range(1, width + 1))

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n, dim=2):
            assert n == width and dim == 2
            events.append(("allocate", wires))
            return list(wires)

        def qborrow(self, n, dim=2):
            assert n == width and dim == 2
            events.append(("borrow", wires))
            return list(wires)

        def qfree(self, qubits):
            events.append(("free", tuple(qubits)))

    builder, inputs = BloqBuilder.from_signature(Signature.build(data=1))
    allocated = builder.allocate(dtype=QAny(width), dirty=dirty)
    bits = builder.split(allocated)
    for bit in reversed(bits):
        builder.free(bit, dirty=dirty)
    composite = builder.finalize(**inputs)
    circuit, outputs = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), data=np.array([cirq.LineQubit(0)])
    )
    assert tuple(circuit.all_operations()) == ()
    assert tuple(outputs["data"]) == (cirq.LineQubit(0),)
    assert events[0] == ("borrow" if dirty else "allocate", wires)
    freed = [qubit for kind, qubits in events[1:] for qubit in qubits]
    assert len(events[1:]) == width
    assert set(freed) == set(wires) and len(freed) == width


@pytest.mark.parametrize("dirty", [False, True])
def test_source_releases_join_of_distinct_allocations_as_one_register(dirty):
    import cirq
    from qualtran import BloqBuilder, QAny, Signature

    events = []

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n, dim=2):
            assert n == 1 and dim == 2
            wire = cirq.LineQubit(len(events) + 1)
            events.append(("allocate", (wire,)))
            return [wire]

        def qborrow(self, n, dim=2):
            wires = self.qalloc(n, dim)
            events[-1] = ("borrow", tuple(wires))
            return wires

        def qfree(self, qubits):
            events.append(("free", tuple(qubits)))

    builder, inputs = BloqBuilder.from_signature(Signature.build(data=1))
    first = builder.allocate(dtype=QAny(1), dirty=dirty)
    second = builder.allocate(dtype=QAny(1), dirty=dirty)
    joined = builder.join([*builder.split(second), *builder.split(first)])
    builder.free(joined, dirty=dirty)
    composite = builder.finalize(**inputs)
    circuit, _ = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), data=np.array([cirq.LineQubit(0)])
    )
    assert tuple(circuit.all_operations()) == ()
    assert [event[0] for event in events] == [
        "borrow" if dirty else "allocate",
        "borrow" if dirty else "allocate",
        "free",
    ]
    assert set(events[2][1]) == {events[0][1][0], events[1][1][0]}


@pytest.mark.parametrize("dirty", [False, True])
def test_source_clean_and_borrowed_work_preserve_full_complex_joint_state(dirty):
    import cirq
    from qualtran import BloqBuilder, QBit, Signature
    from qualtran.bloqs.basic_gates import CNOT

    work = cirq.LineQubit(1)
    events = []

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n, dim=2):
            assert n == dim - 1 == 1
            events.append("allocate")
            return [work]

        def qborrow(self, n, dim=2):
            assert n == dim - 1 == 1
            events.append("borrow")
            return [work]

        def qfree(self, qubits):
            assert tuple(qubits) == (work,)
            events.append("free")

    builder, inputs = BloqBuilder.from_signature(Signature.build(data=1))
    data = inputs["data"]
    borrowed = builder.allocate(dtype=QBit(), dirty=dirty)
    for _ in range(2):
        data, borrowed = builder.add(CNOT(), ctrl=data, target=borrowed)
    builder.free(borrowed, dirty=dirty)
    composite = builder.finalize(data=data)
    circuit, _ = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), data=np.array([cirq.LineQubit(0)])
    )
    assert tuple(circuit.all_operations()) == (
        cirq.CNOT(cirq.LineQubit(0), work),
        cirq.CNOT(cirq.LineQubit(0), work),
    )
    state = torch.tensor([1, 2j, -3j, 4, 2 + 1j, -1, 3, -2j], dtype=torch.complex128)
    if not dirty:
        state.reshape(2, 2, 2)[:, 1, :] = 0
    state /= state.norm()
    actual = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            circuit, initial_state=state.numpy(), qubit_order=cirq.LineQubit.range(3)
        )
        .final_state_vector
    )
    torch.testing.assert_close(torch.as_tensor(actual), state, rtol=0, atol=2e-12)
    if not dirty:
        assert (
            torch.count_nonzero(torch.as_tensor(actual).reshape(2, 2, 2)[:, 1, :]) == 0
        )
    assert events == ["borrow" if dirty else "allocate", "free"]
