"""Wire pool ownership through split, cross-allocation join and partial release."""

import pytest

from flagquantum.compiler.registers.layout.workspace import RegisterWorkspace

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dirty", [False, True])
def test_split_register_releases_individual_wires_in_current_virtual_order(dirty):
    pool = (
        RegisterWorkspace((), borrowed=(7, 3, 5))
        if dirty
        else RegisterWorkspace((7, 3, 5))
    )
    lease = pool.allocate(3, dirty=dirty)
    pool.free_wires((lease.wires[2],))
    pool.free_wires((lease.wires[0],))
    # The still-live middle wire cannot be allocated or freed twice.
    assert pool.allocate(2, dirty=dirty).wires == (7, 5)
    pool.free(lease)
    assert pool.allocate(1, dirty=dirty).wires == (3,)
    with pytest.raises(ValueError, match="live lease"):
        pool.free(lease)
    assert pool.used_wires == (3, 5, 7)


@pytest.mark.parametrize("dirty", [False, True])
def test_joined_distinct_allocations_return_in_requested_release_order(dirty):
    pool = (
        RegisterWorkspace((), borrowed=(1, 4, 8))
        if dirty
        else RegisterWorkspace((1, 4, 8))
    )
    first = pool.allocate(1, dirty=dirty)
    second = pool.allocate(2, dirty=dirty)
    pool.free_wires((second.wires[1], first.wires[0], second.wires[0]))
    assert pool.allocate(3, dirty=dirty).wires == (8, 1, 4)
    for lease in (first, second):
        with pytest.raises(ValueError, match="live lease"):
            pool.free(lease)


def test_cross_pool_join_returns_each_wire_to_its_original_pool():
    pool = RegisterWorkspace((2, 7), borrowed=(3, 8))
    clean = pool.allocate(2)
    dirty = pool.allocate(2, dirty=True)
    pool.free_wires((8, 7, 3, 2))
    assert pool.allocate(2).wires == (7, 2)
    assert pool.allocate(2, dirty=True).wires == (8, 3)
    for lease in (clean, dirty):
        with pytest.raises(ValueError, match="live lease"):
            pool.free(lease)


@pytest.mark.parametrize("wires", [(1, 1), (1, True), (1, 99), (1, -1)])
def test_invalid_partial_release_preserves_all_ownership_and_pool_order(wires):
    pool = RegisterWorkspace((1, 2))
    lease = pool.allocate(2)
    with pytest.raises(ValueError, match="distinct|live wires"):
        pool.free_wires(wires)
    with pytest.raises(ValueError, match="Insufficient"):
        pool.allocate(1)
    pool.free(lease)
    assert pool.allocate(2).wires == (1, 2)


def test_stale_partial_release_fails_even_when_lease_was_not_fully_released():
    pool = RegisterWorkspace((1, 2, 3))
    lease = pool.allocate(3)
    pool.free_wires((2,))
    with pytest.raises(ValueError, match="live wires"):
        pool.free_wires((1, 2))
    # Whole release names the remaining live wires after the split.
    with pytest.raises(ValueError, match="still live"):
        pool.free(lease, order=(3, 2, 1))
    pool.free(lease, order=(3, 1))
    assert pool.allocate(3).wires == (3, 1, 2)


def test_empty_word_lifetimes_remain_distinct_from_empty_wire_release():
    pool = RegisterWorkspace(())
    first = pool.allocate(0)
    second = pool.allocate(0, dirty=True)
    pool.free_wires(())
    pool.free(first)
    pool.free(second)
    for lease in (first, second):
        with pytest.raises(ValueError, match="live lease"):
            pool.free(lease)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dirty", [False, True])
def test_pool_partial_release_matches_original_source_split_join_manager_events(dirty):
    import cirq
    import numpy as np
    from qualtran import BloqBuilder, QAny, Signature

    source_events = []
    source_available = list(reversed((2, 7, 4)))
    source_live = set()

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n, dim=2):
            assert dim == 2
            wires = tuple(source_available.pop() for _ in range(n))
            source_live.update(wires)
            source_events.append(("allocate", wires))
            return [cirq.LineQubit(wire) for wire in wires]

        def qborrow(self, n, dim=2):
            qubits = self.qalloc(n, dim)
            source_events[-1] = ("borrow", source_events[-1][1])
            return qubits

        def qfree(self, qubits):
            wires = tuple(qubit.x for qubit in qubits)
            assert set(wires) <= source_live
            source_live.difference_update(wires)
            source_available.extend(reversed(wires))
            source_events.append(("free", wires))

    builder, inputs = BloqBuilder.from_signature(Signature.build(data=1))
    word = builder.allocate(dtype=QAny(3), dirty=dirty)
    bits = builder.split(word)
    builder.free(bits[1], dirty=dirty)
    joined = builder.join([bits[2], bits[0]])
    builder.free(joined, dirty=dirty)
    composite = builder.finalize(**inputs)
    circuit, _ = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), data=np.array([cirq.LineQubit(0)])
    )
    assert not tuple(circuit.all_operations())
    pool = (
        RegisterWorkspace((), borrowed=(2, 7, 4))
        if dirty
        else RegisterWorkspace((2, 7, 4))
    )
    native_events = []
    for kind, wires in source_events:
        if kind in {"allocate", "borrow"}:
            actual = pool.allocate(len(wires), dirty=dirty).wires
            native_events.append((kind, actual))
        else:
            pool.free_wires(wires)
            native_events.append(("free", wires))
    assert native_events == source_events
    assert not source_live
    assert pool.allocate(3, dirty=dirty).wires == tuple(reversed(source_available))
