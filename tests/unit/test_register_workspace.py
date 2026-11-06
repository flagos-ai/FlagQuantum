"""Check clean allocation and borrowing against the pinned bookkeeping leaves."""

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
from flagquantum.compiler.registers.int_vector import int_effect_bindings
from flagquantum.compiler.registers.workspace import RegisterWorkspace
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    tensor_network_basis_effect,
)

pytestmark = pytest.mark.unit


def test_workspace_reuses_released_wires_without_aliasing_live_leases() -> None:
    pool = RegisterWorkspace((3, 8, 1), borrowed=(6, 4))
    first = pool.allocate(2)
    borrowed = pool.allocate(2, dirty=True)
    second = pool.allocate(1)
    empty = pool.allocate(0)
    assert (first.wires, borrowed.wires, second.wires, empty.wires) == (
        (3, 8),
        (6, 4),
        (1,),
        (),
    )
    with pytest.raises(ValueError, match="Insufficient clean"):
        pool.allocate(1)
    with pytest.raises(ValueError, match="Insufficient borrowed"):
        pool.allocate(1, dirty=True)
    with pytest.raises(ValueError, match="live lease"):
        RegisterWorkspace((3, 8)).free(first)
    pool.free(empty)
    pool.free(first)
    reused = pool.allocate(2)
    assert reused.wires == first.wires
    with pytest.raises(ValueError, match="live lease"):
        pool.free(first)
    pool.free(second)
    pool.free(reused)
    with pytest.raises(ValueError, match="permute"):
        pool.free(borrowed, order=(6, 6))
    pool.free(borrowed, order=(4, 6))
    assert pool.used_wires == (1, 3, 4, 6, 8)
    assert pool.allocate(2, dirty=True).wires == (4, 6)


def test_workspace_validates_aliasing_and_request_size() -> None:
    with pytest.raises(ValueError, match="distinct"):
        RegisterWorkspace((1,), borrowed=(1,))
    with pytest.raises(ValueError, match="nonnegative"):
        RegisterWorkspace((-1,))
    with pytest.raises(ValueError, match="nonnegative width"):
        RegisterWorkspace(()).allocate(-1)
    pool = RegisterWorkspace((1,))
    lease = pool.allocate(1)
    with pytest.raises(ValueError, match="permute"):
        pool.free(lease, order=(True,))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("width", [1, 3])
def test_allocate_free_match_source_binding_and_resources(
    width: int, dirty: bool
) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran import QAny, Side
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    wires = tuple(reversed(range(1, 1 + width)))
    pool = RegisterWorkspace((), borrowed=wires) if dirty else RegisterWorkspace(wires)
    lease = pool.allocate(width, dirty=dirty)
    events: list[tuple[str, tuple[int, ...]]] = []

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2):
            assert n == width and dim == 2
            events.append(("allocate", lease.wires))
            return [cirq.LineQubit(wire) for wire in lease.wires]

        def qborrow(self, n: int, dim: int = 2):
            assert n == width and dim == 2
            events.append(("borrow", lease.wires))
            return [cirq.LineQubit(wire) for wire in lease.wires]

        def qfree(self, qubits):
            events.append(("free", tuple(qubit.x for qubit in qubits)))

    allocate, free = Allocate(QAny(width), dirty), Free(QAny(width), dirty)
    assert allocate.adjoint() == free and free.adjoint() == allocate
    assert (
        allocate.signature[0].side == Side.RIGHT and free.signature[0].side == Side.LEFT
    )
    manager = Manager()
    instruction, outputs = allocate.as_cirq_op(manager)
    assert instruction is None and outputs["reg"].shape == (width,)
    assert tuple(qubit.x for qubit in outputs["reg"]) == wires
    assert free.as_cirq_op(manager, **outputs) == (None, {})
    pool.free(lease)
    assert events == [("borrow" if dirty else "allocate", wires), ("free", wires)]
    for opcode, source in (("allocate", allocate), ("free", free)):
        assert get_cost_value(source, QECGatesCost()).asdict() == {}
        assert get_cost_value(source, QubitCount()) == width
        assert (
            query_qec_gate_cost(BookkeepingCall(opcode, (width, dirty)))
            == QECGateCost()
        )
    assert allocate.call_classically() == (0,)
    if dirty:
        assert free.call_classically(reg=1) == ()
    else:
        with pytest.raises(ValueError, match="non-zero"):
            free.call_classically(reg=1)
    # The source tensor view uses zero even for the dirty flag.
    zero = torch.zeros(1 << width, dtype=torch.complex128)
    zero[0] = 1
    for source in (allocate, free):
        actual = torch.as_tensor(source.tensor_contract(), dtype=zero.dtype).reshape(-1)
        torch.testing.assert_close(actual, zero)


@pytest.mark.ftqc_reference
def test_clean_free_preserves_unnormalized_source_projection() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Free

    wires = (2, 1)
    state = torch.tensor([1, 2j, 3, 4j, -1j, 5, -2, 1 + 2j], dtype=torch.complex128)
    state /= state.norm()
    bra = torch.as_tensor(Free(QAny(2)).tensor_contract(), dtype=state.dtype).reshape(4)
    ordered = state.reshape(2, 2, 2).permute(0, 2, 1).reshape(2, 4)
    plan = build_tensor_network(
        fq.Circuit(3, inputs=state, dtype=state.dtype), dtype=state.dtype
    )
    result = tensor_network_basis_effect(plan, int_effect_bindings(wires, 0))
    torch.testing.assert_close(result[0], ordered @ bra, rtol=0, atol=2e-12)
    assert result.norm() < 1
