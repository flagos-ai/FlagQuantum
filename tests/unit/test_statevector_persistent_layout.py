"""Persistent distributed statevector wire-layout contracts."""

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.backends.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.runtime.backends.statevector.layout import (
    _local_bit_view,
    distributed_swap_rank_local_bits,
    plan_persistent_statevector_layout,
    schedule_statevector_dependency_dag,
)

pytestmark = pytest.mark.unit


def test_layout_swap_persists_and_avoids_repeated_global_gate_exchange():
    circuit = fq.Circuit(5).ry(4, 0.1).rx(4, 0.2).rz(4, 0.3)
    plan = plan_persistent_statevector_layout(circuit, world_size=2)

    assert plan.baseline_communication_gate_count == 3
    assert plan.layout_swap_count == 1
    assert plan.estimated_communication_event_reduction == 2
    assert plan.swaps[0].before_instruction == 0
    assert plan.final_logical_to_physical[4] < 4
    assert plan.segments[0].start_instruction == 0
    assert plan.segments[0].stop_instruction == 3


def test_farthest_next_use_wire_is_evicted_from_local_residency():
    circuit = fq.Circuit(4).x(3).x(0).x(3).x(1).x(2)
    plan = plan_persistent_statevector_layout(circuit, world_size=2)

    first = plan.swaps[0]
    assert first.sharded_logical_wire == 3
    assert first.local_logical_wire == 2
    assert first.local_physical_wire == 2
    assert first.sharded_physical_wire == 3


def test_initial_layout_must_be_a_permutation():
    with pytest.raises(ValueError, match="wire permutation"):
        plan_persistent_statevector_layout(
            fq.Circuit(3).x(0),
            world_size=2,
            initial_logical_to_physical=(0, 0, 2),
        )


def test_dependency_scheduler_reduces_alternating_matching_layout_swaps():
    circuit = fq.Circuit(7)
    for layer in range(3):
        for wire in range(7):
            circuit.ry(wire, 0.1 + 0.01 * (layer * 7 + wire))
        for parity in (0, 1):
            for wire in range(parity, 6, 2):
                circuit.cx(wire, wire + 1)

    baseline = plan_persistent_statevector_layout(circuit, world_size=4)
    scheduled_ir = schedule_statevector_dependency_dag(circuit, world_size=4)
    optimized = plan_persistent_statevector_layout(scheduled_ir, world_size=4)

    assert scheduled_ir.metadata["statevector_dependency_schedule_changed"]
    assert optimized.layout_swap_count < baseline.layout_swap_count


def test_dependency_scheduler_handles_ring_closure_without_semantic_reordering():
    """The wrap-around edge must remain ordered with its two endpoint wires.

    Ring is the stress case for the persistent layout planner: the (n-1, 0)
    edge touches both ends of the logical line.  This regression test makes
    sure the topology-agnostic scheduler can reduce exchanges without
    changing the statevector result.
    """
    circuit = fq.Circuit(7)
    for layer in range(2):
        for wire in range(7):
            circuit.ry(wire, 0.11 + 0.01 * (layer * 7 + wire))
        edges = [(wire, wire + 1) for wire in range(6)]
        edges.append((6, 0))
        for control, target in edges:
            circuit.cx(control, target)

    scheduled = schedule_statevector_dependency_dag(circuit, world_size=4)
    baseline = plan_persistent_statevector_layout(circuit, world_size=4)
    optimized = plan_persistent_statevector_layout(scheduled, world_size=4)
    assert optimized.layout_swap_count <= baseline.layout_swap_count

    canonical = execute_torch_distributed_statevector(circuit)
    reordered = execute_torch_distributed_statevector(scheduled)
    assert torch.allclose(
        reordered.shard_state.amplitudes,
        canonical.shard_state.amplitudes,
        atol=1e-6,
        rtol=1e-6,
    )


def test_dependency_scheduler_preserves_statevector_semantics():
    circuit = fq.Circuit(5)
    for wire in range(5):
        circuit.ry(wire, 0.13 + 0.02 * wire)
    circuit.cx(0, 1).cx(2, 3).cx(1, 2).rz(4, 0.37).cx(3, 4)

    scheduled_ir = schedule_statevector_dependency_dag(circuit, world_size=2)
    canonical = execute_torch_distributed_statevector(circuit)
    scheduled = execute_torch_distributed_statevector(scheduled_ir)

    assert torch.allclose(
        scheduled.shard_state.amplitudes,
        canonical.shard_state.amplitudes,
        atol=1e-6,
        rtol=1e-6,
    )


def test_rank_local_bit_swap_transposes_global_basis_bits(monkeypatch):
    # Two rank shards of a three-wire state.  Values encode canonical basis index.
    shards = [
        torch.tensor([[0, 2, 4, 6]], dtype=torch.complex64),
        torch.tensor([[1, 3, 5, 7]], dtype=torch.complex64),
    ]
    sends = {}

    class Request:
        def wait(self):
            return None

    class Operation:
        def __init__(self, operation, tensor, peer, group, tag):
            self.operation = operation
            self.tensor = tensor
            self.peer = peer

    def exchange(operations):
        send_op, receive_op = operations
        receive_op.tensor.copy_(sends[int(send_op.peer)])
        return [Request(), Request()]

    # Materialize both packed sends before emulating simultaneous P2P.
    sends[0] = _local_bit_view(shards[0], bit_position=1, bit_value=1).contiguous()
    sends[1] = _local_bit_view(shards[1], bit_position=1, bit_value=0).contiguous()
    monkeypatch.setattr(torch.distributed, "P2POp", Operation)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", exchange)
    outputs = []
    for rank in range(2):
        output, count, byte_count = distributed_swap_rank_local_bits(
            shards[rank],
            rank=rank,
            n_wires=3,
            rank_bits=1,
            local_physical_wire=0,
            sharded_physical_wire=2,
        )
        outputs.append(output)
        assert count == 1
        assert byte_count == 16

    assert torch.equal(outputs[0], torch.tensor([[0, 2, 1, 3]], dtype=torch.complex64))
    assert torch.equal(outputs[1], torch.tensor([[4, 6, 5, 7]], dtype=torch.complex64))
