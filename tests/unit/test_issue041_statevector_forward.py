"""Focused contracts for the PyTorch-native sharded forward executor."""

import pytest
import torch

import flagquantum as fq
from flagquantum.core import OPERATOR_SCHEMAS, CircuitIR, Instruction
from flagquantum.runtime.backends.statevector.forward import (
    FullStateMaterializationError,
    StatevectorExchangeWorkspace,
    _independent_tensor_bytes,
    _triton_local_cx_segment_enabled,
    communication_aware_wire_layout,
    execute_torch_distributed_statevector,
)

pytestmark = pytest.mark.unit


def test_dependency_schedule_auto_enables_cx_segments_unless_overridden(monkeypatch):
    scheduled = CircuitIR(
        n_wires=2,
        instructions=(),
        metadata={
            "statevector_dependency_scheduled": True,
            "statevector_dependency_schedule_changed": True,
        },
    )

    monkeypatch.delenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", raising=False)
    assert _triton_local_cx_segment_enabled(scheduled)
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "0")
    assert not _triton_local_cx_segment_enabled(scheduled)
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "1")
    assert _triton_local_cx_segment_enabled(scheduled)


def test_exchange_workspace_classifies_torchrun_peer_tiers(monkeypatch):
    import flagquantum.runtime.backends.statevector.forward as forward

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setattr(forward.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(forward.dist, "get_rank", lambda: 9)
    workspace = StatevectorExchangeWorkspace()

    workspace.record_peer_exchange(10, 64)
    workspace.record_peer_exchange(1, 128)

    assert workspace.intra_node_message_count == 1
    assert workspace.intra_node_bytes == 64
    assert workspace.inter_node_message_count == 1
    assert workspace.inter_node_bytes == 128


def test_single_rank_uses_same_executor_contract_without_distributed_claim():
    circuit = fq.Circuit(3).h(0).cx(0, 2).ry(1, 0.2)
    result = execute_torch_distributed_statevector(circuit)
    assert torch.allclose(result.shard_state.amplitudes, circuit.state())
    assert result.summary()["distribution_semantics"] == "single_device_fast_path"
    assert result.peak_scratch_bytes >= result.plan.per_rank_state_bytes
    assert result.summary()["scratch_accounting"].endswith("including_output_buffer")
    dispatch = result.summary()["kernel_dispatch"]
    assert dispatch["triton_execution_count"] == 0
    assert dispatch["pytorch_fallback_count"] == 3
    assert {
        (record["feature"], record["reason"], record["count"])
        for record in dispatch["decisions"]
    } == {
        ("local_1q", "disabled_by_policy", 2),
        ("local_cx", "input_not_supported", 1),
    }
    with pytest.raises(FullStateMaterializationError, match="forbidden"):
        result.full_state()


def test_local_world_size_defaults_from_torchrun_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")

    result = execute_torch_distributed_statevector(fq.Circuit(3))

    assert result.plan.local_world_size == 2
    assert result.plan.node_count == 2


def test_subgroup_defaults_local_world_size_to_subgroup_size(monkeypatch):
    subgroup = object()
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")

    result = execute_torch_distributed_statevector(
        fq.Circuit(2), process_group=subgroup
    )

    assert result.plan.local_world_size == 2
    assert result.plan.node_count == 1


def test_scratch_accounting_does_not_count_aliasing_chunk_storage():
    owner = torch.empty((1, 8), dtype=torch.complex64)
    view = owner[:, :4].contiguous()
    copy = view.clone()

    assert _independent_tensor_bytes(view, owner) == 0
    assert _independent_tensor_bytes(copy, owner) == copy.numel() * copy.element_size()


def test_unsupported_gate_fails_before_state_initialization(monkeypatch):
    ir = CircuitIR(
        n_wires=1,
        instructions=(Instruction(name="bit_flip", wires=(0,)),),
    )
    initialized = False

    def forbidden(*args, **kwargs):
        nonlocal initialized
        initialized = True
        raise AssertionError("state must not initialize")

    monkeypatch.setattr(
        "flagquantum.runtime.backends.statevector.forward.initialize_statevector_shard",
        forbidden,
    )
    with pytest.raises((KeyError, ValueError)):
        execute_torch_distributed_statevector(ir)
    assert not initialized


def test_every_declared_unitary_operator_executes_in_arbitrary_sequence():
    circuit = fq.Circuit(3)
    for schema in OPERATOR_SCHEMAS.values():
        if not schema.unitary:
            continue
        params = {name: 0.17 for name in schema.parameters}
        circuit.gate(schema.opcode, tuple(range(schema.arity)), **params)
    result = execute_torch_distributed_statevector(circuit)
    assert torch.allclose(result.shard_state.amplitudes, circuit.state(), atol=1e-5)


def test_pytorch_executor_import_does_not_initialize_optional_jax():
    import subprocess
    import sys

    code = (
        "import sys; import flagquantum.runtime.backends.statevector.forward; "
        "assert 'jax' not in sys.modules and 'jaxlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_communication_aware_layout_moves_inactive_wire_to_rank_bits():
    theta = torch.tensor(0.23, requires_grad=True)
    ir = fq.Circuit(6).ry(5, theta).cx(5, 4).ry(4, theta).to_ir()

    remapped, mapping = communication_aware_wire_layout(
        ir, world_size=4, preferred_local_wires=(4,)
    )

    assert mapping[4] < 4 and mapping[5] < 4
    assert {mapping[0], mapping[1]} == {4, 5}
    assert tuple(instruction.wires for instruction in remapped.instructions) == (
        (mapping[5],),
        (mapping[5], mapping[4]),
        (mapping[4],),
    )


def test_communication_aware_layout_penalizes_full_shard_subgroup_alignment():
    circuit = fq.Circuit(6)
    for wire in range(6):
        circuit.ry(wire, torch.tensor(0.1 + wire * 0.01, requires_grad=True))
    for wire in range(5):
        circuit.cx(wire, wire + 1)

    _, mapping = communication_aware_wire_layout(
        circuit.to_ir(), world_size=2, preferred_local_wires=(3,)
    )

    # Both endpoints have the same gate-touch score.  Sharding wire 0 would
    # map its CX partner to the highest local address bit and force subgroup
    # exchange to align every chunk to the complete shard.  Wire 5 keeps the
    # partner on the lowest local bit and therefore preserves bounded chunks.
    assert mapping == tuple(range(6))


def test_multi_node_rank_bit_order_puts_low_activity_wire_on_node_bit(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    circuit = fq.Circuit(28)
    for wire in range(28):
        circuit.ry(wire, torch.tensor(0.1 + wire * 0.001, requires_grad=True))
    for wire in range(27):
        circuit.cx(wire, wire + 1)

    _, mapping = communication_aware_wire_layout(
        circuit.to_ir(), world_size=16, preferred_local_wires=(14,)
    )

    assert mapping[27] == 24
    assert {mapping[24], mapping[25], mapping[26]} == {25, 26, 27}

    monkeypatch.setenv("FQ_STATEVECTOR_TOPOLOGY_AWARE_RANK_BITS", "0")
    _, canonical_rank_bits = communication_aware_wire_layout(
        circuit.to_ir(), world_size=16, preferred_local_wires=(14,)
    )
    assert canonical_rank_bits[24] == 24
    assert canonical_rank_bits[27] == 27


def test_single_rank_communication_aware_result_declares_basis_order():
    result = execute_torch_distributed_statevector(
        fq.Circuit(3).h(2), wire_layout="communication_aware"
    )
    assert result.logical_to_physical_wires == (0, 1, 2)
    assert result.summary()["amplitude_basis_order"] == "canonical_logical"
