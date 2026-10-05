"""Focused contracts for the PyTorch-native sharded forward executor."""

from types import SimpleNamespace

import pytest
import torch

import flagquantum as fq
from flagquantum.core import OPERATOR_SCHEMAS, CircuitIR, Instruction
from flagquantum.runtime.executors.statevector.control_subspace_dispatch import (
    _triton_control_subspace_pack_decision,
    _triton_control_subspace_unpack_decision,
)
from flagquantum.runtime.executors.statevector.cx_segment_dispatch import (
    _triton_local_cx_segment_decision,
    _triton_local_cx_segment_enabled,
)
from flagquantum.runtime.executors.statevector.forward import (
    FullStateMaterializationError,
    StatevectorExchangeWorkspace,
    _independent_tensor_bytes,
    _triton_local_1q_requested,
    _triton_local_cx_requested,
    _vectorized_cross_shard_cx,
    _vectorized_local_gate,
    _vectorized_pair_exchange_gate,
    _wait_for_exchange,
    communication_aware_qubit_layout,
)
from flagquantum.runtime.executors.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.runtime.executors.statevector.forward_sweep import (
    _ShardedForwardSweep,
)
from flagquantum.runtime.executors.statevector.kernel_dispatch import (
    KernelDispatchEvidence,
)
from flagquantum.runtime.executors.statevector.models import (
    StatevectorShard,
    StatevectorShardState,
)
from flagquantum.runtime.executors.statevector.program_cache import (
    remap_instruction_qubits,
)
from flagquantum.runtime.executors.statevector.transpose_dispatch import (
    _triton_transpose_1q_decision,
)

pytestmark = pytest.mark.unit


class _ExchangeRequest:
    def __init__(self):
        self.block_current_stream_calls = 0
        self.wait_calls = 0

    def block_current_stream(self):
        self.block_current_stream_calls += 1

    def wait(self):
        self.wait_calls += 1


def test_flagos_exchange_uses_provider_neutral_wait():
    flagos_request = _ExchangeRequest()
    cuda_request = _ExchangeRequest()

    _wait_for_exchange(flagos_request, SimpleNamespace(type="flagos"))
    _wait_for_exchange(cuda_request, SimpleNamespace(type="cuda"))

    assert flagos_request.wait_calls == 1
    assert flagos_request.block_current_stream_calls == 0
    assert cuda_request.wait_calls == 0
    assert cuda_request.block_current_stream_calls == 1


def test_cx_segment_default_window_and_override(monkeypatch):
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )
    scheduled = CircuitIR(
        n_wires=2,
        instructions=(),
        metadata={
            "statevector_dependency_scheduled": True,
            "statevector_dependency_schedule_changed": True,
        },
    )

    monkeypatch.delenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", raising=False)
    assert _triton_local_cx_segment_enabled(scheduled, shape=(1, 1 << 24))
    assert not _triton_local_cx_segment_enabled(scheduled, shape=(1, 1 << 20))
    assert not _triton_local_cx_segment_enabled(scheduled, shape=(2, 1 << 24))
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "")
    assert _triton_local_cx_segment_enabled(scheduled, shape=(1, 1 << 24))
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "0")
    assert not _triton_local_cx_segment_enabled(scheduled, shape=(1, 1 << 24))
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "1")
    assert _triton_local_cx_segment_enabled(scheduled, shape=(1, 1 << 20))


def test_local_cx_default_window_and_override(monkeypatch):
    monkeypatch.delenv("FQ_STATEVECTOR_TRITON_LOCAL_CX", raising=False)
    assert _triton_local_cx_requested((1, 1 << 24))
    assert not _triton_local_cx_requested((1, 1 << 20))
    assert not _triton_local_cx_requested((2, 1 << 24))

    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_LOCAL_CX", "1")
    assert _triton_local_cx_requested((1, 1 << 20))
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_LOCAL_CX", "0")
    assert not _triton_local_cx_requested((1, 1 << 24))


def test_local_cx_segment_decision_binds_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "1")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_compiler_provenance",
        lambda: ("triton", "3.7.1", "direct", "resolved"),
    )

    decision = _triton_local_cx_segment_decision(
        device_type="cuda",
        dtype="complex64",
        shape=(1, 1 << 24),
    )

    assert decision.accelerated
    assert decision.semantic_id == "statevector.apply.cnot_sequence.local"
    assert decision.implementation_id == "FQKI-TRITON-SV-003-A"
    assert decision.catalog_mismatches == ()


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_local_cx_segment_decision_reports_catalog_mismatch(
    monkeypatch, device_type, dtype, mismatch
):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "1")

    decision = _triton_local_cx_segment_decision(
        device_type=device_type,
        dtype=dtype,
        shape=(1, 1 << 24),
    )

    assert not decision.accelerated
    assert decision.reason == "input_not_supported"
    assert decision.implementation_id is None
    assert decision.catalog_mismatches == (mismatch,)


def test_transpose_1q_decision_binds_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_TRANSPOSE_1Q", "1")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_compiler_provenance",
        lambda: ("triton", "3.7.1", "direct", "resolved"),
    )

    decision = _triton_transpose_1q_decision(
        device_type="cuda",
        dtype="complex64",
    )

    assert decision.accelerated
    assert decision.semantic_id == "statevector.distributed.transpose_apply_1q"
    assert decision.implementation_id == "FQKI-TRITON-SV-006-A"
    assert decision.catalog_mismatches == ()


def test_control_subspace_transport_decisions_bind_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_compiler_provenance",
        lambda: ("triton", "3.7.1", "direct", "resolved"),
    )

    pack = _triton_control_subspace_pack_decision(
        device_type="cuda",
        dtype="complex64",
    )
    unpack = _triton_control_subspace_unpack_decision(
        device_type="cuda",
        dtype="complex64",
    )

    assert pack.accelerated
    assert pack.semantic_id == "statevector.transport.control_subspace_pack"
    assert pack.implementation_id == "FQKI-TRITON-SV-007-A"
    assert pack.catalog_mismatches == ()
    assert unpack.accelerated
    assert unpack.semantic_id == "statevector.transport.control_subspace_unpack"
    assert unpack.implementation_id == "FQKI-TRITON-SV-008-A"
    assert unpack.catalog_mismatches == ()


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_control_subspace_transport_decisions_report_catalog_mismatch(
    monkeypatch, device_type, dtype, mismatch
):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")

    decisions = (
        _triton_control_subspace_pack_decision(
            device_type=device_type,
            dtype=dtype,
        ),
        _triton_control_subspace_unpack_decision(
            device_type=device_type,
            dtype=dtype,
        ),
    )

    for decision in decisions:
        assert not decision.accelerated
        assert decision.reason == "input_not_supported"
        assert decision.implementation_id is None
        assert decision.catalog_mismatches == (mismatch,)


def test_control_subspace_transport_decisions_preserve_runtime_contract(
    monkeypatch,
):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )

    decisions = (
        _triton_control_subspace_pack_decision(
            runtime_supported=False,
            device_type="cuda",
            dtype="complex64",
        ),
        _triton_control_subspace_unpack_decision(
            runtime_supported=False,
            device_type="cuda",
            dtype="complex64",
        ),
    )

    for decision in decisions:
        assert not decision.accelerated
        assert decision.reason == "input_not_supported"
        assert decision.catalog_mismatches == ()


def test_control_subspace_transport_decisions_report_unavailable_triton(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: False,
    )

    decisions = (
        _triton_control_subspace_pack_decision(
            device_type="cuda",
            dtype="complex64",
        ),
        _triton_control_subspace_unpack_decision(
            device_type="cuda",
            dtype="complex64",
        ),
    )

    for decision in decisions:
        assert not decision.accelerated
        assert decision.reason == "triton_unavailable"
        assert decision.catalog_mismatches == ()


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_transpose_1q_decision_reports_catalog_mismatch(
    monkeypatch, device_type, dtype, mismatch
):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_TRANSPOSE_1Q", "1")

    decision = _triton_transpose_1q_decision(
        device_type=device_type,
        dtype=dtype,
    )

    assert not decision.accelerated
    assert decision.reason == "input_not_supported"
    assert decision.implementation_id is None
    assert decision.catalog_mismatches == (mismatch,)


@pytest.mark.parametrize(
    ("enabled", "runtime_supported", "reason"),
    (("0", True, "disabled_by_policy"), ("1", False, "input_not_supported")),
)
def test_transpose_1q_decision_preserves_runtime_policy(
    monkeypatch, enabled, runtime_supported, reason
):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_TRANSPOSE_1Q", enabled)
    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.kernel_dispatch.triton_available",
        lambda: True,
    )

    decision = _triton_transpose_1q_decision(
        runtime_supported=runtime_supported,
        device_type="cuda",
        dtype="complex64",
    )

    assert not decision.accelerated
    assert decision.reason == reason
    assert decision.catalog_mismatches == ()


def test_cpu_cx_segment_gather_is_single_process_only(monkeypatch):
    import flagquantum.runtime.executors.statevector.forward_sweep as forward_sweep

    monkeypatch.setattr(
        forward_sweep, "_single_process_cpu_direct_enabled", lambda: True
    )
    sweep = object.__new__(_ShardedForwardSweep)
    sweep.local_compilation = True
    sweep.world_size = 2
    sweep.resolved_device = torch.device("cpu")
    sweep.cx_segment_scratch = None

    instruction = Instruction(name="cx", wires=(0, 1))
    assert sweep._cx_segment(0, instruction, touched=False) is None


def test_exchange_workspace_classifies_peers_by_its_placement_not_the_environment(
    monkeypatch,
):
    import flagquantum.runtime.executors.statevector.forward as forward

    # LOCAL_WORLD_SIZE describes the world torchrun was handed. A workspace
    # carrying an explicit placement must not re-derive one from it, or a
    # two-node run reports its cross-network exchanges as intra-node.
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setattr(forward.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(forward.dist, "get_rank", lambda: 9)
    workspace = StatevectorExchangeWorkspace(local_world_size=8, rank=9)

    workspace.record_peer_exchange(10, 64)
    workspace.record_peer_exchange(1, 128)

    assert workspace.intra_node_message_count == 1
    assert workspace.intra_node_bytes == 64
    assert workspace.inter_node_message_count == 1
    assert workspace.inter_node_bytes == 128


def test_exchange_workspace_defaults_to_one_rank_per_host():
    # A bare workspace stands for one rank on one host, so every other global
    # rank is a network hop rather than silently assumed local.
    workspace = StatevectorExchangeWorkspace()

    workspace.record_peer_exchange(1, 64)

    assert workspace.intra_node_message_count == 0
    assert workspace.inter_node_message_count == 1
    assert workspace.inter_node_bytes == 64


@pytest.mark.parametrize("world_size", [3, 5, 6, 12])
def test_a_world_size_the_executor_cannot_shard_is_refused_not_silently_run(
    monkeypatch, world_size: int
):
    """The planner describes an arbitrary shard count; this executor does not run it.

    `contiguous_amplitude_range` leaves `sharded_qubits` empty, so every gate
    resolves to a rank-local kernel and no exchange is issued. Each rank would
    return its own contiguous slice of a gate nobody applied -- wrong
    amplitudes reported as a successful distributed run, and `indexed_all_to_all`
    has no transport here. Refusing is the only honest outcome.
    """
    import flagquantum.runtime.executors.statevector.forward_sweep as sweep_module

    monkeypatch.setattr(sweep_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        sweep_module.dist, "get_world_size", lambda group=None: world_size
    )
    monkeypatch.setattr(sweep_module.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(sweep_module.dist, "get_backend", lambda group=None: "gloo")
    circuit = fq.Circuit(4).h(0).cx(0, 3).ry(2, 0.3)

    with pytest.raises(NotImplementedError) as error:
        _ShardedForwardSweep(circuit, device=torch.device("cpu"))

    message = str(error.value)
    assert f"power-of-two world_size (received {world_size})" in message
    assert "contiguous_amplitude_range" in message


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
        ("local_cx", "disabled_by_policy", 1),
    }
    local_1q_dispatch = next(
        record for record in dispatch["decisions"] if record["feature"] == "local_1q"
    )
    local_cx_dispatch = next(
        record for record in dispatch["decisions"] if record["feature"] == "local_cx"
    )
    assert local_1q_dispatch["device_runtime"] == {
        "provider": "pytorch",
        "device_type": "cpu",
    }
    assert local_1q_dispatch["kernel_compiler"] is None
    assert local_1q_dispatch["kernel_route"] == {
        "semantic_id": "statevector.apply.matrix_1q.local",
        "implementation": "pytorch_eager",
        "integration_path": "pytorch",
        "fallback": False,
        "implementation_id": None,
        "catalog_mismatches": ("device",),
    }
    assert local_cx_dispatch["kernel_route"] == {
        "semantic_id": "statevector.apply.cnot.local",
        "implementation": "pytorch_eager",
        "integration_path": "pytorch",
        "fallback": False,
        "implementation_id": None,
        "catalog_mismatches": ("device",),
    }
    with pytest.raises(FullStateMaterializationError, match="forbidden"):
        result.full_state()


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_1q_triton_execution_records_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_LOCAL_1Q", "1")
    circuit = fq.Circuit(3).h(0)

    result = execute_torch_distributed_statevector(circuit, device="cuda")

    torch.testing.assert_close(result.shard_state.amplitudes.cpu(), circuit.state())
    dispatch = result.summary()["kernel_dispatch"]
    assert dispatch["triton_execution_count"] == 1
    route = dispatch["decisions"][0]["kernel_route"]
    assert route["semantic_id"] == "statevector.apply.matrix_1q.local"
    assert route["implementation_id"] == "FQKI-TRITON-SV-001-A"
    assert route["catalog_mismatches"] == ()


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_cnot_triton_execution_records_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_LOCAL_CX", "1")
    circuit = fq.Circuit(3).x(0).cx(0, 1)

    result = execute_torch_distributed_statevector(circuit, device="cuda")

    torch.testing.assert_close(result.shard_state.amplitudes.cpu(), circuit.state())
    dispatch = result.summary()["kernel_dispatch"]
    assert dispatch["triton_execution_count"] == 1
    route = next(
        record["kernel_route"]
        for record in dispatch["decisions"]
        if record["feature"] == "local_cx"
    )
    assert route["semantic_id"] == "statevector.apply.cnot.local"
    assert route["implementation_id"] == "FQKI-TRITON-SV-002-A"
    assert route["catalog_mismatches"] == ()


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_cnot_sequence_records_catalog_identity(monkeypatch):
    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "1")
    circuit = fq.Circuit(3).x(0).cx(0, 1).cx(1, 2)

    result = execute_torch_distributed_statevector(circuit, device="cuda")

    torch.testing.assert_close(result.shard_state.amplitudes.cpu(), circuit.state())
    record = next(
        record
        for record in result.summary()["kernel_dispatch"]["decisions"]
        if record["feature"] == "local_cx_segment"
    )
    route = record["kernel_route"]
    assert record["count"] == 1
    assert route["semantic_id"] == "statevector.apply.cnot_sequence.local"
    assert route["implementation_id"] == "FQKI-TRITON-SV-003-A"
    assert route["catalog_mismatches"] == ()


def test_single_rank_parameterized_gates_preserve_complex128_precision():
    circuit = (
        fq.Circuit(3, dtype=torch.complex128)
        .h(0)
        .rx(2, theta=0.2)
        .cx(0, 2)
        .rz(1, theta=-0.3)
        .cx(2, 1)
    )

    result = execute_torch_distributed_statevector(
        circuit,
        dtype=torch.complex128,
        persistent_qubit_layout=False,
    )

    torch.testing.assert_close(
        result.shard_state.amplitudes,
        circuit.state(),
        atol=1e-13,
        rtol=1e-13,
    )


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
        instructions=(
            Instruction(name="bit_flip", wires=(0,), params={"probability": 0.5}),
        ),
    )
    initialized = False

    def forbidden(*args, **kwargs):
        nonlocal initialized
        initialized = True
        raise AssertionError("state must not initialize")

    monkeypatch.setattr(
        "flagquantum.runtime.executors.statevector.forward_sweep.initialize_statevector_shard",
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
        params = dict.fromkeys(schema.parameters, 0.17)
        circuit.gate(schema.opcode, tuple(range(schema.arity)), **params)
    result = execute_torch_distributed_statevector(circuit)
    assert torch.allclose(result.shard_state.amplitudes, circuit.state(), atol=1e-5)


def test_pytorch_executor_import_does_not_initialize_optional_jax():
    import subprocess
    import sys

    code = (
        "import sys; "
        "import flagquantum.runtime.executors.statevector.forward_executor; "
        "import flagquantum.runtime.executors.statevector.forward_sweep; "
        "assert 'jax' not in sys.modules and 'jaxlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_communication_aware_layout_moves_inactive_wire_to_rank_bits():
    theta = torch.tensor(0.23, requires_grad=True)
    ir = fq.Circuit(6).ry(5, theta).cx(5, 4).ry(4, theta).to_ir()

    remapped, mapping = communication_aware_qubit_layout(
        ir, world_size=4, preferred_local_qubits=(4,)
    )

    assert mapping[4] < 4 and mapping[5] < 4
    assert {mapping[0], mapping[1]} == {4, 5}
    assert tuple(instruction.wires for instruction in remapped.instructions) == (
        (mapping[5],),
        (mapping[5], mapping[4]),
        (mapping[4],),
    )


def test_training_layout_accounts_for_forward_reverse_and_parameter_work():
    theta = torch.tensor(0.23, requires_grad=True)
    circuit = fq.Circuit(5)
    for _ in range(4):
        circuit.ry(0, theta)
    for _ in range(5):
        circuit.x(1)
    ir = circuit.to_ir()

    _, forward_mapping = communication_aware_qubit_layout(
        ir,
        world_size=2,
        preferred_local_qubits=(2, 3, 4),
        optimization_target="forward",
    )
    remapped, training_mapping = communication_aware_qubit_layout(
        ir,
        world_size=2,
        preferred_local_qubits=(2, 3, 4),
        optimization_target="training_step",
    )

    assert forward_mapping[0] == 4
    assert training_mapping[1] == 4
    assert (
        remapped.metadata["statevector_layout_optimization_target"] == "training_step"
    )


def test_communication_aware_layout_rejects_unknown_optimization_target():
    with pytest.raises(ValueError, match="forward or training_step"):
        communication_aware_qubit_layout(
            fq.Circuit(3).to_ir(),
            world_size=2,
            optimization_target="backward_only",
        )


def test_repeat_layout_reuses_the_remapped_program(monkeypatch):
    """A repeat execution must not re-validate the parameters it already has.

    Remapping rebuilds every instruction, and rebuilding one re-validates its
    parameters. Validation of an accelerator-resident angle decides finiteness by
    reading the value back, so rebuilding the program inside the execution region
    synchronized the device once per parameterized gate on every execution. The
    count is what makes that regression observable: an execution that rebuilds
    the program calls the validator again.
    """

    from flagquantum.core import ir as ir_module
    from flagquantum.runtime.executors.statevector import program_cache

    program_cache._REMAPPED_PROGRAM_CACHE.clear()
    validated = 0
    original = ir_module._normalize_angle

    def counting_normalize_angle(value, *, opcode, parameter):
        nonlocal validated
        validated += 1
        return original(value, opcode=opcode, parameter=parameter)

    monkeypatch.setattr(ir_module, "_normalize_angle", counting_normalize_angle)
    theta = torch.tensor(0.23, requires_grad=True)
    ir = fq.Circuit(5).h(0).ry(4, theta).cx(4, 1).rx(3, theta).cx(0, 4).to_ir()

    first, first_mapping = communication_aware_qubit_layout(
        ir, world_size=2, local_world_size=1
    )
    after_first = validated
    second, second_mapping = communication_aware_qubit_layout(
        ir, world_size=2, local_world_size=1
    )

    assert first_mapping == second_mapping and first_mapping != (0, 1, 2, 3, 4)
    assert after_first == 4, "the first layout validates the two bound parameters twice"
    assert validated == after_first, "a repeat layout re-validated the parameters"
    assert tuple(instruction.wires for instruction in second.instructions) == tuple(
        instruction.wires for instruction in first.instructions
    )


def test_layout_does_not_reuse_another_circuits_parameters():
    """A cached program carries its own parameters, not the new circuit's."""

    from flagquantum.runtime.executors.statevector import program_cache

    program_cache._REMAPPED_PROGRAM_CACHE.clear()
    first = fq.Circuit(5).h(0).ry(4, 0.31).cx(4, 1).rx(3, -0.27).cx(0, 4).to_ir()
    second = fq.Circuit(5).h(0).ry(4, 0.77).cx(4, 1).rx(3, 0.11).cx(0, 4).to_ir()

    remapped_first, _ = communication_aware_qubit_layout(
        first, world_size=2, local_world_size=1
    )
    remapped_second, _ = communication_aware_qubit_layout(
        second, world_size=2, local_world_size=1
    )

    def angles(program):
        return [
            instruction.params["theta"]
            for instruction in program.instructions
            if "theta" in instruction.params
        ]

    assert angles(remapped_first) == [0.31, -0.27]
    assert angles(remapped_second) == [0.77, 0.11]


def test_instruction_relabelling_does_not_revalidate_its_parameters(monkeypatch):
    """The persistent layout relabels per step, and must not read back per step.

    The sweep relabels each instruction against the mapping it currently holds.
    Rebuilding an instruction re-validates its parameters, and validating an
    accelerator-resident angle decides finiteness by reading the value back, so a
    rebuild inside the execution region is a device synchronization in the
    measured path. The validator's call count is what makes that visible.
    """

    from flagquantum.core import ir as ir_module
    from flagquantum.runtime.executors.statevector import program_cache

    program_cache._REMAPPED_INSTRUCTION_CACHE.clear()
    validated = 0
    original = ir_module._normalize_angle

    def counting_normalize_angle(value, *, opcode, parameter):
        nonlocal validated
        validated += 1
        return original(value, opcode=opcode, parameter=parameter)

    monkeypatch.setattr(ir_module, "_normalize_angle", counting_normalize_angle)
    instruction = fq.Circuit(4).ry(2, 0.4).to_ir().instructions[0]
    mapping = [0, 1, 3, 2]
    before = validated

    once = remap_instruction_qubits(instruction, mapping)
    after_first = validated
    twice = remap_instruction_qubits(instruction, mapping)

    assert tuple(once.wires) == (3,)
    assert once is twice, "the relabelled instruction was rebuilt a second time"
    assert after_first - before == 1
    assert validated == after_first
    # A mapping that leaves the wires alone must not rebuild at all.
    identity = remap_instruction_qubits(instruction, [0, 1, 2, 3])
    assert identity is instruction
    assert validated == after_first


def test_instruction_relabelling_keeps_cached_entries_distinct():
    """Two instructions that relabel to the same wires stay separate.

    The cache is keyed on the source instruction, so an entry must never be
    handed to a different instruction that happens to share its destination
    wires and its id -- which is only impossible while the source is alive.
    """

    from flagquantum.runtime.executors.statevector import program_cache

    program_cache._REMAPPED_INSTRUCTION_CACHE.clear()
    first = fq.Circuit(4).ry(2, 0.4).to_ir().instructions[0]
    second = fq.Circuit(4).rx(3, -0.9).to_ir().instructions[0]

    remapped_first = remap_instruction_qubits(first, [0, 1, 3, 2])
    remapped_second = remap_instruction_qubits(second, [1, 0, 3, 2])

    assert remapped_first is not remapped_second
    assert remapped_first.name == "ry" and remapped_second.name == "rx"
    assert remapped_first.params != remapped_second.params
    assert tuple(remapped_first.wires) == (3,)
    assert tuple(remapped_second.wires) == (2,)


def test_communication_aware_layout_penalizes_full_shard_subgroup_alignment():
    circuit = fq.Circuit(6)
    for wire in range(6):
        circuit.ry(wire, torch.tensor(0.1 + wire * 0.01, requires_grad=True))
    for wire in range(5):
        circuit.cx(wire, wire + 1)

    _, mapping = communication_aware_qubit_layout(
        circuit.to_ir(), world_size=2, preferred_local_qubits=(3,)
    )

    # Both endpoints have the same gate-touch score.  Sharding wire 0 would
    # map its CX partner to the highest local address bit and force subgroup
    # exchange to align every chunk to the complete shard.  Wire 5 keeps the
    # partner on the lowest local bit and therefore preserves bounded chunks.
    assert mapping == tuple(range(6))


def test_multi_node_rank_bit_order_puts_low_activity_wire_on_node_bit(monkeypatch):
    circuit = fq.Circuit(28)
    for wire in range(28):
        circuit.ry(wire, torch.tensor(0.1 + wire * 0.001, requires_grad=True))
    for wire in range(27):
        circuit.cx(wire, wire + 1)

    _, mapping = communication_aware_qubit_layout(
        circuit.to_ir(),
        world_size=16,
        local_world_size=8,
        preferred_local_qubits=(14,),
    )

    assert mapping[27] == 24
    assert {mapping[24], mapping[25], mapping[26]} == {25, 26, 27}

    monkeypatch.setenv("FQ_STATEVECTOR_TOPOLOGY_AWARE_RANK_BITS", "0")
    _, canonical_rank_bits = communication_aware_qubit_layout(
        circuit.to_ir(),
        world_size=16,
        local_world_size=8,
        preferred_local_qubits=(14,),
    )
    assert canonical_rank_bits[24] == 24
    assert canonical_rank_bits[27] == 27


def test_the_layout_placement_comes_from_the_caller_not_the_environment(monkeypatch):
    # `LOCAL_WORLD_SIZE` describes the world torchrun was handed, which for a
    # subgroup run is not the placement the ranks are actually in. A caller that
    # states the placement gets that one, and an unlaunched process gets the
    # placement-free single-node answer rather than whatever is exported around
    # it.
    circuit = fq.Circuit(28)
    for wire in range(28):
        circuit.ry(wire, torch.tensor(0.1 + wire * 0.001, requires_grad=True))
    for wire in range(27):
        circuit.cx(wire, wire + 1)
    ir = circuit.to_ir()
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")

    _, stated = communication_aware_qubit_layout(
        ir, world_size=16, local_world_size=8, preferred_local_qubits=(14,)
    )
    _, from_environment = communication_aware_qubit_layout(
        ir, world_size=16, preferred_local_qubits=(14,)
    )

    assert stated[27] == 24
    assert from_environment != stated


def test_a_placement_that_is_not_a_rank_count_is_refused():
    with pytest.raises(ValueError, match="positive rank count"):
        communication_aware_qubit_layout(
            fq.Circuit(4).to_ir(), world_size=4, local_world_size=0
        )


def test_single_rank_communication_aware_result_declares_basis_order():
    result = execute_torch_distributed_statevector(
        fq.Circuit(3).h(2), qubit_layout="communication_aware"
    )
    assert result.logical_to_physical_qubits == (0, 1, 2)
    assert result.summary()["amplitude_basis_order"] == "canonical_logical"


class _FakeCollective:
    """A request that reports the wait the executor is required to perform."""

    def __init__(self):
        self.wait_calls = 0
        self.block_current_stream_calls = 0

    def wait(self):
        self.wait_calls += 1

    def block_current_stream(self):
        self.block_current_stream_calls += 1


def _pair_exchange_plan(world_size: int, local_amplitudes: int) -> SimpleNamespace:
    """Enough of a plan for the pair-exchange path, and nothing else."""

    shard = StatevectorShard(
        rank=0,
        world_size=world_size,
        amplitude_start=0,
        amplitude_end=local_amplitudes,
        local_amplitudes=local_amplitudes,
        local_state_bytes=local_amplitudes * 8,
    )
    return SimpleNamespace(
        sharded_qubits=(3,),
        world_size=world_size,
        shards=(shard,),
        n_qubits=4,
        rank_address_bits=1,
    )


def _pair_exchange_state(plan: SimpleNamespace, amplitudes: torch.Tensor):
    return StatevectorShardState(
        rank=0,
        shard=plan.shards[0],
        amplitudes=amplitudes,
        global_indices=torch.empty(0, dtype=torch.long),
    )


def test_cross_shard_cx_records_each_control_subspace_fallback_invocation(
    monkeypatch,
):
    plan = _pair_exchange_plan(world_size=2, local_amplitudes=8)
    amplitudes = torch.arange(8, dtype=torch.float32).to(torch.complex64).reshape(1, 8)
    evidence = KernelDispatchEvidence()

    class _FakeP2POp:
        def __init__(self, operation, tensor, peer, group=None, tag=0):
            self.operation = operation
            self.tensor = tensor
            self.peer = peer
            self.group = group
            self.tag = tag

    def batch_isend_irecv(operations):
        operations[1].tensor.copy_(operations[0].tensor)
        return (_FakeCollective(), _FakeCollective())

    monkeypatch.setattr(torch.distributed, "P2POp", _FakeP2POp)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", batch_isend_irecv)

    _, communication_count, _, _ = _vectorized_cross_shard_cx(
        _pair_exchange_state(plan, amplitudes),
        (0, 3),
        plan=plan,
        chunk_amplitudes=2,
        kernel_dispatch_evidence=evidence,
    )

    assert communication_count == 2
    records = {record["feature"]: record for record in evidence.summary()["decisions"]}
    assert records["control_subspace_pack"]["count"] == 2
    assert records["control_subspace_unpack"]["count"] == 2
    for record in records.values():
        assert record["selected"] == "pytorch"
        assert record["reason"] == "input_not_supported"
        assert record["kernel_route"]["implementation_id"] is None
        assert record["kernel_route"]["catalog_mismatches"] == ("device",)


# The peak the pair-exchange path reports is an exact sum of four live buffers:
# the rank-local output buffer, the chunk when it does not alias the input, the
# exchange buffer, and the numeric temporaries of one combine. Which exchange
# buffer it is depends on the path, and that is the only difference between the
# two cases below -- so the two constants are asserted where they are true
# rather than at a call site that runs at several rank counts.
#
# These exist because the harness that runs on the accelerator could only assert
# a bound that holds everywhere, and a bound that holds everywhere does not
# notice a dropped term. Here the accounting is driven directly, one chunk at a
# time, so the identity is exact.
@pytest.mark.parametrize("with_workspace", [False, True])
def test_two_rank_pair_exchange_counts_both_ranks_gathered_blocks(
    monkeypatch, with_workspace: bool
):
    plan = _pair_exchange_plan(world_size=2, local_amplitudes=8)
    amplitudes = torch.zeros((1, 8), dtype=torch.complex64)
    per_rank = plan.shards[0].local_state_bytes
    requests = []
    workspace = StatevectorExchangeWorkspace() if with_workspace else None

    def all_gather(gathered, local, *, group=None, async_op=False):
        rows = local.shape[0]
        gathered[:rows] = local
        gathered[rows:] = local
        request = _FakeCollective()
        requests.append(request)
        return request

    # `all_gather_single` is preferred when the build provides it, so patching
    # that name is what pins which collective the path takes.
    monkeypatch.setattr(
        torch.distributed, "all_gather_single", all_gather, raising=False
    )

    _, _, _, peak = _vectorized_pair_exchange_gate(
        _pair_exchange_state(plan, amplitudes),
        torch.eye(2, dtype=torch.complex64),
        3,
        plan=plan,
        chunk_amplitudes=8,
        process_group=None,
        workspace=workspace,
        pipeline=False,
    )

    assert requests and requests[0].wait_calls == 1
    # output 1x + gathered 2x + numeric 3x
    assert peak == 6 * per_rank
    assert peak > per_rank
    if workspace is not None:
        # Both ranks' blocks are held at once. A workspace sized for one block
        # would make the peak smaller, and the peak is what a capacity report
        # reads.
        assert workspace.reserved_bytes == 2 * per_rank


@pytest.mark.parametrize("with_workspace", [False, True])
def test_peer_to_peer_pair_exchange_counts_one_peer_buffer(
    monkeypatch, with_workspace: bool
):
    plan = _pair_exchange_plan(world_size=4, local_amplitudes=4)
    amplitudes = torch.zeros((1, 4), dtype=torch.complex64)
    per_rank = plan.shards[0].local_state_bytes
    requests = []
    workspace = StatevectorExchangeWorkspace() if with_workspace else None

    class _FakeP2POp:
        def __init__(self, operation, tensor, peer, group=None, tag=0):
            self.operation = operation
            self.tensor = tensor
            self.peer = peer
            self.tag = tag

    def batch_isend_irecv(operations):
        sent, received = operations
        received.tensor.copy_(sent.tensor)
        request = _FakeCollective()
        requests.append(request)
        return [request]

    monkeypatch.setattr(torch.distributed, "P2POp", _FakeP2POp)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", batch_isend_irecv)

    _, _, _, peak = _vectorized_pair_exchange_gate(
        _pair_exchange_state(plan, amplitudes),
        torch.eye(2, dtype=torch.complex64),
        3,
        plan=plan,
        chunk_amplitudes=4,
        process_group=None,
        workspace=workspace,
        pipeline=False,
    )

    assert requests and requests[0].wait_calls == 1
    # output 1x + one peer buffer 1x + numeric 3x. The gathered path above is
    # one buffer larger, which is the whole of the difference between 6x and 5x.
    assert peak == 5 * per_rank
    assert peak > per_rank
    if workspace is not None:
        assert workspace.reserved_bytes == per_rank


# A rank-local gate reads and writes exactly the amplitudes it owns, and the
# sweep walks that set in disjoint chunks, so the state buffer can be its own
# output buffer. It must only be invited to do so when nothing else needs the
# pre-gate amplitudes, because the backward pass of a training program does.
# These three tests pin the invitation, its effect on the buffer, and the
# decision that withholds it.
def test_rank_local_gate_writes_into_the_caller_buffer_when_invited():
    plan = _pair_exchange_plan(world_size=2, local_amplitudes=8)
    amplitudes = torch.arange(8, dtype=torch.float32).to(torch.complex64).reshape(1, 8)
    # An X on wire 0 pairs each local address with the one that differs in that
    # wire's bit. The bit is taken from the plan rather than written out, so the
    # expectation follows the addressing instead of restating it.
    swap = 1 << (plan.n_qubits - 1 - 0 - len(plan.sharded_qubits))
    indices = torch.arange(amplitudes.shape[1])
    expected = amplitudes[:, indices ^ swap].clone()

    returned, _ = _vectorized_local_gate(
        _pair_exchange_state(plan, amplitudes),
        torch.tensor(((0, 1), (1, 0)), dtype=torch.complex64),
        (0,),
        plan=plan,
        chunk_amplitudes=2,
        output=amplitudes,
    )

    assert returned.amplitudes is amplitudes
    assert torch.equal(amplitudes, expected)


def test_rank_local_gate_keeps_the_caller_buffer_without_an_invitation():
    plan = _pair_exchange_plan(world_size=2, local_amplitudes=8)
    amplitudes = torch.arange(8, dtype=torch.float32).to(torch.complex64).reshape(1, 8)
    untouched = amplitudes.clone()

    returned, _ = _vectorized_local_gate(
        _pair_exchange_state(plan, amplitudes),
        torch.tensor(((0, 1), (1, 0)), dtype=torch.complex64),
        (0,),
        plan=plan,
        chunk_amplitudes=2,
    )

    assert returned.amplitudes is not amplitudes
    assert torch.equal(amplitudes, untouched)
    assert not torch.equal(returned.amplitudes, untouched)


def test_a_program_that_trains_keeps_a_separate_output_buffer():
    """The decision follows the gradient requirement, not the rank count."""

    parameter = torch.tensor(0.43, requires_grad=True)
    trained = _ShardedForwardSweep(
        fq.Circuit(3).h(0).cx(0, 2).ry(1, parameter), device=torch.device("cpu")
    )
    inferred = _ShardedForwardSweep(
        fq.Circuit(3).h(0).cx(0, 2).ry(1, 0.43), device=torch.device("cpu")
    )

    assert any(matrix.requires_grad for matrix in trained.matrices)
    assert trained.local_output_in_place is False
    assert inferred.local_output_in_place is True


@pytest.mark.parametrize(
    ("shape", "expected"),
    (
        ((1, 1 << 10), True),
        ((1, 1 << 16), True),
        ((1, 1 << 20), True),
        ((1, 1 << 24), True),
        ((1, 1 << 12), False),
        ((2, 1 << 20), False),
    ),
)
def test_local_1q_default_window_and_override(monkeypatch, shape, expected):
    monkeypatch.delenv("FQ_STATEVECTOR_TRITON_LOCAL_1Q", raising=False)
    assert _triton_local_1q_requested(shape) is expected

    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_LOCAL_1Q", "0")
    assert not _triton_local_1q_requested(shape)

    monkeypatch.setenv("FQ_STATEVECTOR_TRITON_LOCAL_1Q", "1")
    assert _triton_local_1q_requested(shape)
