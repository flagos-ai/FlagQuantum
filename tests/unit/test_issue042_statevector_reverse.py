"""Single-rank reverse-mode and custom-autograd contracts."""

import math
from types import SimpleNamespace

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.backends.statevector import plan_distributed_statevector
from flagquantum.runtime.backends.statevector.gradient_reduction import (
    AsyncGradientReducer,
)
from flagquantum.runtime.backends.statevector.local_execution import (
    _rank_global_indices,
    use_compact_global_indices,
)
from flagquantum.runtime.backends.statevector.reverse import (
    BackwardExecutionEvidence,
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
    resolve_checkpoint_policy,
)
from flagquantum.runtime.backends.statevector.reverse_adjoint import (
    _compact_reverse_global_indices,
)

pytestmark = pytest.mark.unit


def test_address_sharded_reverse_always_uses_compact_global_indices():
    small_shard = SimpleNamespace(local_amplitudes=1 << 23)
    address_sharded = SimpleNamespace(
        distribution="qubit_address_sharded", shards=(small_shard,)
    )
    range_partitioned = SimpleNamespace(
        distribution="range_partitioned", shards=(small_shard,)
    )

    assert _compact_reverse_global_indices(address_sharded, 0) is True
    assert _compact_reverse_global_indices(range_partitioned, 0) is False


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_address_sharded_indices_are_built_from_local_ordinal(world_size):
    plan = plan_distributed_statevector(fq.Circuit(8), world_size=world_size)
    rank_bits = len(plan.sharded_wires)

    for rank in range(world_size):
        indices = _rank_global_indices(plan, rank, device=torch.device("cpu"))
        expected = (
            torch.arange(plan.shards[rank].local_amplitudes, dtype=torch.long)
            << rank_bits
        ) | rank
        torch.testing.assert_close(indices, expected)
        assert indices.numel() == plan.shards[rank].local_amplitudes


def test_compact_index_policy_is_shared_by_forward_and_reverse():
    plan = plan_distributed_statevector(fq.Circuit(8), world_size=4)

    for rank in range(plan.world_size):
        assert use_compact_global_indices(plan, rank)
        assert _compact_reverse_global_indices(plan, rank)


def test_parameter_gradients_are_all_reduced_in_dtype_packed_buckets(monkeypatch):
    calls = []

    class CompletedWork:
        def wait(self):
            return True

    def fake_all_reduce(tensor, **kwargs):
        calls.append(tensor.clone())
        tensor.add_(10)
        assert kwargs["async_op"] is True
        return CompletedWork()

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    gradients = [
        torch.tensor(1.0),
        torch.tensor(2.0),
        torch.tensor(3.0, dtype=torch.float64),
    ]
    evidence = BackwardExecutionEvidence()

    reducer = AsyncGradientReducer(
        gradients,
        process_group=None,
        evidence=evidence,
        max_parameters=len(gradients),
        max_bytes=sum(
            gradient.numel() * gradient.element_size() for gradient in gradients
        ),
    )
    for index in range(len(gradients)):
        reducer.mark_ready(index, overlap_opportunity=False)
    reducer.finish()

    assert len(calls) == 2
    torch.testing.assert_close(calls[0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(calls[1], torch.tensor([3.0], dtype=torch.float64))
    assert [gradient.item() for gradient in gradients] == [11.0, 12.0, 13.0]
    assert evidence.communication_count == 2
    assert evidence.async_gradient_collective_count == 2
    assert evidence.communication_bytes == 16
    assert evidence.gradient_collective_count == 2
    assert evidence.gradient_collective_bytes == 16


def test_ready_gradient_bucket_launches_before_adjoint_completion(monkeypatch):
    waits = []

    class PendingWork:
        def wait(self):
            waits.append(True)

    def fake_all_reduce(tensor, **kwargs):
        tensor.mul_(2)
        return PendingWork()

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    gradients = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    evidence = BackwardExecutionEvidence()
    reducer = AsyncGradientReducer(
        gradients,
        process_group=None,
        evidence=evidence,
        max_parameters=2,
        max_bytes=1 << 20,
    )

    reducer.mark_ready(0)
    assert evidence.gradient_collective_count == 0
    reducer.mark_ready(1)
    assert evidence.gradient_collective_count == 1
    assert evidence.overlapped_gradient_collective_count == 1
    assert waits == []

    reducer.mark_ready(2)
    reducer.finish()

    assert waits == [True, True]
    assert [gradient.item() for gradient in gradients] == [2.0, 4.0, 6.0]
    assert evidence.async_gradient_collective_count == 2
    assert evidence.communication_bytes == 12
    assert evidence.gradient_collective_count == 2
    assert evidence.gradient_collective_bytes == 12


def test_synchronous_gradient_reduction_has_no_overlap_evidence(monkeypatch):
    calls = []

    def fake_all_reduce(tensor, **kwargs):
        calls.append(kwargs["async_op"])
        tensor.mul_(2)
        return None

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    gradients = [torch.tensor(1.0), torch.tensor(2.0)]
    evidence = BackwardExecutionEvidence()
    reducer = AsyncGradientReducer(
        gradients,
        process_group=None,
        evidence=evidence,
        max_parameters=2,
        max_bytes=1 << 20,
        async_op=False,
    )
    reducer.mark_ready(0)
    reducer.mark_ready(1)
    reducer.finish()

    assert calls == [False]
    assert [gradient.item() for gradient in gradients] == [2.0, 4.0]
    assert evidence.async_gradient_collective_count == 0
    assert evidence.overlapped_gradient_collective_count == 0


def test_multi_layer_multi_parameter_gradients_match_dense_autograd():
    theta = torch.tensor(0.23, requires_grad=True)
    phi = torch.tensor(-0.37, requires_grad=True)
    circuit = fq.Circuit(3).h(0).ry(2, theta).rxx(0, 2, phi).rz(1, theta)
    dense = circuit.expectation_z(2)
    dense_grad = torch.autograd.grad(dense, (theta, phi), retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(circuit, observable_wire=2)
    pending = result.summary()
    assert pending["parameter_gradient_ready"] is False
    assert pending["backward_status"] == "pending"
    assert pending["gradient_reduction"] == "pending"
    assert pending["backward_distribution_semantics"] == "pending"
    assert pending["local_world_size"] == 1
    assert pending["node_count"] == 1
    assert pending["layout_optimization_target"] == "training_step"
    assert len(pending["logical_to_physical_wires"]) == 3
    assert pending["local_state_bytes"] > 0
    assert pending["backward_communication_count"] == 0
    assert pending["backward_communication_bytes"] == 0
    assert pending["peak_backward_scratch_bytes"] == 0
    assert all(item["owner_rank"] is None for item in pending["parameter_ownership"])
    assert all(
        not item["participating_ranks"] for item in pending["parameter_ownership"]
    )
    result.backward()
    assert float(theta.grad) == pytest.approx(float(dense_grad[0]), abs=2e-5)
    assert float(phi.grad) == pytest.approx(float(dense_grad[1]), abs=2e-5)
    assert result.ownership[0].occurrence_count == 2
    completed = result.summary()
    assert completed["backward_uses_full_state_replay"] is False
    assert completed["saved_forward_state_reused"] is True
    assert completed["checkpoint_policy"]["strategy"] == "reversible_adjoint"
    assert (
        completed["checkpoint_policy"]["selection_reason"]
        == "forward_state_reuse_fits_budget"
    )
    assert completed["parameter_gradient_ready"] is True
    assert completed["backward_status"] == "completed"
    assert completed["gradient_distribution"] == "local"
    assert completed["backward_distribution_semantics"] == "single_device_fast_path"
    assert completed["peak_backward_scratch_bytes"] > 0
    assert completed["parameter_ownership"][0]["participating_ranks"] == (0,)
    dispatch = completed["kernel_dispatch"]
    assert dispatch["triton_execution_count"] == 0
    assert dispatch["pytorch_fallback_count"] == 3
    assert {
        (record["feature"], record["reason"]) for record in dispatch["decisions"]
    } == {("vjp_adjoint", "disabled_by_policy")}


def test_parameter_shift_and_finite_difference_match_native_vjp():
    theta = torch.tensor(0.41, requires_grad=True)
    circuit = fq.Circuit(1).ry(0, theta)
    result = execute_torch_distributed_statevector_reverse(circuit)
    result.backward()
    parameter_shift = (math.cos(0.41 + math.pi / 2) - math.cos(0.41 - math.pi / 2)) / 2
    epsilon = 1e-4
    finite_difference = (math.cos(0.41 + epsilon) - math.cos(0.41 - epsilon)) / (
        2 * epsilon
    )
    assert float(theta.grad) == pytest.approx(parameter_shift, abs=1e-5)
    assert float(theta.grad) == pytest.approx(finite_difference, abs=2e-4)


def test_checkpoint_policy_is_versioned_and_fail_closed():
    assert StatevectorCheckpointPolicy().strategy == "auto"
    assert StatevectorCheckpointPolicy().version == "statevector_checkpoint_v2"
    assert StatevectorCheckpointPolicy(strategy="interval", interval=8).interval == 8
    with pytest.raises(ValueError, match="interval > 0"):
        StatevectorCheckpointPolicy(strategy="interval", interval=0)


def test_auto_checkpoint_reuses_forward_state_when_working_set_fits():
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(memory_budget_bytes=4096),
        local_state_bytes=1024,
        device=torch.device("cpu"),
    )

    assert policy.strategy == "reversible_adjoint"
    assert policy.selection_reason == "forward_state_reuse_fits_budget"
    assert policy.estimated_required_bytes == 4096


def test_auto_checkpoint_fails_over_to_rematerialization_when_budget_is_tight():
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(memory_budget_bytes=4095),
        local_state_bytes=1024,
        device=torch.device("cpu"),
    )

    assert policy.strategy == "full_rematerialization"
    assert policy.selection_reason == "reversible_working_set_exceeds_budget"
    assert policy.estimated_required_bytes == 4096


def test_auto_checkpoint_uses_platform_memory_snapshot(monkeypatch):
    platform = SimpleNamespace(
        is_available=lambda: True,
        memory_snapshot=lambda device: SimpleNamespace(free_bytes=10_000),
    )
    monkeypatch.setattr(
        "flagquantum.runtime.backends.statevector.checkpointing.get_platform_runtime",
        lambda device_type: platform,
    )

    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(),
        local_state_bytes=2_000,
        device=torch.device("cuda:0"),
    )

    assert policy.memory_budget_bytes == 7_000
    assert policy.strategy == "full_rematerialization"


def test_explicit_checkpoint_strategy_has_priority_over_auto_budget():
    policy = resolve_checkpoint_policy(
        StatevectorCheckpointPolicy(
            strategy="full_rematerialization", memory_budget_bytes=1
        ),
        local_state_bytes=1024,
        device=torch.device("cpu"),
    )

    assert policy.strategy == "full_rematerialization"
    assert policy.selection_reason == "explicit_strategy"


def test_reverse_exchange_chunk_is_configurable_and_audited(monkeypatch):
    monkeypatch.setenv("FQ_STATEVECTOR_REVERSE_CHUNK_AMPLITUDES", str(1 << 20))
    theta = torch.tensor(0.19, requires_grad=True)
    result = execute_torch_distributed_statevector_reverse(fq.Circuit(1).ry(0, theta))
    result.backward()
    summary = result.summary()
    assert summary["backward_exchange_chunk_amplitudes"] == 1 << 20
    assert summary["backward_exchange_chunk_bytes"] == 8 << 20

    monkeypatch.setenv("FQ_STATEVECTOR_REVERSE_CHUNK_AMPLITUDES", "1000")
    with pytest.raises(ValueError, match="positive power of two"):
        execute_torch_distributed_statevector_reverse(
            fq.Circuit(1).ry(0, torch.tensor(0.2, requires_grad=True))
        )


def test_reversible_adjoint_matches_dense_autograd_on_all_active_wires():
    parameters = tuple(
        torch.tensor(0.11 + 0.07 * wire, requires_grad=True) for wire in range(4)
    )
    circuit = fq.Circuit(4)
    for wire, parameter in enumerate(parameters):
        circuit.ry(wire, parameter)
    for wire in range(3):
        circuit.cx(wire, wire + 1)
    circuit.cx(3, 0)
    dense = circuit.expectation_z(2)
    expected = torch.autograd.grad(dense, parameters, retain_graph=True)

    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_wire=2,
        checkpoint_policy=StatevectorCheckpointPolicy(strategy="reversible_adjoint"),
    )
    result.backward()

    torch.testing.assert_close(result.value, dense.squeeze(), atol=2e-5, rtol=2e-5)
    for parameter, reference in zip(parameters, expected):
        torch.testing.assert_close(parameter.grad, reference, atol=3e-5, rtol=3e-5)
    assert result.summary()["saved_forward_state_reused"] is True


def test_reverse_executor_import_does_not_initialize_jax():
    import subprocess
    import sys

    code = (
        "import sys; import flagquantum.runtime.backends.statevector.reverse; "
        "assert 'jax' not in sys.modules and 'jaxlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_custom_autograd_boundary_passes_double_precision_gradcheck():
    theta = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)

    def expectation(value):
        circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, value)
        return execute_torch_distributed_statevector_reverse(circuit).value

    assert torch.autograd.gradcheck(
        expectation, (theta,), eps=1e-6, atol=1e-5, rtol=1e-4
    )


def test_reverse_memory_debug_uses_active_platform(monkeypatch, capsys):
    monkeypatch.setenv("FQ_STATEVECTOR_DEBUG_MEMORY", "1")
    theta = torch.tensor(0.19, requires_grad=True)
    execute_torch_distributed_statevector_reverse(fq.Circuit(1).ry(0, theta)).backward()
    assert "reversible_adjoint_initialized" in capsys.readouterr().out


def test_backward_failure_is_reported_and_never_marks_gradient_ready(monkeypatch):
    import flagquantum.runtime.backends.statevector.reverse_adjoint as reverse_adjoint

    theta = torch.tensor(0.19, requires_grad=True)
    result = execute_torch_distributed_statevector_reverse(fq.Circuit(1).ry(0, theta))

    def fail(*args, **kwargs):
        raise RuntimeError("injected backward failure")

    monkeypatch.setattr(reverse_adjoint, "_explicit_sharded_adjoint", fail)
    with pytest.raises(RuntimeError, match="injected backward failure"):
        result.backward()
    summary = result.summary()
    assert summary["backward_status"] == "failed"
    assert summary["backward_distribution_semantics"] == "failed"
    assert summary["parameter_gradient_ready"] is False
    assert "injected backward failure" in summary["backward_error"]
    assert result.ownership[0].distribution == "failed"
