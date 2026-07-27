from __future__ import annotations

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit


def input_builder(parameters: torch.Tensor, inputs: torch.Tensor | None) -> fq.Circuit:
    assert inputs is not None
    batch = inputs.shape[0]
    return fq.Circuit(2, bsz=batch).rx(0, inputs).ry(1, parameters[0]).cx(0, 1)


def parameter_batch_builder(parameters: torch.Tensor) -> fq.Circuit:
    batch = parameters.shape[0] if parameters.ndim == 2 else 1
    theta = parameters[:, 0] if parameters.ndim == 2 else parameters[0]
    phi = parameters[:, 1] if parameters.ndim == 2 else parameters[1]
    return fq.Circuit(2, bsz=batch).ry(0, theta).cx(0, 1).rx(1, phi)


def test_input_batch_values_and_gradients_match_per_example() -> None:
    module = fq.Module(input_builder, 1, init=torch.tensor([0.31]))
    inputs = torch.tensor([-0.4, 0.1, 0.7], requires_grad=True)
    batched = module(inputs)
    batched_grad = torch.autograd.grad(
        batched.sum(), (inputs, module.parameters_tensor)
    )
    references = []
    input_grads = []
    parameter_grad = torch.zeros_like(module.parameters_tensor)
    for index in range(inputs.numel()):
        sample = inputs[index].detach().reshape(1).requires_grad_(True)
        value = module(sample)
        sample_grad, shared_grad = torch.autograd.grad(
            value.sum(), (sample, module.parameters_tensor)
        )
        references.append(value.detach())
        input_grads.append(sample_grad)
        parameter_grad += shared_grad
    torch.testing.assert_close(batched, torch.cat(references))
    torch.testing.assert_close(batched_grad[0], torch.cat(input_grads))
    torch.testing.assert_close(batched_grad[1], parameter_grad)


def test_parameter_batch_and_observable_batch_are_explicit() -> None:
    module = fq.Module(parameter_batch_builder, 2)
    parameters = torch.tensor([[0.1, 0.2], [-0.3, 0.5]], requires_grad=True)
    result = module.execute(parameters=parameters)
    references = torch.cat([module.execute(parameters=row).value for row in parameters])
    torch.testing.assert_close(result.value, references)
    assert result.metrics["parameter_batch_size"] == 2
    assert result.metrics["observable_count"] == 1


def test_hamiltonian_grouping_and_batched_evaluation() -> None:
    hamiltonian = fq.Hamiltonian(
        (
            fq.pauli_term(0.5, "Z", (0,)),
            fq.pauli_term(-0.2, "ZZ", (0, 1)),
            fq.pauli_term(0.3, "X", (0,)),
        )
    )
    module = fq.Module(
        parameter_batch_builder,
        (2, 2),
        init=torch.tensor([[0.1, 0.2], [-0.3, 0.5]]),
        hamiltonian=hamiltonian,
        policy=fq.RuntimePolicy(observable="hamiltonian"),
    )
    result = module.execute()
    circuit = parameter_batch_builder(module.parameters_tensor)
    torch.testing.assert_close(result.value, hamiltonian.expectation(circuit))
    assert result.metrics["observable_count"] == 3
    assert result.metrics["observable_group_count"] == 2


def test_hybrid_parallel_plan_has_orthogonal_groups_and_fail_closed_model_parallel() -> (
    None
):
    plan = fq.plan_hybrid_parallel(
        world_size=4,
        data_parallel_size=2,
        state_parallel_size=2,
        input_batch_size=8,
        parameter_batch_size=1,
        observable_count=5,
        observable_group_count=3,
        global_quantum_state_bytes=1024,
        parameter_bytes=64,
        input_bytes=128,
        observable_bytes=40,
    )
    assert plan.state_groups == ((0, 1), (2, 3))
    assert plan.data_groups == ((0, 2), (1, 3))
    assert plan.distribution_semantics == "sharded_across_ranks"
    assert plan.summary()["parallel_dimensions"]["data_parallel"] == 2
    assert plan.summary()["memory_accounting"]["quantum_state_per_rank"] == 512
    assert (
        plan.summary()["communication_accounting"]["ddp_gradient_allreduce_per_step"]
        == 64
    )
    with pytest.raises(NotImplementedError):
        fq.plan_hybrid_parallel(world_size=2, model_parallel_size=2)


def test_single_rank_semantics_and_non_divisible_memory_are_conservative() -> None:
    local = fq.plan_hybrid_parallel(
        world_size=1, global_quantum_state_bytes=1025, input_bytes=129
    )
    assert local.distribution_semantics == "single_device_fast_path"
    sharded = fq.plan_hybrid_parallel(
        world_size=4,
        data_parallel_size=2,
        state_parallel_size=2,
        global_quantum_state_bytes=1025,
        input_bytes=129,
        parameter_bytes=65,
        observable_bytes=41,
    )
    assert sharded.per_rank_quantum_state_bytes == 513
    assert sharded.per_rank_input_bytes == 65
    assert sharded.state_exchange_bytes_per_gate == 1026
    assert sharded.ddp_gradient_allreduce_bytes_per_step == 65
    assert sharded.observable_reduction_bytes_per_step == 41


def test_parallel_context_requires_initialized_matching_global_world(
    monkeypatch,
) -> None:
    module = fq.Module(parameter_batch_builder, 2)
    plan = fq.plan_hybrid_parallel(
        world_size=4, data_parallel_size=2, state_parallel_size=2
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized process group"):
        module.set_parallel_context(state_process_group=None, plan=plan)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    with pytest.raises(ValueError, match="plan.world_size"):
        module.set_parallel_context(state_process_group=object(), plan=plan)


def test_parallel_context_rejects_process_group_size_mismatch(monkeypatch) -> None:
    module = fq.Module(parameter_batch_builder, 2)
    plan = fq.plan_hybrid_parallel(
        world_size=4, data_parallel_size=2, state_parallel_size=2
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: 4 if group is None else 4,
    )
    with pytest.raises(ValueError, match="size does not match"):
        module.set_parallel_context(state_process_group=object(), plan=plan)


def test_parallel_context_rejects_wrong_state_group_ranks(monkeypatch) -> None:
    module = fq.Module(parameter_batch_builder, 2)
    plan = fq.plan_hybrid_parallel(
        world_size=4, data_parallel_size=2, state_parallel_size=2
    )
    group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: 4 if group is None else 2,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda value: [0, 2]
    )
    with pytest.raises(ValueError, match="ranks do not match"):
        module.set_parallel_context(state_process_group=group, plan=plan)
