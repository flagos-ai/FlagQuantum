import os

import pytest
import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.execution import run_advanced

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.distributed_cpu,
    pytest.mark.distributed_multinode,
    pytest.mark.skipif(
        "RANK" not in os.environ,
        reason="requires torchrun with distributed RANK/WORLD_SIZE environment",
    ),
]


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _precision_assert(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-6
) -> float:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    max_error = torch.max(torch.abs(actual_cpu - expected_cpu)).item()
    assert torch.allclose(actual_cpu, expected_cpu, atol=atol, rtol=atol)
    assert max_error <= atol
    return float(max_error)


def _reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(4)
    circuit.h(0).cx(1, 2).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    return circuit


def _parameterized_circuit(params: torch.Tensor, *, device: str) -> fq.Circuit:
    circuit = fq.Circuit(4, device=device)
    circuit.h(0)
    circuit.rx(1, theta=params[0])
    circuit.ry(2, theta=params[1])
    circuit.rz(3, theta=params[2])
    circuit.cx(0, 3)
    circuit.rzz(1, 2, theta=params[3])
    return circuit


def _z_loss_from_state(
    state: torch.Tensor, wires: tuple[int, ...] = (0, 2)
) -> torch.Tensor:
    flat = state.reshape(state.shape[0], -1)
    n_wires = int(torch.log2(torch.tensor(flat.shape[-1], dtype=torch.float32)).item())
    probs = torch.abs(flat) ** 2
    values = []
    for wire in wires:
        bit = (
            torch.arange(flat.shape[-1], device=flat.device) >> (n_wires - 1 - wire)
        ) & 1
        signs = (1 - 2 * bit).to(dtype=probs.dtype)
        values.append((probs * signs).sum(dim=-1))
    return torch.stack(values, dim=-1).sum()


def _training_loss(result: object) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return _z_loss_from_state(result)
    if hasattr(result, "expectation_z"):
        return result.expectation_z((0, 2)).sum()
    if hasattr(result, "to_statevector"):
        return _z_loss_from_state(result.to_statevector())
    return _z_loss_from_state(result.state())


def _loss_and_grad(
    mode: str,
    *,
    device: str,
    world_size: int,
    distributed_executor: str | None = None,
    aggregate_grad: bool = False,
    **options,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = torch.tensor(
        [0.17, -0.31, 0.23, -0.19],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    circuit = _parameterized_circuit(params, device=device)
    run_options = dict(options)
    if distributed_executor is not None:
        run_options["distributed_executor"] = distributed_executor
    if mode.startswith("distributed"):
        run_options["world_size"] = world_size
    result = run_advanced(circuit, mode=mode, device=device, **run_options)
    if mode == "distributed_statevector":
        loss = fq.measure_allZ(result.native())[:, (0, 2)].sum()
    elif mode == "distributed_tensor_network":
        loss = _z_loss_from_state(result.to_statevector())
    else:
        loss = _training_loss(result)
    if loss.requires_grad:
        loss.backward()
    if params.grad is None:
        params.grad = torch.zeros_like(params)
    if aggregate_grad and dist.is_available() and dist.is_initialized():
        dist.all_reduce(params.grad, op=dist.ReduceOp.SUM)
    return loss.detach().cpu(), params.grad.detach().cpu()


def _gradient_precision_assert(
    actual_loss: torch.Tensor,
    actual_grad: torch.Tensor,
    expected_loss: torch.Tensor,
    expected_grad: torch.Tensor,
    *,
    atol: float = 1e-5,
) -> tuple[float, float]:
    loss_error = torch.max(torch.abs(actual_loss - expected_loss)).item()
    grad_error = torch.max(torch.abs(actual_grad - expected_grad)).item()
    assert torch.allclose(actual_loss, expected_loss, atol=atol, rtol=atol)
    assert torch.allclose(actual_grad, expected_grad, atol=atol, rtol=atol)
    assert loss_error <= atol
    assert grad_error <= atol
    return float(loss_error), float(grad_error)


def test_distributed_tensor_network_torchrun_precision_alignment():
    circuit = _reference_circuit()
    world_size = _world_size()
    device = _device()

    distributed = run_advanced(
        circuit,
        mode="distributed_tensor_network",
        world_size=world_size,
        distributed_executor="torch",
        device=device,
        max_intermediate_size=8,
    )
    local = run_advanced(
        circuit,
        mode="tensor_network",
        device=device,
        max_intermediate_size=8,
    )

    max_error = _precision_assert(
        distributed.native().state(), local.native().state(), atol=1e-6
    )
    summary = distributed.summary()
    assert summary["executor"] == "torch_distributed"
    assert summary["state_mode"] == "distributed_tensor_network"
    assert summary["world_size"] == world_size
    assert summary["rank"] == _rank()
    assert summary["slice_tasks"] == len(distributed.tasks)
    assert set(summary["tasks_by_rank"]) == set(range(world_size))
    assert max_error <= 1e-6


def test_distributed_statevector_torchrun_gradient_precision_alignment():
    world_size = _world_size()
    device = _device()

    distributed_loss, distributed_grad = _loss_and_grad(
        "distributed_statevector",
        device=device,
        world_size=world_size,
        aggregate_grad=True,
    )
    reference_loss, reference_grad = _loss_and_grad(
        "statevector",
        device=device,
        world_size=world_size,
    )

    _gradient_precision_assert(
        distributed_loss,
        distributed_grad,
        reference_loss,
        reference_grad,
        atol=1e-5,
    )


def test_distributed_tensor_network_torchrun_gradient_precision_alignment():
    world_size = _world_size()
    device = _device()

    distributed_loss, distributed_grad = _loss_and_grad(
        "distributed_tensor_network",
        device=device,
        world_size=world_size,
        distributed_executor="torch",
        aggregate_grad=True,
        max_intermediate_size=8,
    )
    reference_loss, reference_grad = _loss_and_grad(
        "tensor_network",
        device=device,
        world_size=world_size,
        max_intermediate_size=8,
    )

    _gradient_precision_assert(
        distributed_loss,
        distributed_grad,
        reference_loss,
        reference_grad,
        atol=1e-5,
    )


def test_distributed_mps_torchrun_precision_alignment():
    circuit = _reference_circuit()
    world_size = _world_size()
    device = _device()

    distributed = run_advanced(
        circuit,
        mode="distributed_mps",
        world_size=world_size,
        distributed_executor="torch",
        device=device,
        max_bond=8,
        boundary_transport="auto",
    )
    local = run_advanced(circuit, mode="mps", device=device, max_bond=8)

    max_error = _precision_assert(
        distributed.to_statevector(), local.to_statevector(), atol=1e-6
    )
    summary = distributed.summary()
    assert summary["executor"] == "torch_distributed"
    assert summary["state_mode"] == "distributed_mps"
    assert summary["world_size"] == world_size
    assert summary["rank"] == _rank()
    assert summary["storage"] == "sharded"
    assert summary["mps_execution"] == "site_sharded_sync"
    assert distributed.local_shard_tensors
    assert distributed.sharded_state is not None
    assert max_error <= 1e-6


def test_distributed_mps_torchrun_gradient_precision_alignment():
    world_size = _world_size()
    device = _device()

    distributed_loss, distributed_grad = _loss_and_grad(
        "distributed_mps",
        device=device,
        world_size=world_size,
        distributed_executor="torch",
        max_bond=8,
        boundary_transport="auto",
    )
    reference_loss, reference_grad = _loss_and_grad(
        "mps",
        device=device,
        world_size=world_size,
        max_bond=8,
    )

    _gradient_precision_assert(
        distributed_loss,
        distributed_grad,
        reference_loss,
        reference_grad,
        atol=1e-5,
    )
