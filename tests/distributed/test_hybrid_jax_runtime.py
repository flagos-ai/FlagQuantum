import importlib.util
import os

import pytest
import torch
import torch.distributed as dist

import flagquantum as fq

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.distributed_cpu,
    pytest.mark.distributed_multinode,
    pytest.mark.skipif(
        "RANK" not in os.environ,
        reason="requires torchrun with distributed RANK/WORLD_SIZE environment",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("jax") is None, reason="jax is not installed"
    ),
]


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_hybrid_circuit(values: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(3)
    circuit.rx(0, theta=values[0])
    circuit.ry(1, theta=values[1])
    circuit.rz(2, theta=values[2])
    circuit.cx(0, 1)
    circuit.rxx(1, 2, theta=values[3])
    return circuit


def _hamiltonian() -> fq.Hamiltonian:
    return fq.Hamiltonian(
        [
            fq.pauli_term(0.7, "ZZ", (0, 1)),
            fq.pauli_term(-0.2, "X", (0,)),
            fq.pauli_term(0.13, "YY", (1, 2)),
        ]
    )


def test_rank_local_jax_kernel_gradients_allreduce_match_reference():
    device = _device()
    world_size = _world_size()
    if not dist.is_initialized():
        fq.init_torch_distributed(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            device=device,
            force_initialize=True,
        )
    params = torch.tensor([0.2, -0.1, 0.3, 0.17], device=device, requires_grad=True)
    kernel = fq.compile_quantum_kernel(
        _build_hybrid_circuit,
        params.detach(),
        backend="jax",
        interface="torch",
        mode="statevector",
        n_wires=3,
        hamiltonian=_hamiltonian(),
    )

    local_loss = kernel(params)
    scaled_loss = local_loss / float(world_size)
    scaled_loss.backward()
    assert params.grad is not None
    dist.all_reduce(params.grad, op=dist.ReduceOp.SUM)

    reference_params = params.detach().clone().requires_grad_(True)
    reference_loss = (
        _hamiltonian().expectation(_build_hybrid_circuit(reference_params)).sum()
    )
    reference_loss.backward()
    assert reference_params.grad is not None

    assert torch.allclose(
        local_loss.detach().cpu(), reference_loss.detach().cpu(), atol=1e-5
    )
    assert torch.allclose(
        params.grad.detach().cpu(), reference_params.grad.detach().cpu(), atol=1e-4
    )
    assert kernel.summary()["interface"] == "torch"
