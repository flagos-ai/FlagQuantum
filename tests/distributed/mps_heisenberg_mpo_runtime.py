from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

import flagquantum as fq
import flagquantum.runtime.backends.mps.records as fqxm
from flagquantum.runtime.backends.mps.reverse import (
    execute_torch_distributed_mps_reverse,
)


def _circuit(theta: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(4, device=theta.device)
    for wire in range(4):
        circuit.ry(wire, theta)
    for wire in range(3):
        circuit.rxx(wire, wire + 1, theta=0.5 * theta)
    return circuit


def _statevector_circuit(theta: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(4, device=theta.device)
    circuit.x(1).x(3)
    for wire in range(4):
        circuit.ry(wire, theta)
    for wire in range(3):
        circuit.rxx(wire, wire + 1, theta=0.5 * theta)
    return circuit


def _initial(device: torch.device) -> dict[int, torch.Tensor]:
    rank, world = dist.get_rank(), dist.get_world_size()
    first, last = rank * 4 // world, (rank + 1) * 4 // world
    tensors = {}
    for wire in range(first, last):
        tensor = torch.zeros(1, 1, 2, 1, dtype=torch.complex64, device=device)
        tensor[0, 0, wire % 2, 0] = 1
        tensors[wire] = tensor
    return tensors


def _terms():
    values = []
    for left in range(3):
        values.extend(
            (
                ({left: "x", left + 1: "x"}, 0.7),
                ({left: "y", left + 1: "y"}, -0.4),
                ({left: "z", left + 1: "z"}, 1.2),
            )
        )
    values.extend(({wire: "z"}, 0.13 * (wire + 1)) for wire in range(4))
    return tuple(values)


def main() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend)
    theta = torch.tensor(0.17, device=device, requires_grad=True)
    fused = execute_torch_distributed_mps_reverse(
        _circuit(theta),
        hamiltonian_terms=_terms(),
        device=device,
        max_bond=8,
        initial_mps_tensors=_initial(device),
    )
    fused_value = fused.value.detach()
    fused.backward()
    fused_gradient = theta.grad.detach().clone()

    checkpointed_parameter = torch.tensor(0.17, device=device, requires_grad=True)
    checkpointed = execute_torch_distributed_mps_reverse(
        _circuit(checkpointed_parameter),
        hamiltonian_terms=_terms(),
        device=device,
        max_bond=8,
        initial_mps_tensors=_initial(device),
        compile_site_kernels=device.type == "cuda",
        checkpoint_policy=fqxm.MPSReverseCheckpointPolicy(
            max_saved_bytes=1 << 30,
            save_two_site_factorizations=True,
        ),
    )
    checkpointed_value = checkpointed.value.detach()
    checkpointed.backward()
    checkpointed_gradient = checkpointed_parameter.grad.detach().clone()

    reference_value = torch.zeros_like(fused_value)
    reference_gradient = torch.zeros_like(fused_gradient)
    for observable, coefficient in _terms():
        parameter = torch.tensor(0.17, device=device, requires_grad=True)
        result = execute_torch_distributed_mps_reverse(
            _circuit(parameter),
            observable=observable,
            device=device,
            max_bond=8,
            initial_mps_tensors=_initial(device),
        )
        result.backward()
        reference_value += coefficient * result.value.detach()
        reference_gradient += coefficient * parameter.grad.detach()

    torch.testing.assert_close(fused_value, reference_value, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(fused_gradient, reference_gradient, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(checkpointed_value, fused_value, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        checkpointed_gradient, fused_gradient, rtol=5e-5, atol=5e-5
    )
    statevector_parameter = torch.tensor(0.17, device=device, requires_grad=True)
    statevector_hamiltonian = fq.Hamiltonian(
        [fq.pauli_term(coefficient, observable) for observable, coefficient in _terms()]
    )
    statevector_value = statevector_hamiltonian.expectation(
        _statevector_circuit(statevector_parameter)
    ).sum()
    statevector_value.backward()
    torch.testing.assert_close(
        fused_value, statevector_value.detach(), rtol=5e-5, atol=5e-5
    )
    torch.testing.assert_close(
        fused_gradient, statevector_parameter.grad, rtol=5e-5, atol=5e-5
    )
    print(
        json.dumps(
            {
                "rank": dist.get_rank(),
                "heisenberg_mpo_value_passed": True,
                "heisenberg_mpo_gradient_passed": True,
                "factorization_checkpoint_gradient_passed": True,
                "statevector_gradient_parity_passed": True,
                "objective_scan_pairs": fused.objective_scan_pairs,
                "forward_messages": fused.objective_scan_forward_messages,
                "reverse_messages": fused.objective_scan_reverse_messages,
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
