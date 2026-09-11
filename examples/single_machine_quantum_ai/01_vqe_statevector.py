"""End-to-end VQE on a single CPU/GPU with PyTorch autograd."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    exact_ground_energy,
    print_training_summary,
    time_value_and_grad,
)

import flagquantum as fq  # noqa: E402
from flagquantum.algorithms import (  # noqa: E402
    Hamiltonian,
    hardware_efficient_parameter_count,
    pauli_term,
)


def build_ansatz(
    parameters: torch.Tensor, *, n_wires: int, layers: int, device: str
) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device=device)
    cursor = 0
    for _ in range(layers):
        for wire in range(n_wires):
            circuit.ry(wire, theta=parameters[cursor])
            cursor += 1
            circuit.rz(wire, theta=parameters[cursor])
            cursor += 1
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--compare-torch", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(7)
    n_params = hardware_efficient_parameter_count(args.n_qubits, args.layers)
    parameters = (0.2 * torch.randn(n_params, device=args.device)).requires_grad_(True)
    hamiltonian = Hamiltonian(
        [
            *(pauli_term(0.7, "ZZ", (i, i + 1)) for i in range(args.n_qubits - 1)),
            *(pauli_term(-0.25, "X", (i,)) for i in range(args.n_qubits)),
            pauli_term(0.05, "Z", (0,)),
        ]
    )

    def native_loss(theta: torch.Tensor) -> torch.Tensor:
        circuit = build_ansatz(
            theta, n_wires=args.n_qubits, layers=args.layers, device=args.device
        )
        return hamiltonian.expectation(circuit).sum()

    def training_loss(theta: torch.Tensor) -> torch.Tensor:
        return native_loss(theta)

    optimizer = torch.optim.Adam([parameters], lr=args.lr)
    exact_energy = exact_ground_energy(hamiltonian, args.n_qubits)
    initial = float(training_loss(parameters).detach())
    for step in range(args.steps):
        optimizer.zero_grad()
        loss = training_loss(parameters)
        loss.backward()
        optimizer.step()
        if (
            step == 0
            or step == args.steps - 1
            or (step + 1) % max(1, args.steps // 5) == 0
        ):
            print({"step": step + 1, "energy": float(loss.detach())})

    trained = build_ansatz(
        parameters.detach(),
        n_wires=args.n_qubits,
        layers=args.layers,
        device=args.device,
    )
    native_speed = None
    if args.compare_torch:
        native_speed = time_value_and_grad(
            native_loss,
            parameters.detach(),
            iters=args.bench_iters,
            device=args.device,
        )
    final_energy = float(training_loss(parameters).detach())
    speed_compare = {
        "pytorch_native": (
            native_speed
            if native_speed is not None
            else {
                "status": "unavailable",
                "reason": "pass --compare-torch to benchmark the native path",
            }
        ),
    }
    speed_compare["jax_kernel"] = {
        "status": "see examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py"
    }
    print_training_summary(
        title="Single-Machine VQE (Statevector)",
        example="single_machine_vqe_statevector",
        metrics={
            "theoretical_ground_energy": exact_energy,
            "training_backend": "pytorch",
            "initial_energy": initial,
            "final_energy": final_energy,
            "gap_to_theory": final_energy - exact_energy,
            "optimized_parameter_norm": float(parameters.detach().norm()),
        },
        speed_compare=speed_compare,
        model_summary=trained.plan().summary(),
    )


if __name__ == "__main__":
    main()
