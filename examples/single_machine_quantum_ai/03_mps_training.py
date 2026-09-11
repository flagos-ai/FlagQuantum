"""Local MPS quantum AI training example.

This demonstrates a larger local circuit route. The training objective pushes a
hardware-efficient ansatz toward a low-energy nearest-neighbor ZZ Hamiltonian.
"""

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
import flagquantum.simulation.mps as fqmps  # noqa: E402
from flagquantum.algorithms import (  # noqa: E402
    hardware_efficient_parameter_count,
    zz_chain_hamiltonian,
)


def build_ansatz(
    theta: torch.Tensor, *, n_wires: int, layers: int, device: str
) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device=device)
    cursor = 0
    for _ in range(layers):
        for wire in range(n_wires):
            circuit.ry(wire, theta=theta[cursor])
            cursor += 1
            circuit.rz(wire, theta=theta[cursor])
            cursor += 1
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument(
        "--reference", choices=("auto", "small_exact", "none"), default="auto"
    )
    parser.add_argument("--small-exact-max-wires", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(19)
    n_params = hardware_efficient_parameter_count(args.n_qubits, args.layers)
    parameters = (0.15 * torch.randn(n_params, device=args.device)).requires_grad_(True)
    optimizer = torch.optim.Adam([parameters], lr=args.lr)
    hamiltonian = zz_chain_hamiltonian(args.n_qubits, coupling=-1.0, field=0.1)

    def build(theta: torch.Tensor) -> fq.Circuit:
        return build_ansatz(
            theta, n_wires=args.n_qubits, layers=args.layers, device=args.device
        )

    def native_mps_loss(theta: torch.Tensor) -> torch.Tensor:
        mps = fqmps.run_mps(build(theta), max_bond=args.max_bond)
        return hamiltonian.expectation(mps).sum()

    def training_loss(theta: torch.Tensor) -> torch.Tensor:
        return native_mps_loss(theta)

    exact_energy = (
        exact_ground_energy(hamiltonian, args.n_qubits)
        if args.reference != "none" and args.n_qubits <= args.small_exact_max_wires
        else None
    )
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

    trained_mps = fqmps.run_mps(build(parameters.detach()), max_bond=args.max_bond)
    native_speed = None
    if args.compare_torch:
        native_speed = time_value_and_grad(
            native_mps_loss,
            parameters.detach(),
            iters=args.bench_iters,
            device=args.device,
        )
    final_energy = float(training_loss(parameters).detach())
    print_training_summary(
        title="Single-Machine Quantum AI (MPS)",
        example="single_machine_mps_training",
        metrics={
            "theoretical_ground_energy": exact_energy,
            "reference": (
                f"exact diagonalization <= {args.small_exact_max_wires} wires"
                if exact_energy is not None
                else (
                    "disabled"
                    if args.reference == "none"
                    else "n/a for this wire count"
                )
            ),
            "training_backend": "pytorch",
            "initial_energy": initial,
            "final_energy": final_energy,
            "gap_to_theory": (
                None if exact_energy is None else final_energy - exact_energy
            ),
        },
        speed_compare={
            "pytorch_native_mps": (
                native_speed
                if native_speed is not None
                else {"status": "unavailable", "reason": "run with --compare-torch"}
            ),
            "jax_kernel_mps": {
                "status": "see examples/single_machine_quantum_ai/05_mps_1000q_dimer_training.py"
            },
        },
        model_summary=trained_mps.summary(),
    )


if __name__ == "__main__":
    main()
