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
    jax_available,
    print_training_summary,
    speedup,
    time_value_and_grad,
)

import flagquantum as fq  # noqa: E402


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
    parser.add_argument("--backend", choices=("jax", "torch"), default="torch")
    parser.add_argument("--compare-torch", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(7)
    n_params = fq.hardware_efficient_parameter_count(args.n_qubits, args.layers)
    parameters = (0.2 * torch.randn(n_params, device=args.device)).requires_grad_(True)
    hamiltonian = fq.Hamiltonian(
        [
            *(fq.pauli_term(0.7, "ZZ", (i, i + 1)) for i in range(args.n_qubits - 1)),
            *(fq.pauli_term(-0.25, "X", (i,)) for i in range(args.n_qubits)),
            fq.pauli_term(0.05, "Z", (0,)),
        ]
    )

    def native_loss(theta: torch.Tensor) -> torch.Tensor:
        circuit = build_ansatz(
            theta, n_wires=args.n_qubits, layers=args.layers, device=args.device
        )
        return hamiltonian.expectation(circuit).sum()

    has_jax = False
    jax_error = None
    if args.backend == "jax":
        has_jax, jax_error = jax_available()
    if args.backend == "jax" and not has_jax:
        raise SystemExit(
            f"JAX backend requested but unavailable: {jax_error}. "
            "Use --backend torch to run the native path."
        )
    jax_kernel = None
    if has_jax:
        jax_kernel = fq.compile_quantum_kernel(
            lambda theta: build_ansatz(
                theta, n_wires=args.n_qubits, layers=args.layers, device=args.device
            ),
            parameters.detach(),
            backend="jax",
            interface="torch",
            mode="statevector",
            n_wires=args.n_qubits,
            hamiltonian=hamiltonian,
            jit=True,
        )

    def training_loss(theta: torch.Tensor) -> torch.Tensor:
        if args.backend == "jax":
            return jax_kernel(theta).sum()
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
    jax_speed = (
        time_value_and_grad(
            lambda theta: jax_kernel(theta).sum(),
            parameters.detach(),
            iters=args.bench_iters,
            device=args.device,
        )
        if jax_kernel is not None
        else None
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
    if jax_speed is not None:
        speed_compare["jax_kernel"] = jax_speed
        speed_compare["speedup_jax_over_pytorch"] = speedup(
            native_speed["avg_seconds"] if native_speed else None,
            jax_speed["avg_seconds"],
        )
    print_training_summary(
        title="Single-Machine VQE (Statevector)",
        example="single_machine_vqe_statevector",
        metrics={
            "theoretical_ground_energy": exact_energy,
            "training_backend": args.backend,
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
