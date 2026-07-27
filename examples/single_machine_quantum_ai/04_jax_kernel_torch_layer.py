"""PyTorch training interface with a FlagQuantum JAX quantum kernel."""

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
    speedup,
    time_value_and_grad,
)

import flagquantum as fq  # noqa: E402


def circuit_builder(theta: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(3)
    circuit.rx(0, theta=theta[0])
    circuit.ry(1, theta=theta[1])
    circuit.rz(2, theta=theta[2])
    circuit.cx(0, 1)
    circuit.rxx(1, 2, theta=theta[3])
    circuit.rzz(0, 2, theta=theta[4])
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--mode", choices=("statevector", "mps", "tensor_network"), default="statevector")
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--compare-torch", action="store_true")
    args = parser.parse_args()

    try:
        import jax  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional local install
        print({"example": "jax_kernel_torch_layer", "status": "skipped", "reason": str(exc)})
        return

    hamiltonian = fq.Hamiltonian(
        [
            fq.pauli_term(0.7, "ZZ", (0, 1)),
            fq.pauli_term(-0.3, "X", (0,)),
            fq.pauli_term(0.2, "YY", (1, 2)),
        ]
    )
    exact_energy = exact_ground_energy(hamiltonian, 3)
    layer = fq.QuantumModule(
        circuit_builder,
        n_parameters=5,
        init=torch.linspace(-0.2, 0.2, steps=5),
        policy=fq.RuntimePolicy(
            backend="jax",
            mode=args.mode,
            observable="hamiltonian",
            observable_wires=(0, 1, 2),
            allow_backend_fallback=False,
            mps_max_bond=16 if args.mode == "mps" else None,
        ),
        hamiltonian=hamiltonian,
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=args.lr)
    target = torch.tensor(-0.35)

    def native_loss(theta: torch.Tensor) -> torch.Tensor:
        circuit = circuit_builder(theta)
        if args.mode == "statevector":
            return hamiltonian.expectation(circuit).sum()
        if args.mode == "mps":
            return hamiltonian.expectation(fq.run_mps(circuit, max_bond=16)).sum()
        return hamiltonian.expectation(fq.run_tensor_network(circuit)).sum()

    for step in range(args.steps):
        optimizer.zero_grad()
        prediction = layer()
        loss = (prediction - target) ** 2
        loss.backward()
        optimizer.step()
        if step == 0 or step == args.steps - 1 or (step + 1) % max(1, args.steps // 5) == 0:
            print({"step": step + 1, "prediction": float(prediction.detach()), "loss": float(loss.detach())})

    trained = layer.parameters_tensor.detach()
    native_speed = time_value_and_grad(native_loss, trained, iters=args.bench_iters) if args.compare_torch else None
    jax_speed = time_value_and_grad(
        lambda theta: layer.execute(parameters=theta).value.sum(),
        trained,
        iters=args.bench_iters,
    )
    final_prediction = float(layer().detach())
    print_training_summary(
        title="Single-Machine PyTorch + JAX Quantum Kernel",
        example="single_machine_jax_kernel_torch_layer",
        metrics={
            "mode": args.mode,
            "target": float(target),
            "theoretical_ground_energy": exact_energy,
            "final_prediction": final_prediction,
            "gap_to_target": final_prediction - float(target),
            "trained_parameter_norm": float(layer.parameters_tensor.detach().norm()),
        },
        speed_compare={
            "pytorch_native": native_speed if native_speed is not None else {"status": "unavailable", "reason": "run with --compare-torch"},
            "jax_kernel": jax_speed,
            "speedup_jax_over_pytorch": speedup(
                native_speed["avg_seconds"] if native_speed else None,
                jax_speed["avg_seconds"],
            ),
        },
        model_summary=layer.execute().summary(),
    )


if __name__ == "__main__":
    main()
