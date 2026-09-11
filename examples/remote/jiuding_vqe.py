"""Run an eight-parameter entangling VQE loop on a warm Jiuding executor."""

import argparse
import json
import time

import torch

import flagquantum as fq
from flagquantum.gradients import batched_parameter_shift_gradient
from flagquantum.remote.compute import JiudingClient


def build_ansatz(parameters: torch.Tensor) -> fq.Circuit:
    """Build the fixed four-qubit, two-layer validation ansatz."""

    circuit = fq.Circuit(4)
    for wire in range(4):
        circuit.ry(wire, theta=parameters[wire])
    for wire in range(3):
        circuit.cx(wire, wire + 1)
    for wire in range(4):
        circuit.ry(wire, theta=parameters[wire + 4])
    for wire in range(3):
        circuit.cx(wire, wire + 1)
    return circuit


def hamiltonian() -> fq.Observable:
    """Return a four-term Pauli-Z Hamiltonian as one requested output."""

    observable = fq.Z(0)
    for wire in range(1, 4):
        observable += fq.Z(wire)
    return observable


def reference_gradient(
    parameters: torch.Tensor, output: fq.OutputRequest
) -> torch.Tensor:
    """Compute the local autograd oracle used only for validation."""

    reference = parameters.detach().clone().requires_grad_(True)
    value = fq.run(build_ansatz(reference), outputs=output).expectation().sum()
    value.backward()
    assert reference.grad is not None
    return reference.grad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    parameters = torch.linspace(0.15, 0.85, 8, dtype=torch.float32)
    output = fq.expectation(hamiltonian())
    steps = []
    batch_calls = 0
    cpu_fallback_used = False

    with JiudingClient(workspace=args.workspace) as client:

        def remote_energy(circuit: fq.Circuit) -> float:
            nonlocal cpu_fallback_used
            result = client.run(circuit, target=args.target, outputs=output)
            cpu_fallback_used |= bool(result.provenance["cpu_fallback_used"])
            return float(result.expectation().sum())

        def evaluate_batch(
            circuits: tuple[fq.Circuit, ...],
        ) -> tuple[torch.Tensor, ...]:
            nonlocal batch_calls, cpu_fallback_used
            batch_calls += 1
            results = client.run_batch(
                circuits,
                target=args.target,
                outputs=output,
            )
            cpu_fallback_used |= any(
                bool(result.provenance["cpu_fallback_used"]) for result in results
            )
            steps[-1]["batch_remote_seconds"] = float(
                results[0].runtime["batch_elapsed_seconds"]
            )
            return tuple(result.expectation().sum() for result in results)

        started = time.perf_counter()
        initial_energy = remote_energy(build_ansatz(parameters))
        energy = initial_energy
        for index in range(args.steps):
            steps.append({"step": index + 1, "energy_before": energy})
            expected_gradient = reference_gradient(parameters, output)
            gradient_started = time.perf_counter()
            gradient = batched_parameter_shift_gradient(
                build_ansatz,
                parameters,
                evaluate_batch,
            )
            steps[-1]["gradient_total_seconds"] = time.perf_counter() - gradient_started
            steps[-1]["gradient_norm"] = float(torch.linalg.vector_norm(gradient))
            steps[-1]["maximum_gradient_error"] = float(
                torch.max(torch.abs(gradient - expected_gradient))
            )
            parameters = parameters - args.learning_rate * gradient
            energy = remote_energy(build_ansatz(parameters))
            steps[-1]["energy_after"] = energy
        total_seconds = time.perf_counter() - started

    if batch_calls != args.steps:
        raise RuntimeError("each VQE step must use exactly one gradient batch call")
    if cpu_fallback_used:
        raise RuntimeError("Jiuding GPU VQE used an unexpected CPU fallback")

    print(
        json.dumps(
            {
                "parameter_count": parameters.numel(),
                "shifted_circuits_per_step": 2 * parameters.numel(),
                "steps": steps,
                "gradient_batch_calls": batch_calls,
                "initial_energy": initial_energy,
                "final_energy": energy,
                "energy_decreased": energy < initial_energy,
                "cpu_fallback_used": cpu_fallback_used,
                "total_seconds": total_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
