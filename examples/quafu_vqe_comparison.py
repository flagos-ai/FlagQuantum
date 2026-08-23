"""Compare a two-qubit VQE trajectory on statevector and Quafu hardware."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

import flagquantum as fq


def ansatz(theta: torch.Tensor | float, *, x_basis: bool = False) -> fq.Circuit:
    circuit = fq.Circuit(2)
    circuit.ry(0, theta).ry(1, theta)
    if x_basis:
        circuit.h(0).h(1)
    return circuit


def statevector_energy(theta: torch.Tensor) -> torch.Tensor:
    circuit = ansatz(theta)
    return -(
        circuit.expectation_ps(z=(0, 1))
        + circuit.expectation_ps(x=(0,))
        + circuit.expectation_ps(x=(1,))
    ).sum()


def expectation(counts: dict[str, int], wires: tuple[int, ...]) -> float:
    shots = sum(counts.values())
    return sum(
        count
        * math.prod(1 if bits[wire] == "0" else -1 for wire in wires)
        for bits, count in counts.items()
    ) / shots


def wait_for_result(
    provider: fq.QuafuProvider,
    handle: fq.ProviderTaskHandle,
    *,
    timeout: float,
) -> fq.DeploymentResult:
    started = time.monotonic()
    while True:
        status = provider.query_status(handle)
        elapsed = time.monotonic() - started
        print(
            f"task_id={handle.task_id} basis={handle.payload['vqe_basis']} "
            f"status={status} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if status in {"Finished", "Completed", "Done"}:
            try:
                return provider.fetch_result(handle)
            except RuntimeError as exc:
                if "has no result yet" not in str(exc):
                    raise
        if status in {"Failed", "Cancelled", "Canceled"}:
            raise RuntimeError(f"Quafu task {handle.task_id} ended as {status}")
        if elapsed >= timeout:
            raise TimeoutError(f"Quafu task {handle.task_id} timed out")
        time.sleep(5)


def submit_basis(
    provider: fq.QuafuProvider,
    backend: fq.CloudBackendProfile,
    theta: float,
    basis: str,
    iteration: int,
    shots: int,
    timeout: float,
    target_qubits: list[int] | None,
) -> tuple[fq.DeploymentResult, str]:
    package = fq.create_deployment_package(
        ansatz(theta, x_basis=basis == "x"),
        backend=backend,
        name=f"flagquantum_vqe_{iteration:02d}_{basis}",
        shots=shots,
        metadata={
            "provider_compile": True,
            "provider_options": {
                "compiler": "quarkcircuit",
                "correct": False,
                "open_dd": None,
                "target_qubits": target_qubits or [],
            },
        },
    )
    handle = provider.submit(package)
    handle = fq.ProviderTaskHandle(
        provider=handle.provider,
        task_id=handle.task_id,
        backend_name=handle.backend_name,
        payload=dict(handle.payload) | {"vqe_basis": basis},
    )
    return wait_for_result(provider, handle, timeout=timeout), package.qasm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="Baihua")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--target-qubits", type=int, nargs=2, default=None)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=Path("artifacts/quafu_vqe"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed points from output/comparison.json",
    )
    args = parser.parse_args()
    if not os.getenv("QPU_API_TOKEN"):
        raise RuntimeError("QPU_API_TOKEN is required")

    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD((theta,), lr=0.3)
    trajectory = []
    for iteration in range(args.iterations):
        optimizer.zero_grad()
        energy = statevector_energy(theta)
        energy.backward()
        trajectory.append(
            {
                "iteration": iteration,
                "theta": float(theta.detach()),
                "statevector_energy": float(energy.detach()),
                "gradient": float(theta.grad.detach()),
            }
        )
        optimizer.step()

    provider = fq.QuafuProvider(timeout=30)
    backend = next(
        item
        for item in provider.discover_backends(2)
        if item.name == args.backend
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "comparison.json"
    completed = {}
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        completed = {
            int(point["iteration"]): point
            for point in previous.get("points", ())
            if "quafu_energy" in point
        }
    for point in trajectory:
        iteration = point["iteration"]
        if iteration in completed:
            previous_point = completed[iteration]
            if not math.isclose(
                point["theta"], previous_point["theta"], rel_tol=0.0, abs_tol=1e-9
            ):
                raise RuntimeError(
                    f"resume theta mismatch at iteration {iteration}: "
                    f"{point['theta']} != {previous_point['theta']}"
                )
            point.update(
                {
                    key: value
                    for key, value in previous_point.items()
                    if key not in {"statevector_energy", "gradient", "theta"}
                }
            )
            print(
                f"iteration={iteration} reused task_ids={point['task_ids']}",
                flush=True,
            )
            continue
        z_result, z_qasm = submit_basis(
            provider,
            backend,
            point["theta"],
            "z",
            iteration,
            args.shots,
            args.timeout,
            args.target_qubits,
        )
        x_result, x_qasm = submit_basis(
            provider,
            backend,
            point["theta"],
            "x",
            iteration,
            args.shots,
            args.timeout,
            args.target_qubits,
        )
        zz = expectation(dict(z_result.counts), (0, 1))
        x0 = expectation(dict(x_result.counts), (0,))
        x1 = expectation(dict(x_result.counts), (1,))
        point.update(
            {
                "quafu_energy": -zz - x0 - x1,
                "quafu_expectations": {"ZZ": zz, "X0": x0, "X1": x1},
                "quafu_counts": {
                    "z_basis": dict(z_result.counts),
                    "x_basis": dict(x_result.counts),
                },
                "task_ids": {
                    "z_basis": z_result.handle.task_id,
                    "x_basis": x_result.handle.task_id,
                },
                "qasm": {"z_basis": z_qasm, "x_basis": x_qasm},
            }
        )
        payload = {
            "schema": "flagquantum_quafu_vqe_comparison_v1",
            "backend": backend.name,
            "shots_per_basis": args.shots,
            "physical_qubits": args.target_qubits,
            "hamiltonian": "-Z0 Z1 - X0 - X1",
            "ansatz": "Ry(theta) on q0 and q1",
            "exact_ansatz_minimum": -2.0,
            "points": trajectory,
        }
        result_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"iteration={iteration} theta={point['theta']:.6f} "
            f"sv={point['statevector_energy']:.6f} "
            f"quafu={point['quafu_energy']:.6f}",
            flush=True,
        )

    import matplotlib.pyplot as plt

    iterations = [point["iteration"] for point in trajectory]
    quafu_energies = [point["quafu_energy"] for point in trajectory]
    quafu_smoothed = [
        sum(quafu_energies[max(0, i - 2) : i + 1]) / min(3, i + 1)
        for i in range(len(quafu_energies))
    ]
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(
        iterations,
        [point["statevector_energy"] for point in trajectory],
        "o-",
        label="FlagQuantum statevector",
    )
    plt.plot(
        iterations,
        quafu_energies,
        "s-",
        alpha=0.45,
        label=f"Quafu {backend.name} raw",
    )
    plt.plot(
        iterations,
        quafu_smoothed,
        "D-",
        linewidth=2,
        markersize=4,
        label=f"Quafu {backend.name} 3-point mean",
    )
    plt.axhline(-2.0, color="black", linestyle="--", label="Exact ansatz min")
    plt.xlabel("VQE iteration")
    plt.ylabel("Energy")
    plt.title(r"$H=-Z_0Z_1-X_0-X_1$")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "energy_curve.png", dpi=180)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
