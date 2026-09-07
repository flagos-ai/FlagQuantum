"""Run end-to-end TFIM VQE training trajectories on PyTorch and JAX kernels."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402


def _parameters(n_wires: int, layers: int, device: str) -> torch.Tensor:
    return torch.linspace(
        -0.37,
        0.41,
        steps=layers * n_wires * 3,
        dtype=torch.float32,
        device=device,
    ).reshape(layers, n_wires, 3)


def _circuit(values: torch.Tensor) -> fq.Circuit:
    layers, n_wires, _ = values.shape
    circuit = fq.Circuit(int(n_wires), device=str(values.device))
    for layer in range(int(layers)):
        for wire in range(int(n_wires)):
            circuit.rx(wire, theta=values[layer, wire, 0])
            circuit.ry(wire, theta=values[layer, wire, 1])
            circuit.rz(wire, theta=values[layer, wire, 2])
        for wire in range(int(n_wires) - 1):
            circuit.cx(wire, wire + 1)
        if int(n_wires) > 2:
            circuit.rxx(0, int(n_wires) - 1, theta=values[layer, 0, 0] * 0.25)
    return circuit


def _exact_ground_energy(n_wires: int, coupling: float, field: float) -> float:
    from scipy.sparse import csr_matrix, eye, kron
    from scipy.sparse.linalg import eigsh

    identity = eye(2, format="csr", dtype=np.float64)
    x = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    z = csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))

    def term(operators: dict[int, csr_matrix]) -> csr_matrix:
        result = csr_matrix([[1.0]])
        for wire in range(n_wires):
            result = kron(result, operators.get(wire, identity), format="csr")
        return result

    size = 2**n_wires
    hamiltonian = csr_matrix((size, size), dtype=np.float64)
    for wire in range(n_wires - 1):
        hamiltonian -= coupling * term({wire: z, wire + 1: z})
    for wire in range(n_wires):
        hamiltonian -= field * term({wire: x})
    return float(eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False)[0])


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _train(
    *,
    backend: str,
    n_wires: int,
    layers: int,
    steps: int,
    optimizer_name: str,
    lr: float,
    device: str,
    ground_energy: float,
    tolerance: float,
    checkpoint_dir: Path,
    checkpoint_interval: int,
    resume: bool,
    resume_lr: float | None,
) -> dict[str, Any]:
    seed = _parameters(n_wires, layers, device)
    parameters = seed.detach().clone().requires_grad_(True)
    checkpoint_path = (
        checkpoint_dir / f"{optimizer_name}_{backend}_q{n_wires}_d{layers}.pt"
    )
    saved = None
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        parameters.data.copy_(saved["parameters"])
    hamiltonian = fq.transverse_field_ising(
        n_wires, coupling=0.7, field=0.25, periodic=False
    )
    started = time.perf_counter()
    kernel = None
    if backend == "jax":
        kernel = fq.compile_quantum_kernel(
            _circuit,
            seed,
            backend="jax",
            interface="torch",
            mode="statevector",
            n_wires=n_wires,
            hamiltonian=hamiltonian,
            jit=True,
            compute_dtype="complex64",
        )

    def loss_fn() -> torch.Tensor:
        if kernel is not None:
            return kernel(parameters).sum()
        return hamiltonian.expectation(_circuit(parameters)).sum()

    if optimizer_name == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam([parameters], lr=lr)
    else:
        optimizer = torch.optim.LBFGS(
            [parameters], lr=lr, max_iter=1, history_size=20, line_search_fn=None
        )

    if saved is not None:
        optimizer.load_state_dict(saved["optimizer_state"])
        saved_lr = float(saved["learning_rate"])
        if resume_lr is None and not math.isclose(saved_lr, lr):
            raise ValueError(
                f"checkpoint learning rate is {saved_lr}, requested {lr}; "
                "pass --resume-lr explicitly to change it"
            )
        if resume_lr is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = resume_lr
            lr = resume_lr

    energies: list[float] = list(saved["energies"]) if saved else []
    errors: list[float] = list(saved["energy_errors"]) if saved else []
    wall_seconds: list[float] = list(saved["cumulative_wall_seconds"]) if saved else []
    closure_evaluations = int(saved["closure_evaluations"]) if saved else 0
    reached_step = saved.get("first_converged_step") if saved else None
    reached_seconds = saved.get("first_converged_wall_seconds") if saved else None
    start_step = int(saved["completed_steps"]) if saved else 0
    elapsed_offset = wall_seconds[-1] if wall_seconds else 0.0

    def save_checkpoint(completed_steps: int) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "schema": "flagquantum.vqe_training_checkpoint.v1",
            "backend": backend,
            "optimizer": optimizer_name,
            "n_wires": n_wires,
            "layers": layers,
            "learning_rate": lr,
            "completed_steps": completed_steps,
            "parameters": parameters.detach(),
            "optimizer_state": optimizer.state_dict(),
            "energies": energies,
            "energy_errors": errors,
            "cumulative_wall_seconds": wall_seconds,
            "closure_evaluations": closure_evaluations,
            "first_converged_step": reached_step,
            "first_converged_wall_seconds": reached_seconds,
            "ground_energy": ground_energy,
            "convergence_tolerance": tolerance,
        }
        temporary = checkpoint_path.with_suffix(f".tmp-{os.getpid()}")
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)

    for step in range(start_step, steps):
        if optimizer_name == "adam":
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            optimizer.step()
            closure_evaluations += 1
            energy = float(loss.detach())
        else:
            observed: list[float] = []

            def closure() -> torch.Tensor:
                nonlocal closure_evaluations
                optimizer.zero_grad(set_to_none=True)
                value = loss_fn()
                value.backward()
                observed.append(float(value.detach()))
                closure_evaluations += 1
                return value

            optimizer.step(closure)
            energy = observed[-1]
        _sync(device)
        elapsed = elapsed_offset + time.perf_counter() - started
        error = max(0.0, energy - ground_energy)
        energies.append(energy)
        errors.append(error)
        wall_seconds.append(elapsed)
        if reached_step is None and error <= tolerance:
            reached_step, reached_seconds = step + 1, elapsed
        if (step + 1) % checkpoint_interval == 0 or step + 1 == steps:
            save_checkpoint(step + 1)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            print(
                f"heartbeat backend={backend} optimizer={optimizer_name} "
                f"q={n_wires} depth={layers} step={step + 1}/{steps} "
                f"error={error:.6g} wall={elapsed:.3f}s",
                file=sys.stderr,
                flush=True,
            )

    return {
        "backend": backend,
        "optimizer": optimizer_name,
        "n_wires": n_wires,
        "layers": layers,
        "steps": steps,
        "resumed_from_step": start_step,
        "learning_rate": lr,
        "initial_parameters": seed.detach().cpu().flatten().tolist(),
        "ground_energy": ground_energy,
        "energies": energies,
        "energy_errors": errors,
        "cumulative_wall_seconds": wall_seconds,
        "final_energy": energies[-1],
        "final_energy_error": errors[-1],
        "convergence_tolerance": tolerance,
        "converged": reached_step is not None,
        "first_converged_step": reached_step,
        "first_converged_wall_seconds": reached_seconds,
        "closure_evaluations": closure_evaluations,
        "total_wall_seconds": wall_seconds[-1],
        "checkpoint_path": str(checkpoint_path),
    }


def _first_crossover(torch_run: dict[str, Any], jax_run: dict[str, Any]) -> int | None:
    for step, (torch_time, jax_time) in enumerate(
        zip(torch_run["cumulative_wall_seconds"], jax_run["cumulative_wall_seconds"]),
        start=1,
    ):
        if jax_time <= torch_time:
            return step
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wires", default="4,8,12")
    parser.add_argument("--depths", default="2,4,8")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--optimizers", default="adam")
    parser.add_argument("--adam-lr", type=float, default=0.03)
    parser.add_argument("--lbfgs-lr", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-lr",
        type=float,
        help="Explicitly replace the optimizer learning rate after resume.",
    )
    args = parser.parse_args()

    wires = [int(value) for value in args.wires.split(",")]
    depths = [int(value) for value in args.depths.split(",")]
    optimizers = [value.strip().lower() for value in args.optimizers.split(",")]
    if not set(optimizers) <= {"adam", "lbfgs"}:
        raise ValueError("optimizers must contain only adam or lbfgs")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint-interval must be positive")
    checkpoint_dir = args.checkpoint_dir or args.output.with_suffix("").with_name(
        args.output.stem + "_checkpoints"
    )

    ground_energies = {
        n_wires: _exact_ground_energy(n_wires, coupling=0.7, field=0.25)
        for n_wires in wires
    }
    runs: list[dict[str, Any]] = []
    comparisons = []
    for optimizer_name in optimizers:
        for n_wires in wires:
            for layers in depths:
                pair = []
                for backend in ("pytorch", "jax"):
                    result = _train(
                        backend=backend,
                        n_wires=n_wires,
                        layers=layers,
                        steps=args.steps,
                        optimizer_name=optimizer_name,
                        lr=(
                            args.adam_lr if optimizer_name == "adam" else args.lbfgs_lr
                        ),
                        device=args.device,
                        ground_energy=ground_energies[n_wires],
                        tolerance=args.tolerance,
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_interval=args.checkpoint_interval,
                        resume=args.resume,
                        resume_lr=args.resume_lr,
                    )
                    runs.append(result)
                    pair.append(result)
                comparisons.append(
                    {
                        "optimizer": optimizer_name,
                        "n_wires": n_wires,
                        "layers": layers,
                        "cumulative_time_crossover_step": _first_crossover(*pair),
                        "final_energy_difference_abs": abs(
                            pair[0]["final_energy"] - pair[1]["final_energy"]
                        ),
                    }
                )

    payload = {
        "schema": "flagquantum.jax_jit_vqe_training.v1",
        "benchmark": "jax_jit_end_to_end_vqe_training",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "workload": {
            "hamiltonian": "-0.7 * sum(ZZ) - 0.25 * sum(X), open boundary",
            "ansatz": "RX-RY-RZ chain CX plus closing RXX",
            "wires": wires,
            "depths": depths,
            "steps": args.steps,
            "optimizers": optimizers,
            "convergence_tolerance": args.tolerance,
        },
        "methodology": {
            "same_initial_parameters": True,
            "jax_wall_time_includes_compile": True,
            "wall_time_includes_forward_backward_optimizer_sync": True,
            "ground_energy_method": "scipy sparse eigsh",
            "lbfgs_max_iter_per_optimizer_step": 1,
            "checkpoint_interval_steps": args.checkpoint_interval,
            "resume_enabled": args.resume,
            "checkpoint_dir": str(checkpoint_dir),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(torch.device(args.device)),
        },
        "ground_energies": ground_energies,
        "runs": runs,
        "comparisons": comparisons,
        "limitations": [
            "Single-device development evidence, not distributed scalability evidence.",
            "Convergence depends on ansatz, initialization, optimizer, and tolerance.",
            "One seeded trajectory per configuration; release claims require repeated seeds.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"output": str(args.output), "comparisons": comparisons}, indent=2)
    )


if __name__ == "__main__":
    main()
