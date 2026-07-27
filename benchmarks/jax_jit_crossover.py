"""Measure the cold-start/steady-state crossover of JAX and native PyTorch.

The parent process launches a fresh worker for every qubit-size/repetition pair,
so an in-process XLA executable cache cannot turn a cold compile into a warm hit.
This is a single-device development benchmark, not scalability evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402


def _params(n_wires: int, layers: int, device: str) -> torch.Tensor:
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


def _pytorch_value_and_grad(
    seed: torch.Tensor, hamiltonian: fq.Hamiltonian | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    values = seed.detach().clone().requires_grad_(True)
    circuit = _circuit(values)
    loss = (
        circuit.expectation_z().sum()
        if hamiltonian is None
        else hamiltonian.expectation(circuit).sum()
    )
    loss.backward()
    grad = values.grad if values.grad is not None else torch.zeros_like(values)
    return loss.detach(), grad.detach()


def _jax_value_and_grad(
    kernel: Any, seed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    values = seed.detach().clone().requires_grad_(True)
    loss = kernel(values).sum()
    loss.backward()
    grad = values.grad if values.grad is not None else torch.zeros_like(values)
    return loss.detach(), grad.detach()


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _timed(
    callable_: Any, *, device: str
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    _sync(device)
    started = time.perf_counter()
    result = callable_()
    _sync(device)
    elapsed = time.perf_counter() - started
    return elapsed, result


def _worker(
    n_wires: int,
    layers: int,
    steady_repetitions: int,
    device: str,
    task: str,
) -> dict[str, Any]:
    seed = _params(n_wires, layers, device)
    hamiltonian = (
        fq.transverse_field_ising(n_wires, coupling=0.7, field=0.25, periodic=False)
        if task == "tfim_vqe"
        else None
    )

    pytorch_first, pytorch_reference = _timed(
        lambda: _pytorch_value_and_grad(seed, hamiltonian), device=device
    )
    pytorch_samples = [
        _timed(lambda: _pytorch_value_and_grad(seed, hamiltonian), device=device)[0]
        for _ in range(steady_repetitions)
    ]

    _sync(device)
    build_started = time.perf_counter()
    kernel = fq.compile_quantum_kernel(
        _circuit,
        seed,
        backend="jax",
        interface="torch",
        mode="statevector",
        n_wires=n_wires,
        observable="z_sum",
        hamiltonian=hamiltonian,
        jit=True,
        compute_dtype="complex64",
    )
    _sync(device)
    jax_build = time.perf_counter() - build_started
    jax_first, jax_reference = _timed(
        lambda: _jax_value_and_grad(kernel, seed), device=device
    )
    jax_samples = [
        _timed(lambda: _jax_value_and_grad(kernel, seed), device=device)[0]
        for _ in range(steady_repetitions)
    ]

    loss_error = abs(float(pytorch_reference[0]) - float(jax_reference[0]))
    grad_error = float(
        torch.max(torch.abs(pytorch_reference[1] - jax_reference[1])).item()
    )
    return {
        "n_wires": n_wires,
        "layers": layers,
        "device": device,
        "task": task,
        "pytorch_first_seconds": pytorch_first,
        "pytorch_steady_samples_seconds": pytorch_samples,
        "jax_kernel_build_seconds": jax_build,
        "jax_first_value_grad_seconds": jax_first,
        "jax_cold_total_seconds": jax_build + jax_first,
        "jax_steady_samples_seconds": jax_samples,
        "loss_abs_error": loss_error,
        "grad_max_abs_error": grad_error,
        "correctness_passed": loss_error <= 1e-4 and grad_error <= 1e-4,
        "kernel_summary": kernel.summary(),
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _break_even_calls(
    *, pytorch_first: float, pytorch_steady: float, jax_cold: float, jax_steady: float
) -> int | None:
    for calls in range(1, 100_001):
        pytorch_total = pytorch_first + max(0, calls - 1) * pytorch_steady
        jax_total = jax_cold + max(0, calls - 1) * jax_steady
        if jax_total <= pytorch_total:
            return calls
    return None


def _aggregate(n_wires: int, repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    pytorch_first = _median([row["pytorch_first_seconds"] for row in repetitions])
    pytorch_steady_samples = [
        sample
        for row in repetitions
        for sample in row["pytorch_steady_samples_seconds"]
    ]
    jax_build = _median([row["jax_kernel_build_seconds"] for row in repetitions])
    jax_first = _median([row["jax_first_value_grad_seconds"] for row in repetitions])
    jax_cold = _median([row["jax_cold_total_seconds"] for row in repetitions])
    jax_steady_samples = [
        sample for row in repetitions for sample in row["jax_steady_samples_seconds"]
    ]
    pytorch_steady = _median(pytorch_steady_samples)
    jax_steady = _median(jax_steady_samples)
    return {
        "n_wires": n_wires,
        "state_dimension": 2**n_wires,
        "pytorch_first_seconds_median": pytorch_first,
        "pytorch_steady_seconds_median": pytorch_steady,
        "pytorch_steady_samples_seconds": pytorch_steady_samples,
        "jax_kernel_build_seconds_median": jax_build,
        "jax_first_value_grad_seconds_median": jax_first,
        "jax_cold_total_seconds_median": jax_cold,
        "jax_steady_seconds_median": jax_steady,
        "jax_steady_samples_seconds": jax_steady_samples,
        "steady_speedup_jax_over_pytorch": pytorch_steady / jax_steady,
        "jax_compile_tax_in_steady_steps": jax_cold / jax_steady,
        "break_even_calls": _break_even_calls(
            pytorch_first=pytorch_first,
            pytorch_steady=pytorch_steady,
            jax_cold=jax_cold,
            jax_steady=jax_steady,
        ),
        "loss_abs_error_max": max(row["loss_abs_error"] for row in repetitions),
        "grad_max_abs_error_max": max(row["grad_max_abs_error"] for row in repetitions),
        "correctness_passed": all(row["correctness_passed"] for row in repetitions),
        "cold_repetitions": repetitions,
    }


def _run_fresh_worker(
    *,
    n_wires: int,
    layers: int,
    steady_repetitions: int,
    cache_dir: Path,
    device: str,
    task: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-wire",
        str(n_wires),
        "--layers",
        str(layers),
        "--steady-repetitions",
        str(steady_repetitions),
        "--device",
        device,
        "--task",
        task,
    ]
    env = dict(os.environ)
    env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    env["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cold worker failed for {n_wires} wires with exit "
            f"{completed.returncode}:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wires", default="4,6,8,10,12")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--cold-repetitions", type=int, default=3)
    parser.add_argument("--steady-repetitions", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=("z_sum", "tfim_vqe"), default="z_sum")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-wire", type=int)
    args = parser.parse_args()

    if args.worker_wire is not None:
        print(
            json.dumps(
                _worker(
                    args.worker_wire,
                    args.layers,
                    args.steady_repetitions,
                    args.device,
                    args.task,
                )
            )
        )
        return

    wire_counts = [
        int(value.strip()) for value in args.wires.split(",") if value.strip()
    ]
    depths = (
        [int(value.strip()) for value in args.depths.split(",") if value.strip()]
        if args.depths
        else [args.layers]
    )
    with tempfile.TemporaryDirectory(prefix="fq-jax-jit-") as temporary:
        temporary_root = Path(temporary)
        measurements = []
        for layers in depths:
            for n_wires in wire_counts:
                repetitions = []
                for repetition in range(args.cold_repetitions):
                    cache_dir = temporary_root / f"d{layers}-q{n_wires}-r{repetition}"
                    cache_dir.mkdir(parents=True)
                    repetitions.append(
                        _run_fresh_worker(
                            n_wires=n_wires,
                            layers=layers,
                            steady_repetitions=args.steady_repetitions,
                            cache_dir=cache_dir,
                            device=args.device,
                            task=args.task,
                        )
                    )
                measurement = _aggregate(n_wires, repetitions)
                measurement["layers"] = layers
                measurements.append(measurement)

    payload = {
        "schema": "flagquantum.jax_jit_crossover.v1",
        "benchmark": "jax_jit_cold_vs_steady",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "claim_scope": "development_single_device_jax_kernel_acceleration",
        "workload": {
            "mode": "statevector",
            "task": args.task,
            "observable": (
                "transverse_field_ising_energy" if args.task == "tfim_vqe" else "z_sum"
            ),
            "depths": depths,
            "layers": depths[0] if len(depths) == 1 else None,
            "hamiltonian": (
                "-0.7 * sum(ZZ) - 0.25 * sum(X), open boundary"
                if args.task == "tfim_vqe"
                else None
            ),
            "parameterization": "RX-RY-RZ chain CX plus closing RXX",
            "includes_forward_backward": True,
            "dtype": "complex64",
        },
        "methodology": {
            "cold_repetitions": args.cold_repetitions,
            "steady_repetitions_per_cold_process": args.steady_repetitions,
            "cold_process_isolation": True,
            "persistent_compilation_cache_disabled": True,
            "jax_cold_definition": "kernel construction plus first value-and-gradient call",
            "steady_definition": "median value-and-gradient latency after first call",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(torch.device(args.device))
                if args.device.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
        },
        "measurements": measurements,
        "correctness_passed": all(row["correctness_passed"] for row in measurements),
        "limitations": [
            "Development evidence from the current single-device environment, not release certification.",
            "JAX persistent compilation cache is disabled to expose cold-start cost.",
            "Results do not imply distributed scalability or universal backend superiority.",
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
