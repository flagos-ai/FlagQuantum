"""Compare all-Adam with classical-Adam/quantum-block-QNG on hybrid VQE.

This is a single-device exact-statevector benchmark.  A trainable classical
gate controls the angles in every HVA layer while a separate parameter vector
contains the quantum angles.  The benchmark deliberately reports circuit
evaluations and wall time as well as optimizer steps because a QNG step is
more expensive than an Adam step.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import flagquantum.algorithms as fqa
from flagquantum.ops import get_global_precision, set_global_precision


@dataclass
class CircuitCounter:
    evaluations: int = 0

    def call(self, function: Callable[[], torch.Tensor]) -> torch.Tensor:
        self.evaluations += 1
        return function()


def effective_angles(
    classical: torch.Tensor, quantum: torch.Tensor, *, depth: int
) -> torch.Tensor:
    """Apply one bounded classical gain to each HVA layer."""

    if classical.numel() != depth or quantum.numel() % depth:
        raise ValueError("classical/quantum parameter shapes do not match depth")
    per_layer = quantum.numel() // depth
    gains = 2.0 * torch.sigmoid(classical).repeat_interleave(per_layer)
    return gains * quantum


def _block_quantum_metric(
    state_function: Callable[[torch.Tensor], torch.Tensor],
    quantum: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Exact Fubini--Study metric with layer-local off-diagonal blocks removed."""

    flat = quantum.reshape(-1)
    complex_dtype = (
        torch.complex128 if quantum.dtype == torch.float64 else torch.complex64
    )

    def normalized(value: torch.Tensor) -> torch.Tensor:
        state = state_function(value.reshape_as(quantum)).reshape(-1).to(complex_dtype)
        return state / torch.linalg.vector_norm(state)

    real_jacobian = torch.autograd.functional.jacobian(
        lambda value: normalized(value).real, flat, create_graph=False
    ).reshape(-1, flat.numel())
    imag_jacobian = torch.autograd.functional.jacobian(
        lambda value: normalized(value).imag, flat, create_graph=False
    ).reshape(-1, flat.numel())
    jacobian = torch.complex(real_jacobian, imag_jacobian)
    state = normalized(flat).detach()
    connection = torch.conj(state) @ jacobian
    full = torch.real(
        torch.conj(jacobian).transpose(0, 1) @ jacobian
        - torch.outer(torch.conj(connection), connection)
    )
    full = 0.5 * (full + full.transpose(0, 1))
    blocked = torch.zeros_like(full)
    for start in range(0, flat.numel(), block_size):
        stop = min(start + block_size, flat.numel())
        blocked[start:stop, start:stop] = full[start:stop, start:stop]
    return blocked


def run_experiment(
    *,
    method: str,
    n_wires: int,
    depth: int,
    steps: int,
    seed: int,
    classical_lr: float,
    quantum_lr: float,
    damping: float,
    dtype: torch.dtype,
) -> dict:
    if method not in {"all_adam", "adam_block_qng"}:
        raise ValueError(f"unsupported method {method!r}")

    parameter_count = fqa.heisenberg_hva_parameter_count(n_wires, depth)
    per_layer = parameter_count // depth
    generator = torch.Generator().manual_seed(seed)
    classical = torch.zeros(depth, dtype=dtype, requires_grad=True)
    quantum = (
        0.02 * torch.randn(parameter_count, generator=generator, dtype=dtype)
    ).requires_grad_()
    hamiltonian = fqa.heisenberg_chain_hamiltonian(n_wires)
    exact = float(hamiltonian.ground_energy())
    counter = CircuitCounter()

    def state(c: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        angles = effective_angles(c, q, depth=depth)
        return fqa.heisenberg_hva(
            n_wires,
            depth,
            angles,
            parameterization="bond_resolved_phase",
            initial_state="dimer_singlet",
        ).state().reshape(-1)

    def energy(c: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return counter.call(
            lambda: hamiltonian.expectation(state(c, q)).sum().real
        )

    classical_optimizer = torch.optim.Adam([classical], lr=classical_lr)
    quantum_optimizer = (
        torch.optim.Adam([quantum], lr=quantum_lr)
        if method == "all_adam"
        else None
    )
    started = time.perf_counter()

    def record(step: int, value: torch.Tensor) -> dict:
        scalar = float(value.detach())
        return {
            "optimizer_step": step,
            "circuit_evaluations": counter.evaluations,
            "wall_time_seconds": time.perf_counter() - started,
            "energy": scalar,
            "relative_error": (scalar - exact) / abs(exact),
        }

    trace = [record(0, energy(classical, quantum))]
    for optimizer_step in range(1, steps + 1):
        classical_optimizer.zero_grad(set_to_none=True)
        if quantum_optimizer is not None:
            quantum_optimizer.zero_grad(set_to_none=True)

        loss = energy(classical, quantum)
        classical_gradient, quantum_gradient = torch.autograd.grad(
            loss, (classical, quantum)
        )
        frozen_classical = classical.detach().clone()
        classical.grad = classical_gradient
        classical_optimizer.step()

        if quantum_optimizer is not None:
            quantum.grad = quantum_gradient
            quantum_optimizer.step()
        else:
            # Both group updates use information from the same outer-step
            # parameter point; neither sees the other's just-applied update.

            def counted_state(value: torch.Tensor) -> torch.Tensor:
                return counter.call(lambda: state(frozen_classical, value))

            metric = _block_quantum_metric(
                counted_state, quantum, block_size=per_layer
            )
            identity = torch.eye(
                parameter_count, dtype=metric.dtype, device=metric.device
            )
            direction = torch.linalg.solve(
                metric + damping * identity, quantum_gradient.reshape(-1)
            )
            with torch.no_grad():
                quantum.add_(-quantum_lr * direction.reshape_as(quantum))

        trace.append(record(optimizer_step, energy(classical, quantum)))

    return {
        "method": method,
        "optimizer_assignment": {
            "classical": "adam",
            "quantum": "adam" if method == "all_adam" else "block_qng",
        },
        "final_energy": trace[-1]["energy"],
        "final_relative_error": trace[-1]["relative_error"],
        "total_circuit_evaluations": counter.evaluations,
        "total_wall_time_seconds": trace[-1]["wall_time_seconds"],
        "classical_parameters": classical.detach().tolist(),
        "quantum_parameters": quantum.detach().tolist(),
        "trace": trace,
    }


def write_outputs(payload: dict, json_output: Path, csv_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for experiment in payload["experiments"]:
        for point in experiment["trace"]:
            rows.append({"method": experiment["method"], **point})
    with csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=260720)
    parser.add_argument("--classical-lr", type=float, default=0.03)
    parser.add_argument("--quantum-lr", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    dtype = torch.float64 if args.precision == "float64" else torch.float32
    previous_precision = get_global_precision()
    set_global_precision(torch.complex128 if dtype == torch.float64 else torch.complex64)
    try:
        experiments = [
            run_experiment(
                method=method,
                n_wires=args.n_wires,
                depth=args.depth,
                steps=args.steps,
                seed=args.seed,
                classical_lr=args.classical_lr,
                quantum_lr=args.quantum_lr,
                damping=args.damping,
                dtype=dtype,
            )
            for method in ("all_adam", "adam_block_qng")
        ]
        exact = float(fqa.heisenberg_chain_hamiltonian(args.n_wires).ground_energy())
        payload = {
            "schema": "flagquantum.hybrid_classical_quantum_optimizer.v1",
            "workload": "classically_gated_heisenberg_hva_vqe",
            "execution_semantics": "single_device_fast_path",
            "scalability_claim_allowed": False,
            "n_wires": args.n_wires,
            "depth": args.depth,
            "parameterization": "bond_resolved_phase_with_classical_layer_gates",
            "classical_parameter_count": args.depth,
            "quantum_parameter_count": fqa.heisenberg_hva_parameter_count(
                args.n_wires, args.depth
            ),
            "precision": args.precision,
            "seed": args.seed,
            "steps": args.steps,
            "classical_learning_rate": args.classical_lr,
            "quantum_learning_rate": args.quantum_lr,
            "qng_damping": args.damping,
            "exact_ground_energy": exact,
            "circuit_evaluation_definition": (
                "one objective-state construction or one QNG metric-state construction"
            ),
            "wall_time_definition": (
                "cumulative monotonic time including forward, backward, metric, and update"
            ),
            "experiments": experiments,
        }
        write_outputs(payload, args.json_output, args.csv_output)
        print(json.dumps({
            experiment["method"]: {
                "final_relative_error": experiment["final_relative_error"],
                "circuit_evaluations": experiment["total_circuit_evaluations"],
                "wall_time_seconds": experiment["total_wall_time_seconds"],
            }
            for experiment in experiments
        }, indent=2))
    finally:
        set_global_precision(previous_precision)


if __name__ == "__main__":
    main()
