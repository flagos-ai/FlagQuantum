"""Systematic MLP/quantum-optimizer study for a classically conditioned VQE."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hybrid_classical_quantum_optimizer import (
    CircuitCounter,
    _block_quantum_metric,
)
import flagquantum.algorithms as fqa
from flagquantum.ops import get_global_precision, set_global_precision


METHODS = ("adam", "block_qng", "lbfgs", "spsa")


def parameter_features(n_wires: int, depth: int, dtype: torch.dtype) -> torch.Tensor:
    """Encode layer, gate family, and normalized bond/site position."""

    rows = []
    for layer in range(depth):
        layer_coordinate = 0.0 if depth == 1 else 2.0 * layer / (depth - 1) - 1.0
        for family in range(4):
            width = n_wires if family == 3 else n_wires - 1
            for position in range(width):
                coordinate = 0.0 if width == 1 else 2.0 * position / (width - 1) - 1.0
                one_hot = [float(family == index) for index in range(4)]
                rows.append(
                    [
                        layer_coordinate,
                        coordinate,
                        float(position in {0, width - 1}),
                        *one_hot,
                        1.0,
                    ]
                )
    return torch.tensor(rows, dtype=dtype)


def make_conditioner(dtype: torch.dtype, seed: int, hidden: int) -> torch.nn.Module:
    """Create a small MLP whose initial output is the identity conditioning."""

    with torch.random.fork_rng():
        torch.manual_seed(seed + 17)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, hidden, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2, dtype=dtype),
        )
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def conditioned_angles(
    conditioner: torch.nn.Module,
    features: torch.Tensor,
    quantum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = conditioner(features)
    gain = 2.0 * torch.sigmoid(output[:, 0])
    bias = 0.10 * torch.tanh(output[:, 1])
    return gain * quantum + bias, gain, bias


def _best_at_budget(trace: list[dict], budget: int) -> float | None:
    eligible = [point["relative_error"] for point in trace if point["circuit_evaluations"] <= budget]
    return min(eligible) if eligible else None


def _first_target(trace: list[dict], target: float) -> dict | None:
    return next((point for point in trace if point["relative_error"] <= target), None)


def run_method(
    *,
    method: str,
    n_wires: int,
    depth: int,
    steps: int,
    seed: int,
    hidden: int,
    classical_lr: float,
    quantum_lr: float,
    damping: float,
    spsa_perturbation: float,
    lbfgs_max_iter: int,
    dtype: torch.dtype,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    features = parameter_features(n_wires, depth, dtype)
    count = fqa.heisenberg_hva_parameter_count(n_wires, depth)
    if features.shape != (count, 8):
        raise AssertionError("MLP features do not align with HVA parameters")
    per_layer = count // depth
    generator = torch.Generator().manual_seed(seed)
    quantum = (0.02 * torch.randn(count, generator=generator, dtype=dtype)).requires_grad_()
    conditioner = make_conditioner(dtype, seed, hidden)
    classical_optimizer = torch.optim.Adam(conditioner.parameters(), lr=classical_lr)
    quantum_optimizer = (
        torch.optim.Adam([quantum], lr=quantum_lr) if method == "adam" else None
    )
    hamiltonian = fqa.heisenberg_chain_hamiltonian(n_wires)
    exact = float(hamiltonian.ground_energy())
    counter = CircuitCounter()

    def state_from_angles(angles: torch.Tensor) -> torch.Tensor:
        return fqa.heisenberg_hva(
            n_wires,
            depth,
            angles,
            parameterization="bond_resolved_phase",
            initial_state="dimer_singlet",
        ).state().reshape(-1)

    def energy_from_angles(angles: torch.Tensor) -> torch.Tensor:
        return counter.call(lambda: hamiltonian.expectation(state_from_angles(angles)).sum().real)

    def live_energy() -> torch.Tensor:
        angles, _, _ = conditioned_angles(conditioner, features, quantum)
        return energy_from_angles(angles)

    started = time.perf_counter()

    def record(step: int, value: torch.Tensor) -> dict:
        scalar = float(value.detach())
        return {
            "optimizer_step": step,
            "circuit_evaluations": counter.evaluations,
            "wall_time_seconds": time.perf_counter() - started,
            "energy": scalar,
            "relative_error": max(0.0, (scalar - exact) / abs(exact)),
        }

    trace = [record(0, live_energy())]
    spsa_generator = torch.Generator().manual_seed(seed + 1009)
    for step in range(1, steps + 1):
        classical_optimizer.zero_grad(set_to_none=True)
        if method == "adam":
            assert quantum_optimizer is not None
            quantum_optimizer.zero_grad(set_to_none=True)
        loss = live_energy()
        classical_parameters = tuple(conditioner.parameters())
        gradients = torch.autograd.grad(loss, (*classical_parameters, quantum))
        quantum_gradient = gradients[-1]
        for parameter, gradient in zip(classical_parameters, gradients[:-1]):
            parameter.grad = gradient

        # All quantum optimizers use the same pre-update MLP transformation.
        with torch.no_grad():
            _, frozen_gain, frozen_bias = conditioned_angles(conditioner, features, quantum)
            frozen_gain = frozen_gain.clone()
            frozen_bias = frozen_bias.clone()
        classical_optimizer.step()

        def frozen_energy(value: torch.Tensor) -> torch.Tensor:
            return energy_from_angles(frozen_gain * value + frozen_bias)

        if method == "adam":
            quantum.grad = quantum_gradient
            quantum_optimizer.step()
        elif method == "block_qng":
            def counted_state(value: torch.Tensor) -> torch.Tensor:
                return counter.call(
                    lambda: state_from_angles(frozen_gain * value + frozen_bias)
                )

            metric = _block_quantum_metric(counted_state, quantum, block_size=per_layer)
            identity = torch.eye(count, dtype=metric.dtype, device=metric.device)
            direction = torch.linalg.solve(metric + damping * identity, quantum_gradient)
            with torch.no_grad():
                quantum.add_(-quantum_lr * direction)
        elif method == "lbfgs":
            # The MLP changes the effective quantum objective every outer
            # step, so stale quasi-Newton curvature must not cross that
            # boundary.
            local_lbfgs = torch.optim.LBFGS(
                [quantum],
                lr=quantum_lr,
                max_iter=lbfgs_max_iter,
                history_size=10,
                tolerance_grad=1e-10,
                tolerance_change=1e-12,
            )

            def closure() -> torch.Tensor:
                local_lbfgs.zero_grad(set_to_none=True)
                value = frozen_energy(quantum)
                value.backward()
                return value

            local_lbfgs.step(closure)
        else:
            signs = torch.randint(
                0, 2, quantum.shape, generator=spsa_generator, dtype=torch.int64
            ).to(dtype=dtype).mul_(2).sub_(1)
            plus = frozen_energy(quantum + spsa_perturbation * signs)
            minus = frozen_energy(quantum - spsa_perturbation * signs)
            estimate = (plus - minus) * signs / (2.0 * spsa_perturbation)
            with torch.no_grad():
                quantum.add_(-quantum_lr * estimate)
        trace.append(record(step, live_energy()))

    targets = (1e-2, 1e-3, 1e-4, 1e-5)
    return {
        "method": method,
        "optimizer_assignment": {"classical": "adam", "quantum": method},
        "final_energy": trace[-1]["energy"],
        "final_relative_error": trace[-1]["relative_error"],
        "best_relative_error": min(point["relative_error"] for point in trace),
        "total_circuit_evaluations": counter.evaluations,
        "total_wall_time_seconds": trace[-1]["wall_time_seconds"],
        "target_crossings": {
            str(target): _first_target(trace, target) for target in targets
        },
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=6)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=260720)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--classical-lr", type=float, default=0.01)
    parser.add_argument("--quantum-lr", type=float, default=0.03)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--spsa-perturbation", type=float, default=0.08)
    parser.add_argument("--lbfgs-max-iter", type=int, default=4)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    previous = get_global_precision()
    set_global_precision(torch.complex128 if dtype == torch.float64 else torch.complex64)
    try:
        experiments = [
            run_method(
                method=method,
                n_wires=args.n_wires,
                depth=args.depth,
                steps=args.steps,
                seed=args.seed,
                hidden=args.hidden,
                classical_lr=args.classical_lr,
                quantum_lr=args.quantum_lr,
                damping=args.damping,
                spsa_perturbation=args.spsa_perturbation,
                lbfgs_max_iter=args.lbfgs_max_iter,
                dtype=dtype,
            )
            for method in args.methods
        ]
        budgets = sorted({50, 100, 150, *[e["total_circuit_evaluations"] for e in experiments]})
        payload = {
            "schema": "flagquantum.hybrid_mlp_quantum_optimizer_suite.v1",
            "workload": "mlp_conditioned_heisenberg_hva_vqe",
            "execution_semantics": "single_device_fast_path",
            "scalability_claim_allowed": False,
            "n_wires": args.n_wires,
            "depth": args.depth,
            "steps": args.steps,
            "precision": args.precision,
            "seed": args.seed,
            "initial_state": "dimer_singlet",
            "parameterization": "bond_resolved_phase",
            "classical_model": {"type": "mlp", "features": 8, "hidden": args.hidden, "outputs": 2},
            "classical_parameter_count": 8 * args.hidden + args.hidden + 2 * args.hidden + 2,
            "quantum_parameter_count": fqa.heisenberg_hva_parameter_count(args.n_wires, args.depth),
            "exact_ground_energy": float(fqa.heisenberg_chain_hamiltonian(args.n_wires).ground_energy()),
            "circuit_evaluation_definition": "one objective-state or QNG metric-state construction",
            "wall_time_definition": "forward, backward, metric/closure, optimizer, and recording",
            "equal_evaluation_budget_best_error": {
                str(budget): {
                    experiment["method"]: _best_at_budget(experiment["trace"], budget)
                    for experiment in experiments
                }
                for budget in budgets
            },
            "experiments": experiments,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows = [
            {"method": experiment["method"], **point}
            for experiment in experiments
            for point in experiment["trace"]
        ]
        with args.csv_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({e["method"]: {
            "best_relative_error": e["best_relative_error"],
            "evaluations": e["total_circuit_evaluations"],
            "seconds": e["total_wall_time_seconds"],
        } for e in experiments}, indent=2))
    finally:
        set_global_precision(previous)


if __name__ == "__main__":
    main()
