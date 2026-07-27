"""Diagnose small Heisenberg VQE convergence with staged optimizers."""

from __future__ import annotations

import argparse

import torch

import flagquantum as fq
import flagquantum.algorithms as fqa
from flagquantum.algorithms.optimization import OptimizationStage
from flagquantum.ops import set_global_precision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="grow depths 1..depth with zero-initialized identity layers",
    )
    parser.add_argument("--convergence-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--seed", type=int, default=260720)
    parser.add_argument(
        "--schedule",
        choices=("adam", "adam_lbfgs", "adam_qng", "rotosolve"),
        default="adam_qng",
    )
    args = parser.parse_args()

    real_dtype = torch.float64 if args.precision == "float64" else torch.float32
    set_global_precision(
        torch.complex128 if real_dtype == torch.float64 else torch.complex64
    )

    initial_depth = 1 if args.layerwise else args.depth
    count = fqa.heisenberg_hva_parameter_count(args.n_qubits, initial_depth)
    generator = torch.Generator().manual_seed(args.seed)
    initial = 0.02 * torch.randn(count, generator=generator, dtype=real_dtype)
    hamiltonian = fqa.heisenberg_chain_hamiltonian(args.n_qubits)

    def build_at_depth(depth: int, parameters: torch.Tensor) -> fq.Circuit:
        return fqa.heisenberg_hva(
            args.n_qubits,
            depth,
            parameters,
            parameterization="bond_resolved_phase",
            initial_state="dimer_singlet",
        )

    def builder(parameters: torch.Tensor) -> fq.Circuit:
        return build_at_depth(args.depth, parameters)

    final_count = fqa.heisenberg_hva_parameter_count(args.n_qubits, args.depth)
    block_size = final_count // args.depth
    schedules = {
        "adam": (OptimizationStage("quantum", "adam", 100, lr=0.02),),
        "adam_lbfgs": (
            OptimizationStage("quantum", "adam", 100, lr=0.02),
            OptimizationStage(
                "quantum", "lbfgs", 1, lr=0.8, max_iter=150, history_size=10
            ),
        ),
        "adam_qng": (
            OptimizationStage("quantum", "adam", 40, lr=0.02),
            OptimizationStage(
                "quantum",
                "qng",
                20,
                lr=0.05,
                damping=1e-3,
                block_size=block_size,
            ),
        ),
        "rotosolve": (OptimizationStage("quantum", "rotosolve", 3),),
    }
    if args.layerwise:
        continuation = fqa.run_layerwise_vqe(
            build_at_depth,
            lambda depth: fqa.heisenberg_hva_parameter_count(args.n_qubits, depth),
            initial,
            hamiltonian,
            depths=tuple(range(1, args.depth + 1)),
            stages=schedules[args.schedule],
        )
        result = continuation.stages[-1]
        depth_energies = [stage.history[-1] for stage in continuation.stages]
    else:
        result = fqa.run_hybrid_vqe(
            builder, initial, hamiltonian, stages=schedules[args.schedule]
        )
        depth_energies = [result.history[-1]]
    exact_energy = float(hamiltonian.ground_energy())
    final_energy = result.history[-1]
    print(
        {
            "n_qubits": args.n_qubits,
            "depth": args.depth,
            "parameterization": "bond_resolved_phase",
            "initial_state": "dimer_singlet",
            "schedule": args.schedule,
            "precision": args.precision,
            "layerwise": args.layerwise,
            "depth_energies": depth_energies,
            "first_recorded_energy": result.records[0].loss,
            "final_energy": final_energy,
            "exact_energy": exact_energy,
            "relative_error": (final_energy - exact_energy) / abs(exact_energy),
            "converged": (
                (final_energy - exact_energy) / abs(exact_energy)
                <= args.convergence_tolerance
            ),
            "objective_evaluations": result.evaluations,
        }
    )


if __name__ == "__main__":
    main()
