"""Optimize the original six-spin periodic Ising energy with native qfuncs."""

from dataclasses import dataclass

import flagquantum as fq
from flagquantum import synth

from .plots import plot_results
from .sampling import Sample, sample
from .workflow import Optimization, energy, optimize, source_program


@dataclass(frozen=True)
class Workflow:
    """Exact native optimization followed by two explicit finite sample tables."""

    optimization: Optimization
    program: synth.Program
    uniform_program: synth.Program
    optimized_samples: tuple[Sample, ...]
    uniform_samples: tuple[Sample, ...]


def read_workflow(
    optimization: Optimization,
    program: synth.Program,
    uniform_program: synth.Program,
) -> Workflow:
    """Read actual native marginals and report solutions without wire decoding."""
    optimized_rows = tuple(
        (key[0], probability)
        for key, probability in program.probabilities(
            fq.run(program), registers=("v",), threshold=0
        ).items()
    )
    uniform_rows = tuple(
        (key[0], probability)
        for key, probability in uniform_program.probabilities(
            fq.run(uniform_program), registers=("v",), threshold=0
        ).items()
    )
    optimized_samples = sample(optimized_rows, seed=123)
    uniform_samples = sample(uniform_rows, seed=456)
    ranked = sorted(
        (row for row in optimized_samples if row.count), key=lambda row: row.energy
    )
    print("Initial tail cost:", optimization.initial_cost)
    print("Best tail cost:", optimization.best_cost)
    print("Optimizer termination:", optimization.termination)
    print("Objective evaluations:", len(optimization.trace))
    print("Best five observed rows (spins, energy, count, probability):")
    for row in ranked[:5]:
        print(row.spins, row.energy, row.count, row.probability)
    if not ranked:
        raise ValueError("No actual sample outcomes were recorded.")
    print("Best observed spin configuration:", ranked[0].spins)
    print("Classical validation minimum:", min(energy(word) for word in range(64)))
    print("Optimized resources:", program.report["resources"])
    print("Uniform resources:", uniform_program.report["resources"])
    return Workflow(
        optimization, program, uniform_program, optimized_samples, uniform_samples
    )


def run_workflow() -> Workflow:
    """Run the source five-layer, 100-evaluation workflow on local exact states."""
    optimization = optimize()
    return read_workflow(
        optimization,
        source_program(optimization.parameters),
        source_program((0.0,) * 10),
    )


def run_example() -> None:
    workflow = run_workflow()
    plot_results(
        workflow.optimization, workflow.optimized_samples, workflow.uniform_samples
    )


if __name__ == "__main__":
    run_example()
