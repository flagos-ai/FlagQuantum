"""Read energy and actual particle sectors without imposing an unproved constraint."""

import math
from collections import defaultdict
from dataclasses import dataclass

import flagquantum as fq
from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.main import (
    build_program,
)
from flagquantum import synth

from .optimization import Optimization, hamiltonian, optimize


@dataclass(frozen=True)
class Workflow:
    """Actual native optimum and named-register population diagnostics."""

    optimization: Optimization
    program: synth.Program
    energy: float
    sectors: tuple[tuple[int, float], ...]


def run_workflow() -> Workflow:
    """Run the source four-mode variational workflow with a 200-evaluation budget."""
    optimization = optimize()
    program = build_program(optimization.parameters)
    result = fq.run(
        program.circuit, outputs=[fq.expectation(hamiltonian(), name="energy")]
    )
    energy = float(result.expectation("energy").reshape(()))
    if not math.isfinite(energy):
        raise ValueError("Native expectation must be finite.")
    sectors: defaultdict[int, float] = defaultdict(float)
    for (word,), probability in program.probabilities(
        fq.run(program), registers=("state",), threshold=0
    ).items():
        sectors[word.bit_count()] += probability
    return Workflow(optimization, program, energy, tuple(sorted(sectors.items())))
