"""Compare full states with independent CDF sampling and opposite parity mapping."""

from pathlib import Path

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.hamiltonian_simulation_guide_qdrift.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.pauli_evolution import load_terms, reference_sequence
from examples.synthesis.pauli_evolution import reference_circuit as pauli_circuit


def reference_circuit(time: float = 1.0) -> fq.Circuit:
    width, terms = load_terms(Path(__file__).parents[1] / "hamiltonian.json")
    return pauli_circuit(width, terms, reference_sequence(terms, time, 288))


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("qba",))


if __name__ == "__main__":
    run_example()
