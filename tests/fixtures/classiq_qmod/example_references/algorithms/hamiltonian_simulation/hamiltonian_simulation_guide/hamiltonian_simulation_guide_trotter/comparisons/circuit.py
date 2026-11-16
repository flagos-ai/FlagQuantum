"""Check the retained original against independent dense Circuit exponentials."""

from pathlib import Path

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.hamiltonian_simulation_guide_trotter.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("qba",),
    )


if __name__ == "__main__":
    run_example()
