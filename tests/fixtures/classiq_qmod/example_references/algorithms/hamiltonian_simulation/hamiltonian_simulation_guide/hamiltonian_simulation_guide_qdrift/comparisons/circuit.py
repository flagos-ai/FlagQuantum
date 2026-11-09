"""Test-only independent Circuit check of the original Pauli source bytes."""

from pathlib import Path

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.hamiltonian_simulation_guide_qdrift.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(Path(__file__).with_name("source.qmod"), max_workspace=0)
    compare_states(program, reference_circuit(), tuple(program.registers))


if __name__ == "__main__":
    run_example()
