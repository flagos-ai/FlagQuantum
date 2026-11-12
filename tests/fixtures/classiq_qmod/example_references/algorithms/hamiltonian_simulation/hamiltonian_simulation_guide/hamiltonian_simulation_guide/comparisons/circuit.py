"""Test-only independent Circuit check of the original Pauli source bytes."""

from pathlib import Path

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_guide.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    for time in (0.37, -0.81):
        program = synth.synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"t": time},
            max_workspace=0,
        )
        compare_states(program, reference_circuit(time), tuple(program.registers))


if __name__ == "__main__":
    run_example()
