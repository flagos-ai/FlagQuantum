"""Retain the original integer-parameter source as a test-only comparison."""

from pathlib import Path

from examples.synthesis.applications.physical_systems.fermi_hubbard_model_1d.fermi_hubbard_1d.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    for num_iter in (1, 2):
        program = synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"num_iter": num_iter},
            max_workspace=0,
        )
        compare_states(program, reference_circuit(num_iter), ("qba",))


if __name__ == "__main__":
    run_example()
