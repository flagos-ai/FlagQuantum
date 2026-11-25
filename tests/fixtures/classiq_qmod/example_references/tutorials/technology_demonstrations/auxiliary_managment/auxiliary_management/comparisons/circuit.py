"""Retained original source check outside the native user example library."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.auxiliary_managment.auxiliary_management.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(Path(__file__).with_name("source.qmod"))
    compare_states(program, reference_circuit(), ("out",))


if __name__ == "__main__":
    run_example()
