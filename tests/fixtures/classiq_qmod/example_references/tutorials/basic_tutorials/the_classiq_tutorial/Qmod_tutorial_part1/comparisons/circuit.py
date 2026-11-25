"""Test-only source identity and independent Circuit check."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part1.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(Path(__file__).with_name("source.qmod"))
    compare_states(program, reference_circuit(), ("x",))


if __name__ == "__main__":
    run_example()
