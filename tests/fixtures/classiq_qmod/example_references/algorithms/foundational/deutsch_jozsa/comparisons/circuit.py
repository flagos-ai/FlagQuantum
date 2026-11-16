"""Verify the unchanged original predicate against its complete truth table."""

from pathlib import Path

from examples.synthesis.algorithms.foundational.deutsch_jozsa.comparisons.circuit import (
    circuit_reference,
    complex_predicate,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    program = synthesize(Path(__file__).with_name("source.qmod"))
    compare_states(program, circuit_reference(3, complex_predicate), ("x",))


if __name__ == "__main__":
    run_example()
