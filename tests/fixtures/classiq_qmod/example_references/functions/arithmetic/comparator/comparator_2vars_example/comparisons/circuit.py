"""Verify the immutable original against the independent Circuit oracle."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.comparator.comparator_2vars_example.comparisons.circuit import (
    reference_circuit,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("a", "b", "res"),
    )


if __name__ == "__main__":
    run_example()
