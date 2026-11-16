"""Check the retained original against an independent complete truth table."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.arithmetic_expressions.comparisons.circuit import (
    reference_circuit,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("x", "y", "z"),
    )


if __name__ == "__main__":
    run_example()
