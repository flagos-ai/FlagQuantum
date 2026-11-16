"""Retain an independent full-state check of the unchanged source."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.approximated_state_preparation.comparisons.circuit import (
    reference_circuit,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("out",),
    )


if __name__ == "__main__":
    run_example()
