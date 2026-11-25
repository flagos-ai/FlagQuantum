"""Frozen closed-callback program against an independent Circuit."""

from pathlib import Path

from examples.synthesis.algorithms.foundational.simon.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("x", "res"),
    )


if __name__ == "__main__":
    run_example()
