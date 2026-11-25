"""Retained-source BB84 check outside the native user example directory."""

from pathlib import Path

from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("qba",),
    )


if __name__ == "__main__":
    run_example()
