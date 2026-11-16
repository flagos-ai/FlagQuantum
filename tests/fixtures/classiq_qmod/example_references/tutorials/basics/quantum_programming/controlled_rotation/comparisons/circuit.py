"""Verify the frozen source's controlled rotation against its dense reference."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.quantum_programming.controlled_rotation.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    compare_states(
        synth.synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("res",),
    )


if __name__ == "__main__":
    run_example()
