"""Verify the untouched affine-rotation original against an independent matrix."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.linear_pauli_rotations.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    compare_states(
        synth.synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("x", "ind"),
    )


if __name__ == "__main__":
    run_example()
