"""Retained original source check outside the native user example library."""

from pathlib import Path

from examples.synthesis.applications.optimization.adapt_qaoa.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(
        Path(__file__).with_name("source.qmod"), classical_inputs={"params": [0.1, 0.2]}
    )
    compare_states(program, reference_circuit(), ("v",))


if __name__ == "__main__":
    run_example()
