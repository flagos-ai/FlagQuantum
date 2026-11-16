"""Compare the frozen original circle walk with independent modular shifts."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.quantum_walk_circle.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    compare_states(
        synth.synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"t": 2}
        ),
        reference_circuit(),
        ("x", "coin"),
    )


if __name__ == "__main__":
    run_example()
