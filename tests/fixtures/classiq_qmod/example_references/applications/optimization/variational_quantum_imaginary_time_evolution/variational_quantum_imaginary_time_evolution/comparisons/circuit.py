"""Keep original-source verification outside the native user library."""

from pathlib import Path

from examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize

BINDINGS = {"angles": [0.13, -0.29, 0.41, -0.57, 0.68, -0.83]}


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod"), classical_inputs=BINDINGS),
        reference_circuit(tuple(BINDINGS["angles"])),
        ("nodes",),
    )


if __name__ == "__main__":
    run_example()
