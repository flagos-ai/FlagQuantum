"""Check the retained original against independent dense Circuit exponentials."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.hamiltonian_evolution.suzuki_trotter.comparisons.circuit import (
    reference_circuit,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"x": 0.37}
        ),
        reference_circuit(),
        ("qba",),
    )


if __name__ == "__main__":
    run_example()
