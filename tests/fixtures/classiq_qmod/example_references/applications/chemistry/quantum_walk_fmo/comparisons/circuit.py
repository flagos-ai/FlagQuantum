"""Compare the retained source with the independent native-library Circuit."""

from pathlib import Path

from examples.synthesis.applications.chemistry.quantum_walk_fmo.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"time_point": 0.037},
        ),
        reference_circuit(),
        ("qn",),
    )


if __name__ == "__main__":
    run_example()
