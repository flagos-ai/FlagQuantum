"""Execute the unchanged source and compare its complete complex state."""

from pathlib import Path

from examples.synthesis.applications.benchmarking.quantum_volume.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("target",),
    )


if __name__ == "__main__":
    run_example()
