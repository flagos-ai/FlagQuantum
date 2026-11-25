"""Verify the untouched symmetric-compression original with an independent Circuit."""

from pathlib import Path

from examples.synthesis.community.paper_implementation_project.quantum_compression_algorithm_for_symmetric_states.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    compare_states(
        synth.synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("x",),
    )


if __name__ == "__main__":
    run_example()
