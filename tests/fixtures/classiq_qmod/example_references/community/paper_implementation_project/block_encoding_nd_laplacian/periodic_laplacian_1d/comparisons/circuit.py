"""Compare the frozen source with independently specified unitary matrices."""

from pathlib import Path

from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_1d.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    compare_states(
        synth.synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs=None
        ),
        reference_circuit(),
        ("j", "l"),
    )


if __name__ == "__main__":
    run_example()
