"""Check the untouched packed ND Laplacian against an independent permutation."""

from pathlib import Path

from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_nd.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

if __name__ == "__main__":
    compare_states(
        synth.synthesize(Path(__file__).with_name("source.qmod")),
        reference_circuit(),
        ("j", "l", "k"),
    )
