"""Test-only untouched source comparison against the independent matrix."""

from pathlib import Path

from examples.synthesis.community.paper_implementation_project.block_encoding.laplacian_4x4.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "flagquantum").is_dir()
    )
    source = (
        root
        / "tests/fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/block_encoding/laplacian_BE_Nx4_Ny4.qmod"
    )
    compare_states(
        synth.synthesize(source), reference_circuit(), ("s", "j", "data", "dlt")
    )


if __name__ == "__main__":
    run_example()
