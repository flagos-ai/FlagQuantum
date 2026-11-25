"""Test-only untouched Toeplitz source comparison."""

from pathlib import Path

from examples.synthesis.community.paper_implementation_project.block_encoding.toeplitz.comparisons.circuit import (
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
        / "tests/fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/block_encoding/toeplitz_BE_N16.qmod"
    )
    compare_states(
        synth.synthesize(source), reference_circuit(), ("data", "s", "j", "dlt")
    )


if __name__ == "__main__":
    run_example()
