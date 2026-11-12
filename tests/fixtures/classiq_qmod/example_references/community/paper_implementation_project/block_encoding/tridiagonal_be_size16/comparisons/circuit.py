"""Test-only original/approved source acceptance; native users run Python."""

from pathlib import Path

from examples.synthesis.community.paper_implementation_project.block_encoding.tridiagonal.comparisons.circuit import (
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
        / "tests/fixtures/classiq_qmod/corpus/approved/community/paper_implementation_project/block_encoding/tridiagonal_BE_size16.qmod"
    )
    compare_states(synth.synthesize(source), reference_circuit(), ("s", "j", "data"))


if __name__ == "__main__":
    run_example()
