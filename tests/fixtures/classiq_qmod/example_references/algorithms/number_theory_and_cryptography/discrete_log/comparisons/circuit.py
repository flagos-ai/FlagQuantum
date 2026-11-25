"""Test-only untouched discrete logarithm source comparison."""

from pathlib import Path

from examples.synthesis.algorithms.number_theory_and_cryptography.discrete_log.comparisons.circuit import (
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
        / "tests/fixtures/classiq_qmod/corpus/original/algorithms/number_theory_and_cryptography/discrete_log/discrete_log.qmod"
    )
    compare_states(
        synth.synthesize(source), reference_circuit(), ("alpha", "beta", "func_res")
    )


if __name__ == "__main__":
    run_example()
