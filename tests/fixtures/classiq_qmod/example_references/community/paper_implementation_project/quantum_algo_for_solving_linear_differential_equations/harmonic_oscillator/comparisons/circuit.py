"""Test-only untouched harmonic oscillator comparison."""

from pathlib import Path

from examples.synthesis.community.paper_implementation_project.quantum_algo_for_solving_linear_differential_equations.harmonic_oscillator.comparisons.circuit import (
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
        / "tests/fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator.qmod"
    )
    compare_states(synth.synthesize(source), reference_circuit(), ("x", "ancilla", "y"))


if __name__ == "__main__":
    run_example()
