"""Check the unchanged source against an independent Circuit."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.comparator.comparator_integer_example.comparisons.circuit import (
    reference_circuit,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    program = synthesize(Path(__file__).with_name("source.qmod"))
    compare_states(program, reference_circuit(), ("a", "res"))


if __name__ == "__main__":
    run_example()
