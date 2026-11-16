"""Verify the unchanged original hidden-string program against an independent Circuit."""

from pathlib import Path

from examples.synthesis.algorithms.foundational.bernstein_vazirani.comparisons.circuit import (
    circuit_reference,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    program = synthesize(Path(__file__).with_name("source.qmod"))
    compare_states(program, circuit_reference(13), ("x",))


if __name__ == "__main__":
    run_example()
