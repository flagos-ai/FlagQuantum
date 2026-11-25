"""Frozen Grover source against independent oracle/reflection matrices."""

from pathlib import Path

from examples.synthesis.algorithms.search_and_optimization.grover.three_sat_grover.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    for r in (0, 1, 2):
        program = synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"r": r}
        )
        compare_states(program, reference_circuit(r), ("x",))


if __name__ == "__main__":
    run_example()
