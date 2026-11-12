"""Pinned original source check outside the native user library."""

from pathlib import Path

from examples.synthesis.algorithms.search_and_optimization.qaoa.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    for params in BINDINGS:
        program = synth.synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"params": params}
        )
        compare_states(program, reference_circuit(params), ("v.a", "v.b"))


if __name__ == "__main__":
    run_example()
