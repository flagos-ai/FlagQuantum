"""Frozen closed-callback program against an independent Circuit."""

from pathlib import Path

from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    for params in BINDINGS:
        program = synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"params": params}
        )
        compare_states(program, reference_circuit(params), ("v",))


if __name__ == "__main__":
    run_example()
