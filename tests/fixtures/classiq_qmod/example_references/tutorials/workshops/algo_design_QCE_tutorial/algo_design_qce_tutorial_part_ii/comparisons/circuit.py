"""Compare the exact original tutorial with the independent numeric Circuit."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii.comparisons.circuit import (
    BINDINGS,
    SOURCE_FIELDS,
    reference_circuit,
)
from flagquantum.synth import synthesize


def run_example() -> None:
    for params in BINDINGS:
        program = synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"params": params}
        )
        compare_states(program, reference_circuit(params), SOURCE_FIELDS)


if __name__ == "__main__":
    run_example()
