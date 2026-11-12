"""Check retained source bytes against an independent assignment-cost Circuit."""

from pathlib import Path

from examples.synthesis.applications.logistics.traveling_salesman_problem.comparisons.circuit import (
    BINDINGS,
    SOURCE_FIELDS,
    source_reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    for params in BINDINGS:
        program = synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"params": params}
        )
        compare_states(program, source_reference_circuit(params), SOURCE_FIELDS)


if __name__ == "__main__":
    run_example()
