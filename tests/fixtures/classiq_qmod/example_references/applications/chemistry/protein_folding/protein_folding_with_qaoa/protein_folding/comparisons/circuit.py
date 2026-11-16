"""Frozen folding program against the independent direction-energy Circuit."""

from pathlib import Path

from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_qaoa.protein_folding.comparisons.circuit import (
    BINDINGS,
    SOURCE_FIELDS,
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    for params in BINDINGS:
        program = synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"params": params}
        )
        compare_states(program, reference_circuit(params), SOURCE_FIELDS)


if __name__ == "__main__":
    run_example()
