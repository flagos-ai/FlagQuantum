"""Check the pinned original outside the native example library."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.finance_workshops.combi_workshop_equality_constriants_po.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    for params in BINDINGS:
        program = synth.synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"params": params},
            max_workspace=0,
        )
        compare_states(
            program,
            reference_circuit(params),
            ("w_array.a[0]", "w_array.a[1]", "w_array.a[2]"),
        )


if __name__ == "__main__":
    run_example()
