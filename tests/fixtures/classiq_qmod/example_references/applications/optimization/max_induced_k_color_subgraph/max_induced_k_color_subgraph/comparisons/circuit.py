"""Check the exact original only in test-side source adaptation."""

from pathlib import Path

from examples.synthesis.applications.optimization.max_induced_k_color_subgraph.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    for params in BINDINGS:
        program = synth.synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"params": params},
            max_workspace=0,
        )
        compare_states(program, reference_circuit(params), ("v.x[0]", "v.x[1]"))


if __name__ == "__main__":
    run_example()
