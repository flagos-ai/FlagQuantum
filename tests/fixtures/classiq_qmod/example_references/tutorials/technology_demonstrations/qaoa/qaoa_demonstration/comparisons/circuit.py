"""Keep original-source verification outside the native example library."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.qaoa.qaoa_demonstration.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    for params in BINDINGS:
        program = synth.synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"params_list": params},
            max_workspace=0,
        )
        compare_states(program, reference_circuit(params), ("target",))


if __name__ == "__main__":
    run_example()
