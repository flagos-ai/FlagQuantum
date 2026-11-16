"""Test-only source identity and independent Circuit check."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.optimization.learning_optimization.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    for params in BINDINGS:
        program = synth.synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"params": params}
        )
        compare_states(
            program,
            reference_circuit(params),
            ("v.x[0]", "v.x[1]", "v.constraint_slack_var"),
        )


if __name__ == "__main__":
    run_example()
