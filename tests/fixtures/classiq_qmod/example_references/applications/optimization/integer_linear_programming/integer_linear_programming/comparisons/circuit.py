"""Original integer/slack source validation belongs to test fixtures."""

from pathlib import Path

from examples.synthesis.applications.optimization.integer_linear_programming.comparisons.circuit import (
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
        compare_states(
            program,
            reference_circuit(params),
            (
                "v.x[0]",
                "v.x[1]",
                "v.x[2]",
                "v.monotone_rule_1_slack_var",
                "v.monotone_rule_2_slack_var",
            ),
        )


if __name__ == "__main__":
    run_example()
