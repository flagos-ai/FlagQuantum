"""Validate the exact original source outside the native example library."""

from pathlib import Path

from examples.synthesis.applications.optimization.set_cover.comparisons.circuit import (
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
        compare_states(
            program,
            reference_circuit(params),
            (
                "v.x",
                "v.independent_rule_1_slack_var",
                "v.independent_rule_2_slack_var",
                "v.independent_rule_3_slack_var",
                "v.independent_rule_4_slack_var",
                "v.independent_rule_5_slack_var",
                "v.independent_rule_6_slack_var",
                "v.independent_rule_7_slack_var",
                "v.independent_rule_8_slack_var",
                "v.independent_rule_9_slack_var",
                "v.independent_rule_10_slack_var",
            ),
        )


if __name__ == "__main__":
    run_example()
