"""Validate the exact original source outside the native example library."""

from pathlib import Path

from examples.synthesis.applications.optimization.minimum_dominating_set.comparisons.circuit import (
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
                "v.dominating_rule_0_slack_var",
                "v.dominating_rule_1_slack_var",
                "v.dominating_rule_2_slack_var",
                "v.dominating_rule_3_slack_var",
                "v.dominating_rule_4_slack_var",
                "v.dominating_rule_5_slack_var",
            ),
        )


if __name__ == "__main__":
    run_example()
