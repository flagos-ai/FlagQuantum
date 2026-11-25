"""Check the pinned original outside the native example library."""

from pathlib import Path

from examples.synthesis.applications.finance.portfolio_optimization.comparisons.circuit import (
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
            ("v.w[0]", "v.w[1]", "v.w[2]", "v.budget_rule_slack_var"),
        )


if __name__ == "__main__":
    run_example()
