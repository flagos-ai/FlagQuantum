"""Original electric-grid source remains test evidence only."""

from pathlib import Path

from examples.synthesis.applications.optimization.electric_grid_optimization.comparisons.circuit import (
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
                "v.source_supply_rule_0_slack_var",
                "v.source_supply_rule_1_slack_var",
                "v.source_supply_rule_2_slack_var",
            ),
        )


if __name__ == "__main__":
    run_example()
