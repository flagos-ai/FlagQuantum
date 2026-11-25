"""Exact eighteen-bit electric-grid factors retain full source cost and layers."""

from pathlib import Path
from typing import Literal

import pytest

from examples.synthesis.applications.optimization.electric_grid_optimization import main
from examples.synthesis.applications.optimization.electric_grid_optimization.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import estimate_resources

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/applications/optimization/electric_grid_optimization/electric_grid_optimization.qmod"
)


@pytest.mark.parametrize("dataset", [0, 1])
def test_complete_grid_program_matches_raw_source_and_factored_circuit(
    dataset: int,
) -> None:
    params = BINDINGS[dataset]
    native = synth.synthesize(
        main.main, classical_inputs={"params": params}, max_workspace=0
    )
    original = synth.synthesize(SOURCE, classical_inputs={"params": params})
    reference = reference_circuit(params)
    for program in (native, original):
        registers = (
            ("a", "b", "c", "sa", "sb", "sc")
            if program is native
            else (
                "v.x[0]",
                "v.x[1]",
                "v.x[2]",
                "v.source_supply_rule_0_slack_var",
                "v.source_supply_rule_1_slack_var",
                "v.source_supply_rule_2_slack_var",
            )
        )
        compare_states(program, reference, registers)
        assert program.report["workspace_qubits"] == 0
        assert estimate_resources(program.circuit).n_qubits == 18


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_both_objectives_retain_identical_complete_cost_and_mixer_layers(
    objective: Literal["width", "depth"],
) -> None:
    params = BINDINGS[1]
    native = synth.synthesize(
        main.main,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    original = synth.synthesize(
        SOURCE, classical_inputs={"params": params}, objective=objective
    )
    assert (
        native.circuit.instructions == main.build_program(params).circuit.instructions
    )
    assert (
        original.circuit.instructions
        == synth.synthesize(
            SOURCE, classical_inputs={"params": params}
        ).circuit.instructions
    )
