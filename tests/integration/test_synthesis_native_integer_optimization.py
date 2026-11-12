"""Explicit native integer/slack ports preserve the complete retained program."""

from pathlib import Path
from typing import Literal

import pytest

from examples.synthesis.applications.optimization.integer_linear_programming.comparisons.circuit import (
    BINDINGS,
    cost,
    reference_circuit,
)
from examples.synthesis.applications.optimization.integer_linear_programming.main import (
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import estimate_resources

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/applications/optimization/integer_linear_programming/integer_linear_programming.qmod"
)


@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_integer_optimization_preserves_fields_and_all_amplitudes(
    dataset: int, objective: Literal["width", "depth"]
) -> None:
    params = BINDINGS[dataset]
    native = synth.synthesize(
        main, classical_inputs={"params": params}, objective=objective, max_workspace=0
    )
    original = synth.synthesize(
        SOURCE, classical_inputs={"params": params}, objective=objective
    )
    reference = reference_circuit(params)
    compare_states(native, reference, ("x0", "x1", "x2", "s0", "s1"))
    compare_states(
        original,
        reference,
        (
            "v.x[0]",
            "v.x[1]",
            "v.x[2]",
            "v.monotone_rule_1_slack_var",
            "v.monotone_rule_2_slack_var",
        ),
    )
    for program in (native, original):
        assert program.report["workspace_qubits"] == 0
        assert estimate_resources(program.circuit).n_qubits == 8


def test_integer_cost_retains_two_distinct_slack_bits_and_absolute_offset() -> None:
    assert cost((0,) * 8) == -20
    assert cost((1, 0, 0, 0, 0, 0, 0, 0)) == -10
    assert cost((0, 1, 0, 0, 0, 0, 0, 0)) == -10
    assert cost((1, 1, 0, 0, 0, 0, 0, 0)) == 0
