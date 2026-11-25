"""Source measurement signs, literal count geometry and true query budgets."""

import json
import math
from pathlib import Path

import pytest

from examples.synthesis.applications.optimization.robust_posture_optimization.main import (
    main,
)
from examples.synthesis.applications.optimization.robust_posture_optimization.overview.measurements import (
    SHOTS,
    TARGET,
    Measurement,
    estimate,
    sample,
)
from examples.synthesis.applications.optimization.robust_posture_optimization.overview.quantum import (
    Basis,
    build_program,
    distribution,
)
from examples.synthesis.applications.optimization.robust_posture_optimization.overview.reference import (
    reference_circuit,
)
from examples.synthesis.applications.optimization.robust_posture_optimization.overview.workflow import (
    INITIAL_PARAMETERS,
    run_workflow,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

pytestmark = pytest.mark.integration
BINDINGS = (
    (0.17, -0.23, 0.41, -0.59, 0.67, -0.83),
    (-0.31, 0.47, -0.61, 0.29, -0.73, 0.89),
)


@pytest.mark.parametrize("parameters", BINDINGS)
@pytest.mark.parametrize("basis", ("X", "Y", "Z"))
def test_all_three_source_bases_match_absolute_independent_states(
    parameters: tuple[float, ...], basis: Basis
) -> None:
    program = build_program(parameters, basis)
    compare_states(program, reference_circuit(parameters, basis), ("q",))
    assert program.report["workspace_qubits"] == 0


def test_literal_source_y_basis_has_s_then_h_sign() -> None:
    probabilities = distribution((math.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0), "Y")
    assert probabilities == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-14)


@pytest.mark.parametrize(
    "word,expected",
    ((0, (1.0, 1.0)), (1, (1.0, -1.0)), (2, (-1.0, 1.0)), (3, (-1.0, -1.0))),
)
def test_source_msb_first_count_marginals(
    word: int, expected: tuple[float, float]
) -> None:
    counts = tuple(SHOTS if index == word else 0 for index in range(4))
    assert Measurement("Z", counts).marginals == expected


def test_named_array_low_bit_and_source_first_count_column_are_distinct() -> None:
    probabilities = distribution((0.0, 0.0, 0.0, math.pi, 0.0, 0.0), "Z")
    assert probabilities == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-14)
    assert sample(probabilities, "Z", 7).marginals == (-1.0, 1.0)


def test_actual_samples_are_seeded_integer_counts_and_not_exact_means() -> None:
    probabilities = (0.1, 0.2, 0.3, 0.4)
    value = sample(probabilities, "X", 7)
    assert value == sample(probabilities, "X", 7)
    assert sum(value.counts) == SHOTS
    assert all(type(count) is int and count >= 0 for count in value.counts)
    assert value.marginals != pytest.approx((-0.4, -0.2), abs=1e-14)
    reading = estimate(BINDINGS[0], 7)
    expected = tuple(sum(measurement.marginals) for measurement in reading.measurements)
    assert reading.position == expected
    assert reading.cost == pytest.approx(
        sum((TARGET[i] - expected[i]) ** 2 for i in range(3))
    )


@pytest.mark.parametrize(
    "parameters", ((0.1,), (True,) * 6, (math.nan,) * 6, (math.inf,) * 6)
)
def test_bad_classical_angles_refuse(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="six finite"):
        build_program(parameters, "X")


def test_original_two_qubit_budget_refuses_one_qubit() -> None:
    with pytest.raises(ValueError, match="qubit|width|budget"):
        synth.synthesize(
            main, classical_inputs={"theta": BINDINGS[0]}, max_qubits=1, max_workspace=0
        )


@pytest.mark.parametrize("budget", (0, 31, True))
def test_invalid_source_query_budget_refuses(budget: int) -> None:
    with pytest.raises(ValueError, match="evaluation budget"):
        run_workflow(evaluation_budget=budget)


def test_real_native_optimizer_stops_before_exceeding_objective_query_cap() -> None:
    workflow = run_workflow(evaluation_budget=2, seed=7)
    assert len(workflow.trace) == 2
    assert workflow.trace[0].parameters == pytest.approx(INITIAL_PARAMETERS)
    assert workflow.termination == "evaluation_budget" and not workflow.converged
    assert workflow.selected in workflow.trace
    assert workflow.selected.estimate.cost == min(
        row.estimate.cost for row in workflow.trace
    )
    assert (
        sum(
            sum(reading.counts)
            for row in workflow.trace
            for reading in row.estimate.measurements
        )
        == 6000
    )
    assert sum(workflow.final_y.counts) == 1000
    assert all(
        sum(sum(m.counts) for m in value.measurements) == 3000
        for value in workflow.positions
    )
    assert all(math.isfinite(row.estimate.cost) for row in workflow.trace)


def test_all_fourteen_source_stages_are_mapped_and_unexecuted() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/optimization/robust_posture_optimization/overview"
    )
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        2,
        4,
        5,
        6,
        8,
        9,
        13,
        14,
        15,
        16,
        17,
        18,
        20,
        21,
    ]
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
