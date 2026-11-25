"""Source-size QCE models retain phase, scoring, actual query and readout semantics."""

import json
from pathlib import Path

import pytest

from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii.overview.model import (
    BINDINGS,
    INITIAL,
    Model,
    Parameters,
    feasible_value,
    optimization_cost,
    phase_cost,
)
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii.overview.presentation import (
    sample_table,
)
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii.overview.verification import (
    verify_program,
)
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii.overview.workflow import (
    Observation,
    build_program,
    optimize,
    parameters,
    sample,
    statistics,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "model,width,operations",
    (("maxcut", 3, 24), ("penalty", 9, 163), ("feasible", 5, 856)),
)
@pytest.mark.parametrize("angles", BINDINGS)
def test_independent_absolute_states_and_original_widths(
    model: Model, width: int, operations: int, angles: Parameters
) -> None:
    program = build_program(model, angles)
    check = verify_program(model, angles, program)
    assert check.maximum_state_error < 2e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.resources.n_qubits == width
    assert check.resources.n_operations == operations


def test_quantum_and_classical_penalties_remain_distinct() -> None:
    assert phase_cost("penalty", (7, 3, 0)) == -36 + 2 * 11**2
    assert optimization_cost("penalty", (7, 3, 0)) == -36 + 4 * 11
    assert optimization_cost("feasible", (7, 3)) == optimization_cost(
        "penalty", (7, 3, 15)
    )
    assert phase_cost("feasible", (3, 2)) == -19
    assert phase_cost("feasible", (3, 3)) == 0
    assert feasible_value(3, 2) == 19
    assert feasible_value(3, 3) == 0
    assert feasible_value(6, 0) == 18
    assert feasible_value(5, 1) == 0
    assert phase_cost("maxcut", (1,)) == -2
    assert phase_cost("maxcut", (6,)) == -2


@pytest.mark.parametrize(
    "model,budget,shots",
    (("maxcut", 40, 1000), ("penalty", 50, 10), ("feasible", 50, 10)),
)
def test_actual_objective_caps_last_query_and_sampling(
    model: Model, budget: int, shots: int
) -> None:
    run = optimize(model)
    assert len(run.evaluations) == budget
    assert run.evaluations[0].parameters == INITIAL
    assert run.selected == run.evaluations[-1]
    assert not run.converged and run.termination == "evaluation_budget"
    assert sum(row.count for row in run.statistics_samples) == 1000
    assert sum(row.count for row in run.table_samples) == shots
    table = sample_table(run)
    assert sum(row["count"] for row in table) == shots
    if model != "maxcut":
        assert all(
            row["feasible_value"] == feasible_value(int(row["a"]), int(row["b"]))
            for row in table
        )
        result = statistics(run.statistics_samples)
        assert result.actual_shots == 1000
        assert result.overall_score == pytest.approx(
            result.feasibility_probability * result.average_feasible_value
        )


def test_readout_statistics_ignore_slack_and_handle_no_feasible_shots() -> None:
    rows = (Observation((3, 2, 15), 3, 0.3), Observation((3, 3, 0), 7, 0.7))
    result = statistics(rows)
    assert result.feasibility_probability == 0.3
    assert result.average_feasible_value == 19
    assert result.overall_score == 5.7
    assert statistics((Observation((7, 3), 10, 1.0),)).average_feasible_value == 0


def test_actual_draws_reproducible_and_not_injected_optima() -> None:
    probabilities: dict[tuple[int, ...], float] = {(0, 0): 0.5, (7, 3): 0.5}
    first = sample(probabilities, shots=1000, seed=19)
    assert first == sample(probabilities, shots=1000, seed=19)
    assert first != sample(probabilities, shots=1000, seed=20)
    assert sum(row.count for row in first) == 1000
    assert all(row.probability == row.count / 1000 for row in first)


@pytest.mark.parametrize(
    "shots,seed", ((0, 0), (True, 0), (10, -1), (10, True), (10, 2**63))
)
def test_sampling_limits_refused(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample({(0, 0): 1.0}, shots=shots, seed=seed)


@pytest.mark.parametrize("angles", ((0.1,), (float("nan"),) * 6, (float("inf"),) * 6))
def test_parameters_need_six_finite_angles(angles: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="six finite"):
        parameters(angles)


def test_all_source_code_stages_and_solution_replays_mapped() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/workshops/algo_design_qce_tutorial/algo_design_qce_tutorial_part_ii/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        3,
        5,
        10,
        14,
        17,
        21,
        25,
        27,
        29,
    ]
    for row in proof["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"
    text = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert 'penalty_solution = optimize("penalty"' in text
    assert 'feasible_solution = optimize("feasible"' in text
    assert "classiq import" not in text
