"""All source portfolio scenarios, bounded executions and independent states."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.workshops.finance_workshops.combi_workshop_equality_constriants_po.main import (
    build_program as frozen_equality,
)
from examples.synthesis.tutorials.workshops.finance_workshops.combi_workshop_inequality_constriants_po.main import (
    build_program as frozen_inequality,
)
from examples.synthesis.tutorials.workshops.finance_workshops.portfolio_programs import (
    holdings_program,
    slack_program,
)
from examples.synthesis.tutorials.workshops.finance_workshops.portfolio_workflow import (
    EQUALITY_SCENARIOS,
    INEQUALITY_SCENARIOS,
    Scenario,
    assignments,
    bindings,
    build_program,
    classical_optima,
    cost,
    distribution,
    mean_cost,
    run_workflow,
    sample,
    verify,
)
from flagquantum import synth

pytestmark = pytest.mark.integration
SCENARIOS = (*EQUALITY_SCENARIOS, *INEQUALITY_SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_all_source_costs_against_independent_expanded_covariance(
    scenario: Scenario,
) -> None:
    domain = assignments(scenario)
    assert len(domain) == (512 if scenario.constraint == "inequality" else 64)
    for assignment in domain:
        a, b, c = assignment[:3]
        slack = assignment[3] if len(assignment) == 4 else 0
        expected = (
            0.9 * (a * a + b * b + c * c)
            + a * b
            - 1.4 * a * c
            - 0.4 * b * c
            - scenario.return_scale * (3 * a + 4 * b - c)
            + scenario.penalty * (a + b + c + slack - 6) ** 2
        )
        assert cost(scenario, assignment) == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
@pytest.mark.parametrize("binding", range(3))
def test_original_schedule_and_nonzero_signed_absolute_states(
    scenario: Scenario, binding: int
) -> None:
    check = verify(scenario, bindings(scenario)[binding])
    assert check.maximum_state_error < 3e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.resources.n_qubits == scenario.width


@pytest.mark.parametrize("scenario", (EQUALITY_SCENARIOS[-1], INEQUALITY_SCENARIOS[-1]))
def test_final_source_scenario_agrees_with_frozen_quantum_parent(
    scenario: Scenario,
) -> None:
    parameters = bindings(scenario)[1]
    program = build_program(scenario, parameters)
    frozen = (
        frozen_inequality if scenario.constraint == "inequality" else frozen_equality
    )(parameters)
    actual = fq.run(program).statevector()
    expected = fq.run(frozen).statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    order = tuple(w for name in scenario.registers for w in program.registers[name])
    old_order = tuple(w for name in scenario.registers for w in frozen.registers[name])
    torch.testing.assert_close(
        actual.reshape((2,) * scenario.width).permute(order),
        expected.reshape((2,) * scenario.width).permute(old_order),
        atol=3e-11,
        rtol=0,
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_callback_cap_prevents_extra_execution_and_selects_real_evaluation(
    scenario: Scenario,
) -> None:
    bounded = replace(scenario, evaluation_budget=3)
    with patch(
        "examples.synthesis.tutorials.workshops.finance_workshops.portfolio_workflow.fq.run",
        wraps=fq.run,
    ) as run:
        workflow = run_workflow(bounded, shots=193, seed=1940)
    assert run.call_count == 4  # Three actual objectives and one final readout.
    assert len(workflow.trace) == 3
    assert workflow.termination == "evaluation_budget" and not workflow.converged
    assert workflow.selected == min(workflow.trace, key=lambda row: row.cost)
    assert mean_cost(bounded, distribution(bounded, workflow.program)) == pytest.approx(
        workflow.selected.cost, abs=1e-10
    )
    assert sum(row.count for row in workflow.samples) == 193
    assert list(workflow.samples) == sorted(
        workflow.samples, key=lambda row: (row.cost, row.assignment)
    )
    assert workflow.best_observed == workflow.samples[0]
    check = verify(bounded, workflow.selected.parameters)
    assert check.maximum_state_error < 3e-11


def test_unconstrained_optimum_may_exceed_the_declared_budget() -> None:
    first, second, final = EQUALITY_SCENARIOS
    assert (
        first.layers,
        first.return_scale,
        first.penalty,
        first.evaluation_budget,
    ) == (4, 1.5, 0, 30)
    assert (
        second.layers,
        second.return_scale,
        second.penalty,
        second.evaluation_budget,
    ) == (3, 1.7, 0, 10)
    assert (
        final.layers,
        final.return_scale,
        final.penalty,
        final.evaluation_budget,
    ) == (3, 10, 2, 40)
    assert classical_optima(first)[0][0] == (3, 2, 2)
    assert classical_optima(first)[0][1] == pytest.approx(-11.2)
    assert {a for a, _ in classical_optima(second)} == {(2, 3, 1), (3, 3, 2)}
    assert classical_optima(final)[0][0] == (3, 3, 0)
    rows = sample(first, {(3, 2, 2): 1.0}, shots=17, seed=0)
    assert rows[0].holdings == (3, 2, 2) and not rows[0].budget_feasible
    assert rows[0].zero_penalty  # No penalty exists in this source scenario.
    assert rows[0].count == 17  # The unfiltered unconstrained answer is retained.


def test_joint_numeric_slack_cost_and_distinct_feasibility_diagnostics() -> None:
    scenario = INEQUALITY_SCENARIOS[0]
    assert cost(scenario, (1, 2, 1, 2)) != cost(scenario, (1, 2, 1, 0))
    rows = sample(
        scenario, {(1, 2, 1, 2): 0.5, (1, 2, 1, 0): 0.5}, shots=193, seed=1941
    )
    assert len(rows) == 2 and all(row.budget_feasible for row in rows)
    assert {row.zero_penalty for row in rows} == {True, False}
    assert {row.slack for row in rows} == {0, 2}
    assert mean_cost(scenario, {(1, 2, 1, 2): 0.5, (1, 2, 1, 0): 0.5}) == pytest.approx(
        sum(row.cost for row in rows) / 2
    )
    for profile, optimum, value in (
        (scenario, (2, 2, 1, 1), -9.7),
        (INEQUALITY_SCENARIOS[-1], (3, 3, 0, 0), -184.8),
    ):
        assert classical_optima(profile)[0][0] == optimum
        assert classical_optima(profile)[0][1] == pytest.approx(value)


@pytest.mark.parametrize("scenario", (EQUALITY_SCENARIOS[0], INEQUALITY_SCENARIOS[0]))
@pytest.mark.parametrize("limit", ("qubits", "operations", "memory"))
def test_resource_budget_refusals(scenario: Scenario, limit: str) -> None:
    with pytest.raises(ValueError, match="budget|limit|memory|candidate"):
        synth.synthesize(
            slack_program if scenario.constraint == "inequality" else holdings_program,
            classical_inputs={
                "params": bindings(scenario)[1],
                "layers": scenario.layers,
                "return_scale": scenario.return_scale,
                "penalty": scenario.penalty,
            },
            max_qubits=scenario.width - 1 if limit == "qubits" else scenario.width,
            max_operations=1 if limit == "operations" else 2048,
            max_workspace=0,
            memory_limit_bytes=1 if limit == "memory" else 4 * 1024**2,
        )


@pytest.mark.parametrize("parameters", ((0.1,), (True,) * 8, (float("nan"),) * 8))
def test_invalid_classical_angles(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="finite radian parameters"):
        build_program(EQUALITY_SCENARIOS[0], parameters)


@pytest.mark.parametrize(
    "probabilities",
    ({(0, 0, 0): 0.5}, {(0, 0, 0): float("nan")}, {(0, 0, 0): -1.0, (1, 0, 0): 2.0}),
)
def test_invalid_probabilities(probabilities: dict[tuple[int, ...], float]) -> None:
    with pytest.raises(ValueError, match="normalized finite"):
        mean_cost(EQUALITY_SCENARIOS[0], probabilities)


@pytest.mark.parametrize("assignment", ((0, 0), (4, 0, 0), (True, 0, 0), (0, 0, 0, 8)))
def test_invalid_named_assignments(assignment: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="assignment must match"):
        cost(EQUALITY_SCENARIOS[0], assignment)


@pytest.mark.parametrize("budget", (0, True, 201))
def test_invalid_evaluation_budget(budget: int) -> None:
    with pytest.raises(ValueError, match="evaluation budget"):
        replace(EQUALITY_SCENARIOS[0], evaluation_budget=budget)


@pytest.mark.parametrize(
    "directory,count",
    (
        ("combi_workshop_equality_constriants_po", 18),
        ("combi_workshop_inequality_constriants_po", 16),
    ),
)
def test_all_source_cells_mapped_and_quantum_parents_unchanged(
    directory: str, count: int
) -> None:
    root = Path(__file__).resolve().parents[2]
    overview = (
        root
        / "examples/synthesis/tutorials/workshops/finance_workshops"
        / directory
        / "overview"
    )
    proof = json.loads((overview / "provenance.json").read_text())
    assert len(proof["cell_mapping"]) == len(proof["reviewed_code_cells"]) == count
    assert (
        hashlib.sha256((overview.parent / "main.py").read_bytes()).hexdigest()
        == proof["frozen_quantum_entry_sha256"]
    )
    notebook = json.loads((overview / "walkthrough.ipynb").read_text())
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "classiq" not in code and ".qmod" not in code and "source_format" not in code
