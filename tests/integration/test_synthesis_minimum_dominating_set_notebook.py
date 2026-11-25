"""Verify the twenty-bit minimum dominating set Notebook without hosted services."""

import itertools
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.minimum_dominating_set.comparisons.circuit import (
    BINDINGS,
    classical_cost,
    reference_circuit,
)
from examples.synthesis.applications.optimization.minimum_dominating_set.main import (
    main,
)
from examples.synthesis.applications.optimization.minimum_dominating_set.overview import (
    model,
    workflow,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.graph_optimization import lower_tail_cost
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def instance() -> model.Model:
    return model.fresh_model()


@pytest.fixture(scope="module")
def table(instance: model.Model) -> torch.Tensor:
    return model.energies(instance)


def test_fresh_graph_and_all_decisions(
    instance: model.Model, table: torch.Tensor
) -> None:
    assert sorted(instance.graph.edges) == [
        (0, 1),
        (0, 3),
        (0, 5),
        (1, 2),
        (1, 4),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (4, 5),
    ]
    assert tuple(map(len, instance.slack_weights)) == (2, 2, 3, 2, 3, 2)
    assert table.numel() == 1048576
    minima = table.reshape(-1, 64).min(dim=0).values
    feasible = 0
    for word in range(64):
        selected = {i for i in range(6) if word & (1 << i)}
        dominated = selected.union(
            *(set(instance.graph.neighbors(i)) for i in selected)
        )
        expected = len(dominated) == 6
        assert instance.dominates(word) == expected
        assert int(minima[word]) == len(selected) + (
            0 if expected else 10 * (6 - len(dominated))
        )
        feasible += int(expected)
    assert feasible == 53
    assert model.exact_solutions(instance) == (3, 5, 6, 9, 12, 17, 18, 24, 33, 36, 48)


def test_energy_word_bit_order(table: torch.Tensor) -> None:
    masks = (0, (1 << 20) - 1, *[1 << wire for wire in range(20)])
    combinations = itertools.combinations(range(20), 2)
    for word in (*masks, *((1 << a) | (1 << b) for a, b in combinations)):
        bits = tuple((word >> wire) & 1 for wire in range(20))
        assert int(table[word]) == -classical_cost(bits)


@pytest.mark.parametrize("parameters", BINDINGS)
def test_independent_complete_state(parameters: tuple[float, ...]) -> None:
    program = workflow.build_program(parameters)
    compare_states(program, reference_circuit(parameters), model.REGISTER_NAMES)
    assert program.report["workspace_qubits"] == 0


def test_actual_distribution_bit_order() -> None:
    program = workflow.build_program(BINDINGS[0])
    probabilities = workflow.distribution(program).reshape(-1, 64).sum(dim=0)
    actual = program.probabilities(fq.run(program), registers=("x",), threshold=0)
    expected = torch.tensor(
        [actual.get((word,), 0.0) for word in range(64)], dtype=torch.float64
    )
    torch.testing.assert_close(probabilities, expected, rtol=0, atol=1e-12)


def test_cvar_fractional_ties() -> None:
    masses = (0.1, 0.25, 0.25, 0.4)
    costs = (0, 7, 7, 23)
    value = workflow.lower_cvar(
        torch.tensor(masses, dtype=torch.float64), torch.tensor(costs)
    )
    assert value == pytest.approx(lower_tail_cost(masses, costs, 0.7), abs=1e-14)


def test_real_finite_samples(instance: model.Model, table: torch.Tensor) -> None:
    probabilities = workflow.distribution(workflow.build_program(BINDINGS[0]))
    rows = workflow.sample(probabilities, table, instance, seed=2718)
    assert sum(row.count for row in rows) == 1000
    for row in rows:
        assert row.count > 0 and row.decisions == row.word % 64
        assert row.feasible == instance.dominates(row.decisions)
        assert row.cost == -classical_cost(
            tuple((row.word >> i) & 1 for i in range(20))
        )
    assert rows == tuple(sorted(rows, key=lambda row: (row.cost, row.word)))


@pytest.mark.parametrize("params", [(0.0,) * 11, (0.0,) * 11 + (float("nan"),)])
def test_invalid_angles(params: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="twelve finite"):
        workflow.build_program(params)


def test_source_width_budget_is_checked() -> None:
    with pytest.raises(ValueError, match=r"synthesis\.constraints\.max_qubits"):
        synth.synthesize(
            main,
            classical_inputs={"params": BINDINGS[0]},
            max_qubits=19,
            max_workspace=0,
        )


def test_notebook_all_stages_without_embedded_results() -> None:
    root = Path(model.__file__).parent
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 15
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
    assert [cell["index"] for cell in provenance["reviewed_code_cells"]] == [
        3,
        5,
        7,
        8,
        10,
        12,
        14,
        16,
        19,
        21,
        23,
        25,
        26,
        28,
        29,
    ]
