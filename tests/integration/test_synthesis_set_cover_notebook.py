"""Validate the original 23-bit Set Cover workflow and independent amplitudes."""

import itertools
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.set_cover.comparisons.circuit import (
    BINDINGS,
    classical_cost,
    reference_circuit,
)
from examples.synthesis.applications.optimization.set_cover.main import main
from examples.synthesis.applications.optimization.set_cover.overview import (
    model,
    workflow,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.graph_optimization import lower_tail_cost
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def instance() -> model.Model:
    return model.source_model()


@pytest.fixture(scope="module")
def table(instance: model.Model) -> torch.Tensor:
    return model.costs(instance)


def test_full_joint_cost_table_and_exact_minima(
    instance: model.Model, table: torch.Tensor
) -> None:
    assert len(instance.sets) == 8 and len(instance.coverers) == 10
    assert sum(instance.widths) == 23
    assert table.shape == (1 << 23,) and table.dtype == torch.int16
    assert (int(table.min()), int(table.max())) == (4, 650)
    minima = table.reshape(256, -1).min(dim=1).values.tolist()
    all_elements = set(range(1, 11))
    feasible = set()
    for word, energy in enumerate(minima):
        covered = set().union(
            *(subset for i, subset in enumerate(instance.sets) if word & (1 << i))
        )
        assert energy == word.bit_count() + 10 * len(all_elements - covered)
        assert instance.covers(word) == (covered == all_elements)
        if covered == all_elements:
            feasible.add(word)
    assert len(feasible) == 63
    assert model.exact_solutions(instance) == (15, 30, 58, 90, 114, 141, 240)


def test_energy_field_bit_order(instance: model.Model, table: torch.Tensor) -> None:
    inputs = (0, (1 << 23) - 1, *[1 << bit for bit in range(23)])
    pairs = ((1 << a) | (1 << b) for a, b in itertools.combinations(range(23), 2))
    for word in (*inputs, *pairs):
        fields: list[int] = []
        offset = 23
        for width in instance.widths:
            offset -= width
            fields.extend((word >> (offset + bit)) & 1 for bit in range(width))
        assert int(table[word]) == -classical_cost(tuple(fields))


@pytest.mark.parametrize("parameters", BINDINGS)
def test_independent_full_state(parameters: tuple[float, ...]) -> None:
    program = workflow.build_program(parameters)
    compare_states(program, reference_circuit(parameters), model.NAMES)
    assert program.report["workspace_qubits"] == 0


def test_named_decision_bit_order_and_actual_sampling(
    instance: model.Model, table: torch.Tensor
) -> None:
    program = workflow.build_program(BINDINGS[0])
    probabilities = workflow.distribution(program)
    readout = program.probabilities(
        fq.run(program), registers=("x",), max_amplitudes=1 << 23, threshold=0
    )
    expected = torch.tensor(
        [readout.get((i,), 0.0) for i in range(256)], dtype=torch.float64
    )
    torch.testing.assert_close(
        probabilities.reshape(256, -1).sum(1), expected, rtol=0, atol=1e-12
    )
    rows = workflow.sample(probabilities, table, instance, seed=2718)
    assert sum(row.count for row in rows) == 1000
    assert rows == tuple(sorted(rows, key=lambda row: (row.cost, row.word)))
    for row in rows:
        assert row.cost == int(table[row.word])
        assert row.decisions == row.word >> 15
        assert row.feasible == instance.covers(row.decisions)


def test_fractional_cost_bin() -> None:
    masses = (0.1, 0.25, 0.25, 0.4)
    costs = (0, 7, 7, 23)
    actual = workflow.lower_cvar(
        torch.tensor(masses, dtype=torch.float64), torch.tensor(costs)
    )
    assert actual == pytest.approx(lower_tail_cost(masses, costs, 0.7), abs=1e-14)


def test_width_budget_rejects_original_encoding() -> None:
    with pytest.raises(ValueError, match=r"synthesis\.constraints\.max_qubits"):
        synth.synthesize(
            main,
            classical_inputs={"params": BINDINGS[0]},
            max_qubits=22,
            max_workspace=0,
        )


def test_notebook_keeps_every_source_stage() -> None:
    root = Path(model.__file__).parent
    n = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in n["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 14 and all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        3,
        5,
        6,
        8,
        10,
        12,
        14,
        17,
        19,
        21,
        23,
        24,
        27,
        28,
    ]
