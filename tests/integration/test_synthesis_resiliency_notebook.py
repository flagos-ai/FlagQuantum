"""Check independent original-size resiliency costs, states and sample contracts."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.telecom.resiliency_planning.main import (
    build_program,
)
from examples.synthesis.applications.telecom.resiliency_planning.overview import (
    problem,
    reference,
    workflow,
)

pytestmark = pytest.mark.integration


def test_independent_quadratic_costs_and_validity() -> None:
    words = [0, (1 << 24) - 1] + [1 << i for i in range(24)]
    words += [(1 << i) | (1 << j) for i in range(24) for j in range(i)]
    for support, _ in reference.factors():
        words.extend(
            sum(((row >> position) & 1) << bit for position, bit in enumerate(support))
            for row in range(1 << len(support))
        )
    rows = torch.tensor(words, dtype=torch.int64)
    expected = torch.tensor(
        [reference.scalar_cost(word) for word in words], dtype=torch.float64
    )
    torch.testing.assert_close(problem.values(rows)[0], expected, atol=1e-12, rtol=0)
    feasible = problem.feasible_words()
    assert feasible and all(problem.valid(word) for word in feasible)
    assert not problem.valid(0) and not problem.valid((1 << 24) - 1)
    physical = torch.tensor([1 << bit for bit in range(24)])
    assert workflow.physical_to_logical(physical).tolist() == [
        1 << (23 - bit) for bit in range(24)
    ]


@pytest.mark.parametrize(
    "parameters",
    [
        workflow.initial_parameters(),
        tuple((-1) ** i * (0.017 + 0.003 * i) for i in range(40)),
    ],
)
def test_full_state_absolute_phase_and_original_width(
    parameters: tuple[float, ...],
) -> None:
    program = build_program(parameters)
    actual = fq.run(program).statevector().reshape(-1)
    expected = fq.run(reference.circuit(parameters)).statevector().reshape(-1)
    assert program.circuit.n_wires == 24
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert float(actual.abs().square().sum()) == pytest.approx(1, abs=1e-12)


def test_all_original_stages_and_invalid_budgets() -> None:
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/telecom/resiliency_planning/overview/walkthrough.ipynb"
        ).read_text()
    )
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert [cell["metadata"]["source_stage"] for cell in cells] == [
        1,
        3,
        4,
        6,
        7,
        9,
        11,
        13,
        14,
        15,
        17,
        18,
        19,
        20,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
    ]
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
    assert len(workflow.initial_parameters()) == 40 and problem.LAYERS == 20
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="positive integers"):
            workflow.optimize(queries=invalid)
