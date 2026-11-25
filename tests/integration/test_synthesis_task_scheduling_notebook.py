"""Verify both full task instances, original penalties, and coherent native lowering."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.logistics.task_scheduling_problem.overview import (
    problem,
    quantum,
    reference,
)

pytestmark = pytest.mark.integration


def test_all_small_costs_and_large_pair_coefficients() -> None:
    small = problem.create()
    words = torch.arange(1 << small.width, dtype=torch.int64)
    actual = problem.costs(small, words)
    expected = torch.tensor(
        [problem.original_cost(small, int(word)) for word in words], dtype=torch.float64
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    large = problem.create(True)
    assert (small.width, large.width) == (12, 24)
    assert large.slack_weights(2) == (1, 2, 1)
    assert sum(c / 2 ** len(s) for s, c in large.polynomial) == 851
    base, linear, pairs = reference.coefficients(large)
    dictionary = dict(large.polynomial)
    assert base == dictionary[()]
    for bit, coefficient in enumerate(linear):
        assert coefficient == dictionary.get((bit,), 0)
    for first, second, coefficient in pairs:
        assert coefficient == dictionary.get((first, second), 0)


@pytest.mark.parametrize("large", [False, True])
def test_complete_nonzero_state(large: bool) -> None:
    data = problem.create(large)
    parameters = quantum.initial_parameters(data)
    native = quantum.build(data, parameters)
    actual = fq.run(native).statevector().reshape(-1)
    expected = fq.run(reference.circuit(data, parameters)).statevector().reshape(-1)
    assert native.circuit.n_wires == data.width
    assert native.report["workspace_qubits"] == 0
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_independent_classical_schedules_and_source_stage_map() -> None:
    for large, finish in ((False, 3), (True, 5)):
        model = problem.create(large)
        cost, matrix = problem.classical(model)
        assert cost == finish and problem.valid(model, matrix)
        assert len(matrix) == len(model.capacity)
        assert all(len(row) == model.tasks for row in matrix)
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/logistics/task_scheduling_problem/overview/walkthrough.ipynb"
        ).read_text()
    )
    cells = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 28
    assert all(c["execution_count"] is None and not c["outputs"] for c in cells)
