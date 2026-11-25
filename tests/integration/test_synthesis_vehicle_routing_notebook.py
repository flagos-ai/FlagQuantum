"""Verify the full original routing circuit, cost oracle, and workflow scale."""

import json
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("numpy")

import flagquantum as fq
from examples.synthesis.applications.logistics.vehicle_routing_problem.overview import (
    problem,
    quantum,
    reference,
)

pytestmark = pytest.mark.integration


def test_all_costs_and_fresh_original_geometry() -> None:
    data = problem.fresh()
    assert data == problem.fresh()
    assert data.positions[0] == pytest.approx((7.71320643266746, 0.207519493594015))
    assert data.maximum == pytest.approx(10.073880213846256)
    values = problem.costs(data)
    base, linear, pairs = reference.coefficients(data)
    words = torch.arange(65536, dtype=torch.int64)
    bits = ((words[:, None] >> torch.arange(16)) & 1).to(torch.float64)
    independent = torch.full_like(values, base)
    for bit, coefficient in enumerate(linear):
        independent += coefficient * bits[:, bit]
    for first, second, coefficient in pairs:
        independent += coefficient * bits[:, first] * bits[:, second]
    torch.testing.assert_close(values, independent, atol=1e-12, rtol=0)
    feasible = [word for word in range(65536) if problem.valid(word)]
    assert len(feasible) == 24
    best, words_optimal = problem.classical_reference(data)
    assert best == pytest.approx(0.0776670874935556)
    assert all(problem.valid(word) for word in words_optimal)


@pytest.mark.parametrize(
    "parameters",
    [
        quantum.initial_parameters(),
        tuple((-1) ** i * (0.07 + 0.03 * i) for i in range(24)),
    ],
)
def test_complete_state_and_global_phase(parameters: tuple[float, ...]) -> None:
    data = problem.fresh()
    program = quantum.build(data, parameters)
    actual = fq.run(program).statevector().reshape(-1)
    expected = reference.state(data, parameters)
    assert program.circuit.n_wires == 16
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert math.isclose(float(actual.abs().square().sum()), 1, abs_tol=1e-12)


def test_original_stage_map_and_sampling_budget() -> None:
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/logistics/vehicle_routing_problem/overview/walkthrough.ipynb"
        ).read_text()
    )
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 36
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
    assert (
        problem.WIDTH,
        problem.LAYERS,
        problem.SHOTS,
        problem.OBJECTIVE_QUERIES,
    ) == (16, 12, 30000, 26)
