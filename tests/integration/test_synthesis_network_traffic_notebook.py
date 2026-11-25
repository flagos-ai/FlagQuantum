"""Check every factor and complete native traffic states at nonzero bindings."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.telecom.network_traffic_optimization.main import (
    build_program,
)
from examples.synthesis.applications.telecom.network_traffic_optimization.overview import (
    problem,
    reference,
    workflow,
)

pytestmark = pytest.mark.integration


def test_independent_cost_tables_and_integral_flow() -> None:
    words = set(range(256))
    words.update(range((1 << 24) - 256, 1 << 24))
    for support, _ in reference.factors():
        for row in range(1 << len(support)):
            words.add(
                sum(
                    ((row >> position) & 1) << bit
                    for position, bit in enumerate(support)
                )
            )
    rows = torch.tensor(sorted(words), dtype=torch.int64)
    actual = problem.values(rows)[0]
    expected = torch.tensor(
        [reference.scalar_cost(int(word)) for word in rows], dtype=torch.float64
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    word, objective, latency = problem.classical()
    assert (objective, latency) == (10, 9)
    assert problem.valid(word)
    assert all(problem.chosen_edges(word, d) for d in range(3))
    assert not problem.valid(0)
    assert not problem.valid((1 << 24) - 1)


@pytest.mark.parametrize(
    "parameters",
    [
        workflow.initial_parameters(),
        tuple((-1) ** i * (0.05 + 0.11 * i) for i in range(10)),
    ],
)
def test_complete_state_and_global_phase(parameters: tuple[float, ...]) -> None:
    program = build_program(parameters)
    actual = fq.run(program).statevector().reshape(-1)
    expected = fq.run(reference.circuit(parameters)).statevector().reshape(-1)
    assert program.circuit.n_wires == 24
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert float(actual.abs().square().sum()) == pytest.approx(1, abs=1e-12)


def test_original_stage_map_and_full_scale() -> None:
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/telecom/network_traffic_optimization/overview/walkthrough.ipynb"
        ).read_text()
    )
    cells = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert [c["metadata"]["source_stage"] for c in cells] == [
        2,
        3,
        6,
        7,
        9,
        11,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        22,
        23,
        25,
    ]
    assert all(c["execution_count"] is None and not c["outputs"] for c in cells)
    assert (problem.WIDTH, problem.LAYERS, problem.SHOTS, problem.QUERIES) == (
        24,
        5,
        10000,
        12,
    )
