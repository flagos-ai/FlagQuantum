"""Check fresh graph energy, exact native amplitudes and finite-shot reporting."""

import itertools
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.qaoa.qaoa_demonstration.main import (
    HAMILTONIAN,
)
from examples.synthesis.tutorials.technology_demonstrations.qaoa.qaoa_demonstration.overview import (
    model,
    quantum,
    workflow,
)
from examples.synthesis.tutorials.technology_demonstrations.qaoa.qaoa_demonstration.overview.main import (
    historical_energies,
)

pytestmark = pytest.mark.integration


def test_graph_and_every_word_recreate_the_retained_energy() -> None:
    candidate = model.build_model()
    source_edges = {
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 6),
        (1, 2),
        (1, 3),
        (1, 5),
        (1, 6),
        (2, 3),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
        (5, 6),
    }
    assert set(candidate.edges) == source_edges
    assert len(candidate.conflicts) == 6
    exact = {}
    for size in range(8):
        for clique in itertools.combinations(range(7), size):
            violated = sum(
                i != j and (min(i, j), max(i, j)) not in source_edges
                for i in clique
                for j in clique
            )
            exact[clique] = 2 * violated**2 - size
    assert len(candidate.energies) == 128
    assert all(
        cost == exact[model.decode(word)]
        for word, cost in enumerate(candidate.energies)
    )
    assert (min(candidate.energies), max(candidate.energies)) == (-4, 281)
    retained = {
        sum(1 << lane for lane, _ in term.factors): float(term.coefficient)
        for term in HAMILTONIAN.terms
    }
    assert len(candidate.coefficients) == 61
    assert dict(candidate.coefficients) == retained


@pytest.mark.parametrize("seed", [0.031, -0.023])
def test_absolute_amplitudes_match_dense_circuit(seed: float) -> None:
    candidate = model.build_model()
    angles = tuple(seed * (i + 1) for i in range(20))
    native = quantum.build_program(candidate, angles)
    diagonal = torch.tensor(candidate.energies, dtype=torch.float64)
    wires = tuple(range(7))
    reference = fq.Circuit(7, dtype=torch.complex128)
    for wire in wires:
        reference.h(wire)
    for gamma, beta in zip(angles[::2], angles[1::2], strict=True):
        reference.any(*wires, matrix=torch.diag(torch.exp(-1j * gamma * diagonal)))
        for wire in wires:
            reference.rx(wire, beta)
    expected = fq.run(reference).statevector().reshape(-1)
    actual = fq.run(native).statevector().reshape(-1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.tensor(quantum.probabilities(native), dtype=torch.float64),
        expected.abs().square(),
        atol=1e-12,
        rtol=0,
    )


def test_measured_query_ceiling_and_resource_basis() -> None:
    result = workflow.run_workflow(query_budget=2)
    assert len(result.trace) == 2
    assert result.budget_reached and not result.converged
    assert result.iterations is None
    assert result.mean == min(row.cost for row in result.trace)
    assert result.mean == pytest.approx(32.2226)
    assert result.nodes == (1, 2, 3, 5)
    assert result.clique and result.count == 12
    assert result.resources.n_qubits == 7
    assert set(result.resources.operation_counts) == {"cx", "u3"}


def test_notebook_stages_and_historical_trace() -> None:
    directory = Path(model.__file__).parent
    book = json.loads((directory / "walkthrough.ipynb").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == len(provenance["reviewed_code_cells"]) == 12
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
    assert len(historical_energies()) == 218


def test_invalid_angles_and_query_ceiling() -> None:
    with pytest.raises(ValueError, match="twenty finite angles"):
        quantum.build_program(model.build_model(), (float("nan"),) * 20)
    with pytest.raises(ValueError, match="four hundred measured queries"):
        workflow.run_workflow(query_budget=401)
