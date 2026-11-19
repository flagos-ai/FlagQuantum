"""Check both original QSVT parity sequences and full evolution states."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qsvt.overview import (
    even_phases,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("seed", [132, 247])
def test_original_evolution_and_full_state(seed: int) -> None:
    value = workflow.run(seed)
    assert value.model.pauli_counts == (4, 4)
    assert value.model.alpha == 0.75
    assert value.resources.n_qubits == 6
    assert value.cosine.phases.numel() == 33
    assert value.sine.phases.numel() == 34
    assert max(value.cosine.residual, value.sine.residual) < 1e-10
    assert max(value.block_error, value.full_state_error) < 1e-10
    assert value.evolution_error < workflow.EPS
    assert 0.24 < value.success_probability < 0.26
    torch.testing.assert_close(value.state, value.expected, atol=1e-7, rtol=0)
    # Z on the source's bit one contributes to the two high bits.
    diagonal = value.model.target.diag().real
    torch.testing.assert_close(
        diagonal,
        torch.tensor([0.7, 0.3, 0.1, 0.5], dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )


def test_even_fit_derivative_at_center() -> None:
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
        response,
        signal,
    )

    x = torch.tensor([0.13, 0.51, 0.88], dtype=torch.float64)
    half = torch.tensor([0.12, -0.16, 0.21], dtype=torch.float64)
    values, jacobian = even_phases.evaluate(signal(x), half)
    torch.testing.assert_close(
        values, response(x, torch.cat((half, half[:-1].flip((0,))))), atol=1e-12, rtol=0
    )
    perturbation = torch.tensor([0, 0, 1e-6], dtype=torch.float64)
    plus, _ = even_phases.evaluate(signal(x), half + perturbation)
    minus, _ = even_phases.evaluate(signal(x), half - perturbation)
    torch.testing.assert_close(
        jacobian[:, -1], (plus - minus) / 2e-6, atol=1e-9, rtol=0
    )


def test_source_stages() -> None:
    directory = Path(workflow.__file__).parent
    book = json.loads((directory / "walkthrough.ipynb").read_text())
    proof = json.loads((directory / "provenance.json").read_text())
    cells = [c for c in book["cells"] if c["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 12
    assert all(c["execution_count"] is None for c in cells)
