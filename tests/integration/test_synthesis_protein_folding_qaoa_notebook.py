"""Verify the complete native objective, signed state and source layout."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_qaoa.protein_folding.comparisons.circuit import (
    BINDINGS,
    cost,
    reference_circuit,
)
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_qaoa.protein_folding.overview import (
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def test_all_source_energies_and_lattice_minimizers() -> None:
    energies = model.costs()
    assert all(type(value) is int for value in energies)
    assert energies == tuple(
        -cost(tuple(word >> i & 1 for i in range(7))) for word in range(128)
    )
    minima = tuple(
        word for word, value in enumerate(energies) if value == min(energies)
    )
    assert min(energies) == -2600 and len(minima) == 2
    assert all(len(model.lattice(word)) == 6 for word in minima)


@pytest.mark.parametrize("params", BINDINGS)
def test_native_absolute_state_and_named_cost_word(params: tuple[float, ...]) -> None:
    program = quantum.build_program(params)
    state = fq.run(program).statevector().flatten()
    reference = fq.run(reference_circuit(params)).statevector().flatten()
    # The reference declares the fold first and the contact in the low bit.
    reordered = reference[[((word & 63) << 1) | (word >> 6) for word in range(128)]]
    torch.testing.assert_close(state, reordered, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        quantum.probabilities(program), reordered.abs().square(), atol=1e-12, rtol=0
    )


def test_cvar_uses_fractional_boundary_mass() -> None:
    values = torch.tensor([1, -3, 2], dtype=torch.float64)
    assert float(workflow.lower_tail_mean(values)) == pytest.approx(
        (-3 + 1 + 0.1 * 2) / 2.1
    )


def test_source_stages_are_unexecuted() -> None:
    path = Path(model.__file__).parent
    book = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 14
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
