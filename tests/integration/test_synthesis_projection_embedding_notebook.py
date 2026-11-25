"""Original-size projection embedding and independent coherent/energy oracles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from examples.synthesis.applications.chemistry.projection_based_embedding.projected_based_embedding_tutorial.overview.workflow import (
        Result,
    )

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def completed() -> Result:
    pytest.importorskip("numpy")
    pytest.importorskip("pyscf")
    from examples.synthesis.applications.chemistry.projection_based_embedding.projected_based_embedding_tutorial.overview import (
        workflow,
    )

    return workflow.run()


def test_original_chemistry_and_four_embedding_validations(completed: Result) -> None:
    molecule = completed.molecule
    assert molecule.one_body.shape == (8, 8) and molecule.two_body.shape == (8, 8, 8, 8)
    assert molecule.dft_energy == pytest.approx(-76.42037833355553, abs=1e-8)
    assert molecule.fci_energy == pytest.approx(-76.03806086432257, abs=1e-8)
    assert molecule.density_trace == pytest.approx(10, abs=1e-8)
    assert molecule.active_metric_error < 1e-12
    assert len(molecule.orbital_weights) == 5 and min(molecule.orbital_weights) > 0.3
    assert len(molecule.virtual_singular_values) == 5
    assert molecule.environment_projector_norm == molecule.embedding_potential_norm == 0
    assert molecule.environment_correction == 0
    assert molecule.fermion == molecule.physical_fermion


def test_native_frame_and_all_complete_nonzero_state_amplitudes(
    completed: Result,
) -> None:
    assert completed.molecule.reduction.n_qubits == 14
    assert len(completed.state_errors) == 2 and max(completed.state_errors) < 1e-12
    assert len(completed.energy_errors) == 3 and max(completed.energy_errors) < 1e-9
    assert all(abs(term.coefficient) > 0.1 for term in completed.molecule.trimmed.terms)
    assert completed.molecule.physical_trimmed == completed.molecule.trimmed


def test_actual_full_optimizer_budget_and_unrepaired_estimates(
    completed: Result,
) -> None:
    result = completed.optimization
    assert len(result.trace) == 100
    assert all(len(row.parameters) == 30 for row in result.trace)
    assert result.selected in result.trace
    assert result.selected.energy == min(row.energy for row in result.trace)
    assert result.query_budget_reached and not result.converged
    assert result.physical_trimmed_energy == pytest.approx(
        result.selected.energy, abs=1e-10
    )
    assert result.physical_untrimmed_energy >= completed.molecule.fci_energy - 1e-9


def test_invalid_singles_angles_fail_before_execution(completed: Result) -> None:
    from examples.synthesis.applications.chemistry.projection_based_embedding.projected_based_embedding_tutorial.overview import (
        variational,
    )

    operators = variational.operators(completed.molecule)
    assert len(operators) == 30
    with pytest.raises(ValueError, match="thirty"):
        variational.circuit(completed.molecule, operators, (0.0,))
    with pytest.raises(ValueError, match="finite"):
        variational.circuit(completed.molecule, operators, (float("nan"),) * 30)


def test_all_source_stages_are_unexecuted_native_cells() -> None:
    base = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/chemistry/projection_based_embedding/projected_based_embedding_tutorial/overview"
    )
    notebook = json.loads((base / "walkthrough.ipynb").read_text())
    proof = json.loads((base / "provenance.json").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 21
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
