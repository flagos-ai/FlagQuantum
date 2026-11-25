"""Validate original-width oscillator matrices and both native ansatz schedules."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq

np = pytest.importorskip("numpy")

from examples.synthesis.applications.chemistry.quantum_drude_oscillator.overview import (  # noqa: E402
    workflow,
)
from examples.synthesis.applications.chemistry.quantum_drude_oscillator.overview.model import (  # noqa: E402
    ladder_projector,
    oscillator_hamiltonian,
    position_operator,
)
from examples.synthesis.applications.chemistry.quantum_drude_oscillator.overview.quantum import (  # noqa: E402
    build_program,
)
from examples.synthesis.applications.chemistry.quantum_drude_oscillator.overview.reference import (  # noqa: E402
    analytical_position_product,
    reference_circuit,
    transition_matrix,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("bits", (1, 2))
def test_position_operator_matches_analytic_pauli_product(bits: int) -> None:
    dimension = 2**bits
    lowering = ladder_projector(0, dimension, raising=False)
    raising = ladder_projector(0, dimension, raising=True)
    np.testing.assert_array_equal(raising.T, lowering)
    position = position_operator(dimension)
    np.testing.assert_allclose(
        np.kron(position, position), analytical_position_product(bits), atol=1e-14
    )


@pytest.mark.parametrize("count,dimension", [(2, 2), (2, 4), (2, 8), (2, 16), (3, 4)])
@pytest.mark.parametrize("distance", [3.1, 5.5])
def test_full_matrix_matches_independent_transition_enumeration(
    count: int, dimension: int, distance: float
) -> None:
    np.testing.assert_allclose(
        oscillator_hamiltonian(count, dimension, distance),
        transition_matrix(count, dimension, distance),
        rtol=0,
        atol=1e-13,
    )


@pytest.mark.parametrize("count,angles", [(2, 12), (3, 40)])
@pytest.mark.parametrize("scale", [0.037, -0.081])
def test_native_and_circuit_absolute_states(
    count: int, angles: int, scale: float
) -> None:
    binding = tuple(scale * (index + 1) for index in range(angles))
    program = build_program(count, binding)
    actual = fq.run(program).statevector()
    expected = fq.run(reference_circuit(count, binding)).statevector()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert actual.numel() == 2 ** (2 * count)
    assert tuple(program.registers) == ("state",)


def test_training_rereads_actual_native_final_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "ITERATION_BUDGET", 3)
    run = workflow.optimize_distance(2, 3.0)
    assert run.iterations == 3
    assert run.evaluations == 4
    assert len(run.objective_energies) == 4
    assert run.objective_energies[0] > run.energy >= run.reference_energy
    matrix = oscillator_hamiltonian(2, 4, 3.0)
    assert workflow.energy_at(2, run.angles, matrix) == pytest.approx(
        run.energy, abs=1e-12
    )


@pytest.mark.parametrize("angles", [[0.0] * 11, [float("nan")] * 12])
def test_binding_rejects_invalid_angles(angles: list[float]) -> None:
    with pytest.raises(ValueError, match="12 finite angles"):
        build_program(2, angles)


def test_notebook_is_clean_and_maps_every_source_stage() -> None:
    directory = Path(workflow.__file__).with_name("walkthrough.ipynb").parent
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(provenance["reviewed_code_cells"]) == 28
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in cells
    )
    assert [
        record["native_code_cell"] for record in provenance["reviewed_code_cells"]
    ] == list(range(1, 29))
