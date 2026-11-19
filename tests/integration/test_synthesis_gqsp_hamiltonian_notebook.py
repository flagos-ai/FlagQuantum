"""Check all phase rows, two complex states and the original evolution scale."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq

np = pytest.importorskip("numpy")

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_gqsp.overview import (  # noqa: E402
    phases,
    quantum,
    reference,
    workflow,
)
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.model import (  # noqa: E402
    ALPHA,
    TIME,
    fresh_input,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def fitted() -> phases.Factorization:
    return phases.factor(TIME * ALPHA)


def test_source_degree_and_independent_laurent_values(
    fitted: phases.Factorization,
) -> None:
    assert len(fitted.rows) == len(fitted.coefficients) == 67
    assert fitted.completion_residual < 1e-10
    assert fitted.stripping_residual < 1e-10
    assert fitted.reconstructed_residual < 1e-10
    assert fitted.approximation_error < 1e-7
    theta = (np.arange(251) + 0.31) * 2 * np.pi / 251
    powers = np.arange(-33, 34)
    values = np.exp(1j * theta[:, None] * powers) @ np.asarray(fitted.coefficients)
    expected = phases.SCALE * np.exp(-1j * TIME * ALPHA * np.cos(theta))
    assert np.max(np.abs(values - expected)) < 1e-7


@pytest.mark.parametrize("seed", (7, 19))
def test_full_native_state_matches_independent_dense_controls(
    fitted: phases.Factorization, seed: int
) -> None:
    values = tuple(map(float, fresh_input(seed)))
    circuit = quantum.evolve(values, fitted)
    native = fq.run(circuit).statevector()
    expected = fq.run(reference.reference_circuit(values, fitted)).statevector()
    torch.testing.assert_close(native, expected, rtol=0, atol=1e-12)
    assert native.numel() == 32
    assert circuit.n_qubits == 5


@pytest.mark.parametrize("seed", (7, 19))
def test_absolute_projections_require_no_phase_alignment(
    fitted: phases.Factorization, seed: int
) -> None:
    pauli = workflow.block_projection(seed)
    evolved = workflow.evolution_projection(fitted, seed)
    assert pauli.error < 1e-12
    assert evolved.error < 1e-7
    assert pauli.scale == ALPHA
    assert evolved.scale == 1 / phases.SCALE
    assert evolved.fidelity == pytest.approx(1.0, abs=1e-12)


def test_reject_nonfinite_phase_fit() -> None:
    with pytest.raises(ValueError, match="time must be finite"):
        phases.factor(float("nan"))


def test_all_clean_source_stages_are_mapped() -> None:
    root = Path(workflow.__file__).parent
    book = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 13
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
