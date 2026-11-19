"""Verify the complete Pauli block, controlled powers, and complex phase."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.integration


def test_complete_pauli_block_matches_independent_dense_matrix() -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.model import (
        ALPHA,
        DTYPE,
        hamiltonian_matrix,
    )
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.quantum import (
        hamiltonian_block,
    )

    actual = []
    for vector in torch.eye(4, dtype=torch.float64):
        program = hamiltonian_block(tuple(map(float, vector)))
        actual.append(fq.run(program).statevector().flatten()[:4])
    torch.testing.assert_close(
        torch.stack(actual, dim=1).to(DTYPE),
        hamiltonian_matrix() / ALPHA,
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("seed", (7, 19))
def test_no_phase_or_norm_correction_on_full_source_projection(seed: int) -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.workflow import (
        run_workflow,
    )

    hamiltonian, evolved = run_workflow(seed)
    assert hamiltonian.circuit.n_qubits == 4
    assert evolved.circuit.n_qubits == 10
    assert hamiltonian.error < 1e-12
    assert evolved.error < 1e-7
    assert evolved.fidelity == pytest.approx(1.0, abs=1e-12)
    assert len(evolved.circuit.to_ir().instructions) < 1024


def test_quadrature_degree_and_coefficient_parity() -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.model import (
        ALPHA,
        DEGREE,
        TIME,
        expansion,
    )

    poly = expansion()
    assert len(poly.coefficients) == DEGREE + 1 == 34
    assert poly.scale == pytest.approx(5.6264790337731325, abs=1e-12)
    assert all(
        coef.real == 0 if degree % 2 else coef.imag == 0
        for degree, coef in enumerate(poly.coefficients)
    )
    grid = np.linspace(-1, 1, 8193)
    fitted = np.polynomial.chebyshev.chebval(grid, poly.coefficients)
    assert np.max(np.abs(fitted - np.exp(-1j * ALPHA * TIME * grid))) < 1.5e-8


def test_inactive_control_preserves_the_complete_five_qubit_state() -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.quantum import (
        controlled_walk,
    )

    circuit = fq.Circuit(5, dtype=torch.complex128).h(0).h(1).h(2).x(3)
    expected = fq.run(circuit).statevector()
    circuit.compose(controlled_walk(), qubits=(0, 1, 2, 3, 4))
    torch.testing.assert_close(
        fq.run(circuit).statevector(), expected, atol=2e-12, rtol=0
    )


def test_original_stages_are_unexecuted() -> None:
    from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview import (
        __file__ as overview_path,
    )

    assert overview_path is not None
    source = Path(overview_path).parent
    book = json.loads((source / "walkthrough.ipynb").read_text())
    proof = json.loads((source / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 15
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
