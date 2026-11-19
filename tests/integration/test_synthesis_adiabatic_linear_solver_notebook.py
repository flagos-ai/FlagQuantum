"""Full state, reconstruction, and budget checks for all three adiabatic runs."""

import json
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("index", (0, 1, 2))
def test_full_state_and_direct_solution(index: int) -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.overview.model import (
        decompose,
    )
    from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.overview.reference import (
        reference_circuit,
    )
    from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.overview.workflow import (
        SCENARIOS,
        evolve,
        source_models,
    )

    original, perturbed = source_models()
    model = (original, perturbed, perturbed)[index]
    groups = decompose(model.h0), decompose(model.h1)
    _, time, steps = SCENARIOS[index]
    result = evolve(model, time, steps)
    circuit = reference_circuit(model, groups, time, steps)
    expected = fq.run(circuit).statevector().detach().cpu().numpy().reshape(8)
    np.testing.assert_allclose(result.state, expected, atol=2e-11, rtol=0)
    np.testing.assert_allclose(
        model.matrix @ model.exact[:4],
        model.rhs / np.linalg.norm(np.linalg.solve(model.matrix, model.rhs)),
        atol=1e-12,
        rtol=0,
    )
    assert result.fidelity == pytest.approx(
        (0.9692151882, 0.8106493159, 0.9512522382)[index], abs=1e-9
    )
    resources = result.program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 3
    assert np.count_nonzero(np.abs(result.state[4:]) > 1e-6) == 4


def test_fresh_endpoints_and_declared_order() -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.comparisons.circuit import (
        H0,
        H1,
    )
    from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.overview.model import (
        decompose,
        pauli_matrix,
        spectrum,
    )
    from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.overview.workflow import (
        source_models,
    )

    for model in source_models():
        points, energies = spectrum(model)
        assert len(points) == 100 and points[0] == 0.0 and points[-1] == 1.0
        for ix, matrix in ((0, model.h0), (-1, model.h1)):
            pieces = decompose(matrix)
            reconstructed = sum(
                term.coefficient * pauli_matrix(term.axes) for term in pieces
            )
            np.testing.assert_allclose(reconstructed, matrix, atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                energies[ix], np.linalg.eigvalsh(matrix), atol=1e-12, rtol=0
            )
    perturbed = source_models()[1]
    for terms, pinned in ((decompose(perturbed.h0), H0), (decompose(perturbed.h1), H1)):
        assert tuple(
            "".join("IXYZ"[axis] for axis in term.axes) for term in terms
        ) == tuple(word for word, _ in pinned)
        np.testing.assert_allclose(
            [term.coefficient for term in terms],
            [weight for _, weight in pinned],
            atol=1e-12,
            rtol=0,
        )


def test_reject_oversized_schedule_and_insufficient_width() -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.overview.quantum import (
        build_program,
    )
    from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.overview.workflow import (
        source_models,
    )

    model = source_models()[0]
    with pytest.raises(ValueError, match="step count"):
        build_program(model, 7, 101)
    with pytest.raises(ValueError, match=r"max_qubits=2, actual=3"):
        build_program(model, 7, 50, max_qubits=2)


def test_notebook_source_stages_and_clean_outputs() -> None:
    from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.overview import (
        __file__ as overview_path,
    )

    assert overview_path is not None
    root = Path(overview_path).parent
    book = json.loads((root / "walkthrough.ipynb").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(provenance["reviewed_code_cells"]) == 24
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
