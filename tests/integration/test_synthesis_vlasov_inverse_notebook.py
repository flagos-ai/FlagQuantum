"""Check original-size inversion, full coherent state, polynomial and execution bounds."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.model import (
    evaluate,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    response,
)
from examples.synthesis.applications.plasma.vlasov_ampere.overview import (
    execution,
    model,
    quantum,
    reference,
    workflow,
)
from examples.synthesis.applications.plasma.vlasov_ampere.vlasov_ampere_qiskit.overview import (
    physics,
)

pytestmark = pytest.mark.integration


def test_original_full_workflow_and_all_operator_entries() -> None:
    result = workflow.run()
    torch.testing.assert_close(
        result.encoded_matrix, physics.matrix(), atol=1e-12, rtol=0
    )
    assert result.state.numel() == 262144
    assert result.resources.n_qubits == 18 and result.resources.n_operations == 13560
    assert result.resources.basis == "static_instruction_sequence"
    assert result.full_state_error < 1e-10 and result.polynomial_error < 1e-10
    assert len(result.model.phases) == 252
    assert result.model.kappa == pytest.approx(20.212685987466962, abs=1e-10)
    assert result.model.inverse_normalization == pytest.approx(
        14.308109711660414, abs=1e-10
    )
    assert result.linear_solution_error == pytest.approx(0.0383841827342418, abs=1e-10)
    assert result.projection_probability == pytest.approx(
        0.006865736245701943, abs=1e-12
    )
    rhs = 1j * result.model.source.to(torch.complex128) * result.model.source_norm
    expected = torch.linalg.solve(physics.matrix(), rhs)[64::8]
    torch.testing.assert_close(result.classical_field, expected, atol=1e-12, rtol=0)
    assert (
        float(result.state.reshape(2, 4, 256, 128)[:, 1:].abs().square().sum()) < 1e-24
    )


def test_fresh_full_degree_response_on_an_independent_grid() -> None:
    value = model.build()
    grid = torch.linspace(-1.0, 1.0, 8193, dtype=torch.float64)
    target = evaluate(value.coefficients, grid)
    actual = response(grid, value.fitted.phases)
    torch.testing.assert_close(actual, target, atol=1e-10, rtol=0)
    assert float(target.abs().max()) < 0.951
    assert value.fitted.iterations > 0


@pytest.mark.parametrize(
    "phases,segment",
    [((0.13, -0.27, 0.19, -0.41), 32), ((-0.17, 0.29, -0.23, 0.37), 64)],
)
def test_segmented_coherent_transfer_and_all_clean_work(
    phases: tuple[float, ...], segment: int
) -> None:
    value = replace(model.build(), phases=torch.tensor(phases, dtype=torch.float64))
    program = quantum.circuit(value)
    assert len(program.to_ir().instructions) > 2 * segment
    actual = execution.execute(program, segment_size=segment)
    expected = reference.full_state(value)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert float(actual.norm()) == pytest.approx(1.0, abs=1e-12)
    assert float(actual.reshape(2, 4, 256, 128)[:, 1:].abs().square().sum()) < 1e-24


def test_invalid_bounds_and_original_stage_map() -> None:
    value = model.build()
    for size in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            execution.execute(quantum.circuit(value), segment_size=size)
    with pytest.raises(ValueError, match="128 by 128"):
        model.build(torch.eye(2, dtype=torch.complex128))
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/plasma/vlasov_ampere/overview/walkthrough.ipynb"
        ).read_text()
    )
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert [cell["metadata"]["source_stage"] for cell in cells] == [
        3,
        4,
        5,
        8,
        14,
        16,
        18,
        20,
        22,
        25,
        27,
        30,
        34,
        37,
        39,
        41,
        44,
        46,
        49,
        53,
        56,
        59,
        60,
        62,
        64,
        66,
        67,
    ]
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
