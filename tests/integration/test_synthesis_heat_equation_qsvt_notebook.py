"""Verify fresh heat equation block reconstruction and full source iterations."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.heat_eq_qsvt.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.cfd.heat_eq_qsvt.main import PHASES
from examples.synthesis.applications.cfd.heat_eq_qsvt.overview import (
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def result() -> workflow.Result:
    return workflow.run_workflow()


def test_fresh_fit_preserves_the_source_cubic(result: workflow.Result) -> None:
    assert result.fit.max_error < 1e-12
    assert result.fit.phases == pytest.approx(PHASES, abs=1e-14, rel=0)
    parameters = torch.linspace(-1, 1, 203, dtype=torch.float64)
    target = result.fit.linear * parameters + result.fit.cubic * parameters**3
    actual = model.response(result.fit.phases, parameters)
    torch.testing.assert_close(actual.real, target, atol=1e-12, rtol=0)
    assert float(actual.imag.abs().max()) < 1e-12


def test_all_eight_columns_of_signed_operator(result: workflow.Result) -> None:
    exact = torch.diag(torch.full((8,), 1.125, dtype=torch.float64))
    neighbor = torch.diag(torch.full((7,), -0.0625, dtype=torch.float64), diagonal=1)
    exact = exact + neighbor + neighbor.T
    torch.testing.assert_close(result.model.operator, exact, atol=0, rtol=0)
    torch.testing.assert_close(result.measured.real, exact, atol=1e-12, rtol=0)
    assert float(result.measured.imag.abs().max()) < 1e-12
    assert result.matrix_error < 1e-12


@pytest.mark.parametrize("step", range(4))
def test_every_iteration_keeps_full_signed_state_and_clean_workspace(
    result: workflow.Result, step: int
) -> None:
    vector = result.inputs[step]
    program = quantum.synthesize_iteration(
        result.model.rotation, vector, result.fit.phases
    )
    state = quantum.full_state(program).reshape(256, -1)
    reference = reference_circuit(result.fit.phases, tuple(map(float, vector)))
    expected = fq.run(reference).statevector().flatten()
    torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-24
    independent_projection = torch.stack([expected[index << 4] for index in range(8)])
    anchor = torch.exp(1j * torch.angle(expected[result.gauge_indices[step]]))
    readout = (independent_projection / anchor).real
    assert result.projection_norms[step] == pytest.approx(float(readout.norm()))
    if step < 3:
        torch.testing.assert_close(
            result.inputs[step + 1],
            readout / readout.norm(),
            atol=1e-12,
            rtol=0,
        )


def test_fourth_angle_remains_unused(result: workflow.Result) -> None:
    vector = result.inputs[0]
    phases = result.fit.phases
    expected = quantum.full_state(
        quantum.synthesize_iteration(result.model.rotation, vector, phases)
    )
    changed = quantum.full_state(
        quantum.synthesize_iteration(
            result.model.rotation, vector, (*phases[:3], phases[3] + 1.2)
        )
    )
    torch.testing.assert_close(changed, expected, atol=1e-12, rtol=0)


def test_all_source_stages_and_fields(result: workflow.Result) -> None:
    book = json.loads(Path(model.__file__).with_name("walkthrough.ipynb").read_text())
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 19
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
    assert result.fields.shape == (4, 8)
    assert result.model.times.numel() == 1024
    for i, field in enumerate(result.fields):
        gain = torch.tensor(result.projection_norms[:i], dtype=torch.float64).prod()
        torch.testing.assert_close(
            field, gain * result.model.initial_norm * result.inputs[i]
        )
