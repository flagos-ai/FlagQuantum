"""Verify the source Hamiltonian, adaptive mixers, and termination rule."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.adapt_qaoa.comparisons.circuit import (
    cost,
)
from examples.synthesis.applications.optimization.adapt_qaoa.overview import (
    model,
    quantum,
    resources,
    workflow,
)

pytestmark = pytest.mark.integration


def mixer_matrix(mixer: model.Mixer) -> torch.Tensor:
    unit = torch.eye(2, dtype=torch.complex128)
    gate = (
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        if mixer.axis == "X"
        else torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    )
    total = torch.zeros((256, 256), dtype=torch.complex128)
    for support in mixer.supports:
        term = gate if 7 in support else unit
        for bit in range(6, -1, -1):
            term = torch.kron(term, gate if bit in support else unit)
        total += term
    return total


def independent_costs() -> torch.Tensor:
    return torch.tensor(
        [cost(tuple((word >> bit) & 1 for bit in range(8))) for word in range(256)],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("index", (0, 1, 2, 10, 18, 46))
@pytest.mark.parametrize("angles", ((0.13, 0.29), (-0.27, 0.52)))
def test_absolute_mixer_states_and_gradients(
    index: int, angles: tuple[float, float]
) -> None:
    mixer = model.pool()[index]
    operator = mixer_matrix(mixer)
    diagonal = independent_costs()
    gamma, beta = angles
    expected = torch.matrix_exp(-1j * beta * operator) @ (
        torch.exp(1j * gamma * diagonal) / 16
    )
    program = quantum.build_program((mixer,), angles)
    state = fq.run(program).statevector().flatten()
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    direct = torch.vdot(
        state, operator @ (diagonal * state) - diagonal * (operator @ state)
    ).abs()
    torch.testing.assert_close(
        model.gradients(state, diagonal, (mixer,))[0], direct, atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 0


def test_full_loop_and_source_basis() -> None:
    candidates = model.pool()
    assert len(candidates) == len(set(candidates)) == 74
    torch.testing.assert_close(model.costs(), independent_costs(), atol=2e-14, rtol=0)
    run = workflow.run_workflow()
    assert run.layers[0].mixers == (0,)
    assert all(layer.gradient_norm >= 0.027 for layer in run.layers[:-1])
    assert run.layers[-1].gradient_norm < 0.027
    for first, second in zip(run.layers[:-1], run.layers[1:], strict=True):
        assert second.mixers == (*first.mixers, first.next_mixer)
    for opt in (run.ordinary, *(row.optimization for row in run.layers)):
        assert 1 <= len(opt.trace) <= 60
        assert opt.parameters == min(opt.trace, key=lambda row: row.cost).parameters
    diagonal = independent_costs()
    for rows in (run.ordinary_samples, run.adaptive_samples):
        assert sum(row.count for row in rows) == 1000
        assert all(
            row.energy == pytest.approx(float(diagonal[row.word]), abs=2e-14)
            for row in rows
        )
    report = resources.source_basis_metrics(run.ordinary.program)
    assert report.resources.n_qubits == 8 and report.conversion_identity
    assert set(report.resources.operation_counts) <= {"cz", "rz", "rx", "ry"}
    assert report.maximum_state_error < 2e-12


def test_all_source_stages_unexecuted() -> None:
    path = Path(quantum.__file__).parent
    notebook = json.loads((path / "walkthrough.ipynb").read_text())
    proof = json.loads((path / "provenance.json").read_text())
    stages = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 20
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in stages
    )
