"""Validate source-scale polynomial fits and both actual LCU states."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview import (
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def test_all_original_configurations_and_inverse_errors() -> None:
    result = workflow.run_workflow()
    matrix = model.source_matrix()
    assert matrix.shape == (16, 16) and int(matrix.count_nonzero()) == 45
    eigenvalues = torch.linalg.eigvalsh(matrix.T @ matrix).sqrt() / result.nozzle.alpha
    assert float(eigenvalues[0]) == pytest.approx(
        result.nozzle.smallest, abs=1e-13, rel=0
    )
    assert float(eigenvalues[-1]) == pytest.approx(
        result.nozzle.largest, abs=1e-13, rel=0
    )
    assert result.nozzle.cks_degree == 2575
    assert result.nozzle.cks_power == 75940
    assert (
        tuple((a.degree, a.method) for a in result.approximations)
        == model.APPROXIMATIONS
    )
    short, long, uniform, cks = result.approximations
    for case in (short, long):
        a = result.nozzle.smallest
        expected = 1 / math.cosh(
            (case.degree + 1) / 2 * math.acosh((1 + a * a) / (1 - a * a))
        )
        assert case.relative_error == pytest.approx(expected, abs=5e-12, rel=0)
    assert long.relative_error < short.relative_error
    assert uniform.uniform_error < 0.01666 < long.uniform_error
    assert cks.coefficients.numel() == 128 and cks.uniform_error > long.uniform_error
    assert result.exact.resources.operation_counts["ry"] == 127
    assert result.approximate.resources.operation_counts["ry"] == 15
    assert result.approximate.resources.operation_counts["cx"] == 14
    assert result.approximate.bits_grouped == 3
    assert result.approximate.amplitude_error < model.PREPARATION_ERROR
    assert result.approximate.probability_error < model.PREPARATION_ERROR
    torch.testing.assert_close(
        result.exact.coefficients, result.fitted, atol=1e-13, rtol=0
    )


@pytest.mark.parametrize("tolerance", [0.0, 0.03])
@pytest.mark.parametrize("skewed", [False, True])
def test_full_native_state_matches_independent_circuit(
    tolerance: float, skewed: bool
) -> None:
    if skewed:
        raw = torch.linspace(1.2, 0.4, 128, dtype=torch.float64)
    else:
        nozzle = model.build_model()
        raw = model.fit_alternating(
            model.coefficient_approximation(nozzle, 255, "relative").coefficients
        ).abs()
    probabilities = raw / raw.sum()
    program, bits = quantum.compile_distribution(probabilities, tolerance)
    expected = probabilities.clone()
    for first in range(0, 128, 1 << bits):
        section = expected[first : first + (1 << bits)]
        section[:] = section.mean()
    amplitudes = expected.sqrt().to(torch.complex128)
    unit = torch.eye(128, dtype=torch.complex128)
    delta = unit[:, 0] - amplitudes
    preparation = unit - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    circuit = fq.Circuit(7, dtype=torch.complex128).any(*range(7), matrix=preparation)
    actual = fq.run(program).statevector().flatten()
    reference = fq.run(circuit).statevector().flatten()
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
    assert float((actual - probabilities.sqrt()).norm()) <= tolerance + 1e-12
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("tolerance", [-0.03, math.nan])
def test_invalid_error_refuses(tolerance: float) -> None:
    probabilities = torch.full((128,), 1 / 128, dtype=torch.float64)
    with pytest.raises(ValueError, match="tolerance"):
        quantum.compile_distribution(probabilities, tolerance)


def test_all_source_stages() -> None:
    folder = Path(workflow.__file__).parent
    notebook = json.loads((folder / "walkthrough.ipynb").read_text())
    proof = json.loads((folder / "provenance.json").read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 10
    assert all(c["execution_count"] is None for c in code)
