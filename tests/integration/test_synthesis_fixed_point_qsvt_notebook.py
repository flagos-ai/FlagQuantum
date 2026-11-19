"""Verify fresh phases, all arithmetic words, and the complete native state."""

import json
from itertools import product
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth

np = pytest.importorskip("numpy")

from examples.synthesis.algorithms.amplitude_amplification.qsvt_fixed_point.comparisons.circuit import (  # noqa: E402
    reference_circuit,
)
from examples.synthesis.algorithms.amplitude_amplification.qsvt_fixed_point.overview import (  # noqa: E402
    fitting,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def fit() -> fitting.PhaseFit:
    return fitting.fit_source()


def test_exact_arithmetic_truth_table() -> None:
    marked = set(quantum.matching_words())
    assert len(marked) == 2
    for a, b, c in product(range(4), range(2), range(8)):
        assert ((a, b, c) in marked) == ((a + b, c - a) == (3, 2))


def test_degree_domain_target_and_physically_verified_phases(
    fit: fitting.PhaseFit,
) -> None:
    assert len(fit.coefficients) == len(fit.phases) == 26
    assert fit.coefficients[::2] == (0.0,) * 13
    assert abs(fit.coefficients[-1]) > 0.04
    assert fit.target_error < 0.035
    assert fit.phase_error < 1e-10
    assert fit.amplitude_bound < 1
    points = np.cos(np.arange(421) * np.pi / 420)
    polynomial = fitting.polynomial(points, fit.coefficients)
    actual = fitting.phase_response(
        torch.from_numpy(points), torch.tensor(fit.phases, dtype=torch.float64)
    )
    np.testing.assert_allclose(actual.numpy(), polynomial, rtol=0, atol=1e-10)


@pytest.mark.parametrize("perturbation", [0.0, 0.037])
def test_native_absolute_amplitudes_and_clean_work(
    fit: fitting.PhaseFit, perturbation: float
) -> None:
    phases = tuple(
        value + perturbation * (index % 3 - 1) for index, value in enumerate(fit.phases)
    )
    program = quantum.build_program(phases)
    actual = fq.run(program).statevector().flatten()
    width = actual.numel().bit_length() - 1
    wires = tuple(
        wire for name in ("a", "b", "c", "aux") for wire in program.registers[name]
    )
    work = tuple(wire for wire in range(width) if wire not in wires)
    ordered = actual.reshape((2,) * width).permute((*wires, *work)).reshape(128, -1)
    reference = fq.run(reference_circuit(phases)).statevector().flatten()
    torch.testing.assert_close(ordered[:, 0], reference, atol=1e-12, rtol=0)
    assert float(ordered[:, 1:].abs().square().sum()) < 1e-24


def test_original_sample_count_and_five_sigma(fit: fitting.PhaseFit) -> None:
    result = workflow.run_search(fit)
    assert sum(count for _, count in result.counts) == 2048
    assert abs(result.good - result.expected) <= 5 * result.sigma
    assert result.actual_probability == pytest.approx(
        result.predicted_probability, abs=2e-12
    )
    assert result.resources.n_qubits == 12


def test_insufficient_width_is_refused(fit: fitting.PhaseFit) -> None:
    with pytest.raises(synth.SynthesisError, match="max_qubits=11"):
        quantum.build_program(fit.phases, max_qubits=11)


def test_notebook_retains_all_clean_source_stages() -> None:
    root = Path(workflow.__file__).parent
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 13
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in cells
    )
