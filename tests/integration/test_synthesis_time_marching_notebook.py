"""Verify fresh QSVT fits, the coherent counter, and both finite outcomes."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.overview.approximation import (
    amplitude_polynomial,
    exponential,
)
from examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.overview.quantum import (
    counter_permutation,
)
from examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.overview.workflow import (
    SHOTS,
    run,
)

pytestmark = pytest.mark.integration


def test_original_four_slices_and_million_draws_per_path() -> None:
    result = run()
    assert result.classical.shape == (100, 4)
    assert result.magnus_block_error < 1e-10
    assert result.scalar_fit_error < 1e-5
    assert result.amplified.exact_prob > 100 * result.naive.exact_prob
    assert result.amplified.normalized_error < 0.1
    for path in (result.naive, result.amplified):
        assert path.resources.n_qubits == 11
        assert path.state_error < 1e-10
        assert 0 < sum(path.accepted_counts) < SHOTS
        assert float(path.sampled_magnitudes.square().sum()) == pytest.approx(
            sum(path.accepted_counts) / SHOTS
        )


def test_fresh_fitted_degrees_and_coherent_counter_permutation() -> None:
    ex = exponential(0.5)
    amp = amplitude_polynomial()
    assert (
        ex.even_fit.phases.numel(),
        ex.odd_fit.phases.numel(),
        amp.phase_fit.phases.numel(),
    ) == (7, 8, 8)
    assert (
        max(ex.even_fit.residual, ex.odd_fit.residual, amp.phase_fit.residual) < 1e-10
    )
    assert amp.interval_error < 0.01 and amp.maximum_magnitude < 1
    for flag_bits in (5, 6):
        permutation = counter_permutation(flag_bits)
        indices = torch.arange(permutation.size(0))
        expected = indices.clone()
        expected[:8] = (indices[:8] - 1) % 8
        assert torch.equal(permutation.abs().argmax(dim=0), expected)


def test_original_cells_have_no_checked_in_outputs() -> None:
    overview = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/algorithms/quantum_differential_equations_solvers/time_marching/overview"
    )
    cells = json.loads((overview / "walkthrough.ipynb").read_text())["cells"]
    proof = json.loads((overview / "provenance.json").read_text())
    stages = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 14
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in stages
    )
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        7,
        8,
        10,
        12,
        16,
        19,
        21,
        25,
        27,
        30,
        32,
        34,
        35,
        39,
    ]
