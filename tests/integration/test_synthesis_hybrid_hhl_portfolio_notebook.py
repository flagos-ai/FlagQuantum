"""Verify the measured feed-forward path and independent complete HHL states."""

import json
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from examples.synthesis.hhl_reference import householder, matrix_hhl_circuit

pytest.importorskip("numpy")
from examples.synthesis.applications.finance.hybrid_hhl_for_portfolio_optimization.portfolio_optimization_with_hhl.overview import (
    model,
    quantum,
    workflow,
)

pytestmark = pytest.mark.integration


def test_spectroscopy_matches_the_direct_fourier_sum() -> None:
    m = model.build_model()
    direct_forcing = torch.tensor((2.4, 1.0, 0.0, 0.0), dtype=torch.float64)
    torch.testing.assert_close(
        m.operator @ m.direct, direct_forcing / direct_forcing.norm()
    )
    bins = torch.arange(16, dtype=torch.float64)
    shifted = (m.eigenvalues + m.eigenvalue_bound) * m.scale
    exponent = 2j * torch.pi * (shifted[:, None] - bins[None, :] / 16)
    kernels = torch.exp(exponent[:, :, None] * bins).mean(-1)
    weights = (m.eigenvectors.T @ m.normalized_input).to(torch.complex128)
    expected = (kernels * weights[:, None]).T @ m.eigenvectors.T.to(torch.complex128)
    torch.testing.assert_close(
        quantum.state(quantum.spectroscopy(m)), expected.flatten(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("profile", ["reduced_depth", "reduced_width", "basic_width"])
def test_hhl_matches_absolute_circuit_state(profile: str) -> None:
    m = model.build_model()
    spectroscopy = quantum.spectroscopy(m)
    counts = workflow.sample(spectroscopy, ("phase",), 9121)
    spectrum = workflow.infer_spectrum(m, counts)
    assert spectrum.selected_words == (6, 10)
    table = (
        m.basic_angles
        if profile == "basic_width"
        else model.reduced_table(m, spectrum.selected_words)
    )
    target: Literal["depth", "width"] = (
        "depth" if profile == "reduced_depth" else "width"
    )
    native = quantum.build_hhl(m, table, target)
    phase_bits = len(table).bit_length() - 1
    phase = tuple(range(2, 2 + phase_bits))
    data = (phase_bits + 2, phase_bits + 3)
    compared = (phase_bits + 4, phase_bits + 5)
    reference = matrix_hhl_circuit(
        phase_bits + 6,
        data,
        phase,
        0,
        tuple(tuple(map(complex, row)) for row in m.evolution),
        table,
        tuple(map(float, m.normalized_input)),
    )
    reference.any(*compared, matrix=householder(tuple(map(float, m.comparison))))
    reference.h(1).gate("cswap", (1, data[0], compared[0]))
    reference.gate("cswap", (1, data[1], compared[1])).h(1)
    assert native.registers == {"indicator": (0,), "test": (1,)}
    torch.testing.assert_close(
        quantum.state(native),
        fq.run(reference).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )


def test_complete_finite_workflow_and_resource_basis() -> None:
    m, spectrum, profiles = workflow.run_experiment()
    depth, width, basic = profiles
    assert sum(spectrum.counts) == model.SHOTS
    assert tuple(
        i
        for i, angle in enumerate(model.reduced_table(m, spectrum.selected_words))
        if angle
    ) == (3, 5)
    assert len(m.basic_angles) == 16
    assert depth.counts == width.counts == (1154, 799, 91, 4)
    assert basic.counts == (1132, 797, 108, 11)
    assert depth.overlap == pytest.approx(0.9569688990161648)
    assert basic.overlap == pytest.approx(0.9028433144351062)
    assert tuple(row.resources.n_qubits for row in profiles) == (9, 9, 10)
    # Rejecting the pulse proposal recovers the general factor sequence. Both
    # paths must preserve the same finite outcomes and complete converted state.
    with patch(
        "flagquantum.compiler.optimization.two_qubit_synthesis._verified_cx_pulses",
        return_value=False,
    ):
        _, _, control = workflow.run_experiment()
    assert tuple(row.resources.operation_counts["cx"] for row in control) == (
        612,
        612,
        824,
    )
    assert tuple(row.resources.operation_counts["cx"] for row in profiles) == (
        498,
        498,
        672,
    )
    for actual, reference in zip(profiles, control, strict=True):
        assert actual.counts == reference.counts
        assert actual.overlap == reference.overlap
        assert actual.resources.n_qubits == reference.resources.n_qubits
        assert reference.projective_error < 1e-12
    for row in profiles:
        assert sum(row.counts) == model.SHOTS
        assert set(row.resources.operation_counts) == {"cx", "u3"}
        assert row.projective_error < 1e-12
    assert workflow.conditional_overlap((1024, 1024, 0, 0)) is None
    with pytest.raises(ValueError, match="four observed phase rows"):
        workflow.infer_spectrum(m, (2048,) + (0,) * 15)


def test_book_covers_all_stages() -> None:
    overview = Path(model.__file__).parent
    book = json.loads((overview / "walkthrough.ipynb").read_text())
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    proof = json.loads((overview / "provenance.json").read_text())
    assert len(code) == len(proof["reviewed_code_cells"]) == 29
    assert all(row["execution_count"] is None and row["outputs"] == [] for row in code)
