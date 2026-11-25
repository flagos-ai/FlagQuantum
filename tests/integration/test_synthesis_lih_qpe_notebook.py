"""Validate the fresh six-bit LiH model and the complete local QPE workflow."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.qpe_for_molecules.comparisons.circuit import (
    pauli_rotation,
)
from examples.synthesis.applications.chemistry.qpe_for_molecules.overview import (
    model,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("scale,coherent", [(1.0, False), (0.73, True)])
def test_full_states_against_ungrouped_circuit(scale: float, coherent: bool) -> None:
    fresh = model.build_model()
    target = torch.arange(1, 65, dtype=torch.float64)
    vector = (
        target / target.norm() if coherent else torch.eye(64, dtype=torch.float64)[0]
    )
    if coherent:
        vector[1::2] *= -1
    circuit = fq.Circuit(13, dtype=torch.complex128)
    if coherent:
        basis = torch.eye(64, dtype=torch.complex128)
        difference = basis[:, 0] - vector
        prepare = basis - 2 * torch.outer(
            difference, difference.conj()
        ) / difference.vdot(difference)
        circuit.any(*range(6), matrix=prepare)
    for wire in range(6, 13):
        circuit.h(wire)
    for i, (power, repeat) in enumerate(
        zip(workflow.POWERS, workflow.STEPS, strict=True)
    ):
        angle = -math.pi * scale * power / repeat
        for _ in range(repeat):
            for term in (*fresh.terms, *reversed(fresh.terms)):
                pauli_rotation(circuit, term, angle, 12 - i)
    phase_grid = torch.arange(128, dtype=torch.float64)
    inverse = torch.exp(
        -2j * math.pi * torch.outer(phase_grid, phase_grid) / 128
    ) / math.sqrt(128)
    circuit.any(*range(6, 13), matrix=inverse)
    program = workflow.build_program(fresh, scale, tuple(map(float, vector)))
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(circuit).statevector().flatten()
    assert actual.numel() == 8192
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


def test_workflow_recomputes_terms_and_samples_phase_marginal() -> None:
    result = workflow.run_workflow()
    assert len(set(model.sector_words())) == 64
    assert model.sector_words()[0] == model.HF_OCCUPATION
    assert len(result.model.coefficient_magnitudes) == 231
    assert len(result.model.terms) == 49 and result.model.reference_delta < 1e-7
    assert result.model.full_ci == pytest.approx(-7.8823869936, abs=1e-8, rel=0)
    assert result.model.full_ci < result.model.active_ci < result.model.rhf
    assert result.model.normalization == pytest.approx(17.61546224, abs=1e-6, rel=0)
    assert result.gamma == pytest.approx(0.0039068098, abs=1e-8, rel=0)
    assert workflow.STEPS == (1, 1, 1, 2, 4, 10, 26)
    assert result.adaptive_bound == (11, 29, 82, 231, 651, 1841, 5206)
    assert result.naive_bound == (11, 21, 41, 82, 163, 326, 651)
    assert sum(result.counts) == workflow.SHOTS
    assert math.fsum(result.exact) == pytest.approx(1.0, abs=1e-11, rel=0)
    assert (result.phases[0], result.phases[64], result.phases[-1]) == (
        -0.5,
        0.0,
        63 / 128,
    )
    assert result.lowest_peak == result.peaks[0]
    assert (
        abs(result.lowest_peak - result.model.full_ci)
        < result.model.normalization / 128
    )


def test_plateau_and_boundary_peaks() -> None:
    assert workflow.peak_indices((0.95, 0.1, 0.4, 0.4, 0.2, 0.3), 0.25) == (2,)
    assert workflow.peak_indices((0.0, 0.5, 0.0), 0.5) == (1,)
    assert not workflow.peak_indices((0.1, 0.2, 0.3), 0.05)


def test_all_source_stages() -> None:
    directory = Path(workflow.__file__).parent
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    stages = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(provenance["reviewed_code_cells"]) == 19
    assert all(cell["execution_count"] is None for cell in stages)
