"""Verify both Fourier states and the complete finite reset channel."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.comparisons.circuit import (
    kraus_operators,
    reference_density,
)
from examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.overview.model import (
    DELTA,
    DTYPE,
    ENERGIES,
    ROUNDS,
    gaussian,
    gibbs_probabilities,
    glauber_angles,
)
from examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.overview.quantum import (
    build_probe,
    build_thermal,
)
from examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.physics import (
    ANGLES,
    GAUSSIAN,
)
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("gaussian_window", (False, True))
def test_full_fourier_state(gaussian_window: bool) -> None:
    program = build_probe(gaussian_window=gaussian_window)
    actual = fq.run(program).statevector().reshape(4, 64)
    window = torch.tensor(
        gaussian(6) if gaussian_window else (1 / 8,) * 64, dtype=DTYPE
    )
    elapsed = math.pi / 5 * torch.arange(64, dtype=torch.float64)
    shift = torch.tensor(
        [value - ENERGIES[0] for value in ENERGIES], dtype=torch.float64
    )
    before_fft = window * torch.exp(1j * shift[:, None] * elapsed)
    expected = torch.fft.fft(before_fft, dim=1, norm="ortho") / 2
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_fresh_rates_and_complete_thermal_density() -> None:
    assert gaussian(4) == pytest.approx(GAUSSIAN, abs=1e-14, rel=0)
    assert glauber_angles() == pytest.approx(ANGLES, abs=1e-14, rel=0)
    program = build_thermal()
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 10 and resources.channel_count == 400
    output = fq.run(program)
    kraus = kraus_operators(DELTA)
    torch.testing.assert_close(
        (kraus.mH @ kraus).sum(dim=0), torch.eye(4, dtype=DTYPE), atol=1e-11, rtol=0
    )
    reference = reference_density(
        kraus, ROUNDS, torch.tensor([1, 0, 0, 0], dtype=DTYPE)
    )
    actual = output.state
    assert isinstance(actual, torch.Tensor)
    full = torch.zeros((1024, 1024), dtype=DTYPE)
    full[::256, ::256] = reference
    torch.testing.assert_close(actual.reshape(1024, 1024), full, atol=1e-10, rtol=0)
    marginal = program.probabilities(output, registers=("state",))
    assert all(
        abs(float(marginal[(word,)]) - weight) < 0.025
        for word, weight in enumerate(gibbs_probabilities())
    )


def test_complete_unexecuted_notebook() -> None:
    from examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.overview import (
        __file__ as path,
    )

    assert path is not None
    root = Path(path).parent
    book = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 16
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
