"""Preserve exact-amplification sampling and the source's full Grover plot data."""

import math
from pathlib import Path

import pytest
import torch

from examples.synthesis.functions.amplitude_amplification.exact_amplitude_amplification.comparisons.circuit import (
    run_example as compare_circuit,
)
from examples.synthesis.functions.amplitude_amplification.exact_amplitude_amplification.main import (
    plot_trajectories,
    sample_counts,
    trajectories,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("seed", [0, 17, 71])
def test_executed_sampling_succeeds_and_explicit_auxiliary_is_zero(seed: int) -> None:
    assert sample_counts(1000, seed) == {(1, 0): 1000}


def test_quantum_program_retains_raw_phase_and_clean_hidden_workspace() -> None:
    compare_circuit()


def test_full_classical_trajectories_match_independent_rotation_products() -> None:
    standard, tuned = trajectories()
    theta = math.asin(math.sqrt(0.07))
    rounds = math.ceil(math.pi / (4 * theta) - 0.5)
    assert rounds == 3
    assert len(standard) == len(tuned) == rounds + 1
    for states, angle in ((standard, theta), (tuned, math.pi / 14)):
        initial = torch.tensor([math.sin(angle), math.cos(angle)], dtype=torch.float64)
        for step, actual in enumerate(states):
            rotation = torch.tensor(
                [
                    [math.cos(2 * step * angle), math.sin(2 * step * angle)],
                    [-math.sin(2 * step * angle), math.cos(2 * step * angle)],
                ],
                dtype=torch.float64,
            )
            torch.testing.assert_close(
                torch.tensor(actual, dtype=torch.float64),
                rotation @ initial,
                atol=1e-14,
                rtol=0,
            )
            assert math.hypot(*actual) == pytest.approx(1, abs=1e-14)
    assert tuned[-1] == pytest.approx((1, 0), abs=1e-14)
    assert standard[-1][1] < -0.2  # The untuned trajectory overshoots.


def test_notebook_plot_is_written_and_figures_are_closed(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    output = tmp_path / "trajectories.png"
    before = plt.get_fignums()
    plot_trajectories(output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000
    assert plt.get_fignums() == before


@pytest.mark.parametrize("shots", [0, -1, True])
def test_sampling_rejects_invalid_shot_counts(shots: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        sample_counts(shots)
