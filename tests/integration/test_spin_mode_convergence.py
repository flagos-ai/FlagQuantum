"""Spin-mode population, coherence, cutoff and training acceptance."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum.lindblad as fql
from examples.spin_mode_convergence import (
    FINAL_TIME,
    THERMAL_RATIO,
    main,
    reference_density,
    spin_mode_plan,
    thermal_excited_probability,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("cutoff", [3, 5, 7])
def test_full_density_conservation_and_numeric_replay(cutoff: int) -> None:
    plan = spin_mode_plan(cutoff, torch.tensor(0.6, dtype=torch.float64))
    result = fql.run(plan)
    assert result.density_matrices is not None
    for index, time in enumerate((0.0, FINAL_TIME / 2, FINAL_TIME)):
        expected = reference_density(cutoff, 0.6, time)
        torch.testing.assert_close(
            result.density_matrices[index], expected, atol=1e-10, rtol=0
        )
    assert result.density_matrices[-1, cutoff, 1].imag > 0.2
    total = result.expectation("total")
    torch.testing.assert_close(total, total[0].expand_as(total), atol=1e-10, rtol=0)
    # A finite mode incorrectly traps the highest excited state; retain that error.
    assert result.populations[-1, -1].item() == pytest.approx(
        (1 - THERMAL_RATIO)
        * THERMAL_RATIO ** (cutoff - 1)
        / (1 - THERMAL_RATIO**cutoff),
        abs=1e-10,
    )
    restored = fql.LindbladPlan.from_json(plan.to_json())
    assert restored.dimensions == (2, cutoff)
    replay = fql.run(restored)
    assert replay.density_matrices is not None
    assert torch.equal(replay.density_matrices, result.density_matrices)


def test_thermal_truncation_converges_and_gradient_matches_closed_blocks() -> None:
    errors = []
    for cutoff in (3, 5, 7):
        coupling = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
        value = fql.run(spin_mode_plan(cutoff, coupling)).expectation("excited")[-1]
        (gradient,) = torch.autograd.grad(value, coupling)
        derivative = sum(
            -(1 - THERMAL_RATIO)
            * THERMAL_RATIO**n
            / (1 - THERMAL_RATIO**cutoff)
            * FINAL_TIME
            * math.sqrt(n + 1)
            * math.sin(2 * 0.6 * math.sqrt(n + 1) * FINAL_TIME)
            for n in range(cutoff - 1)
        )
        assert gradient.item() == pytest.approx(derivative, abs=1e-9)
        errors.append(abs(value.item() - thermal_excited_probability(0.6, FINAL_TIME)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 5e-5


def test_user_workflow_trains_and_replays() -> None:
    main()
