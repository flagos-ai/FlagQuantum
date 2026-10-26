"""Noncommuting pulse order, leakage, integration order and control derivatives."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

import flagquantum.lindblad as fql
from examples.driven_three_level import (
    initial_controls,
    main,
    pulse_plan,
    reference_states,
)

pytestmark = pytest.mark.integration


def test_complex_density_matches_independent_ordered_unitaries() -> None:
    controls = initial_controls()
    result = fql.run(pulse_plan(controls))
    assert result.density_matrices is not None
    torch.testing.assert_close(
        result.density_matrices, reference_states(controls), atol=1e-10, rtol=0
    )
    assert result.expectation("leakage")[-1] > 0.001
    total = torch.stack(
        [result.expectation(n) for n in ("ground", "excited", "leakage")]
    ).sum(0)
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-10, rtol=0)
    reversed_state = reference_states(controls.flip(0))[-1]
    assert float(abs(reversed_state - result.density_matrices[-1]).max()) > 0.01


def test_each_pulse_gradient_matches_independent_finite_difference() -> None:
    controls = initial_controls().requires_grad_()
    result = fql.run(pulse_plan(controls))
    (gradient,) = torch.autograd.grad(result.expectation("leakage")[-1], controls)
    step = 1e-4
    for pulse in range(4):
        for quadrature in range(2):
            values = []
            for multiple in (-2, -1, 1, 2):
                shifted = controls.detach().clone()
                shifted[pulse, quadrature] += multiple * step
                values.append(reference_states(shifted)[-1, 2, 2].real)
            expected = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (
                12 * step
            )
            assert gradient[pulse, quadrature].item() == pytest.approx(
                expected.item(), abs=1e-9
            )


def test_refinement_preserves_pulse_boundaries_and_fourth_order_convergence() -> None:
    controls = initial_controls()
    reference = reference_states(controls)
    errors = []
    for subdivisions in (1, 2, 4, 8):
        result = fql.run(
            pulse_plan(controls, subdivisions=subdivisions, method="runge-kutta")
        )
        assert result.density_matrices is not None
        boundaries = result.density_matrices[::subdivisions]
        errors.append(float(abs(boundaries - reference).max()))
    assert all(a / b > 10 for a, b in pairwise(errors))
    assert errors[-1] < 1e-6


def test_training_and_numeric_replay_user_workflow() -> None:
    main()
