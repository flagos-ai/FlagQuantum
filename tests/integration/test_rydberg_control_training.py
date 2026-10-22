"""A native moving pair trains through existing parameter and optimizer APIs."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum.lindblad as fql
from examples.train_rydberg_controls import control_plan, train_controls

pytestmark = pytest.mark.integration


def dense_rr(amplitudes: torch.Tensor, intervals: int = 128) -> torch.Tensor:
    # Independent four-state unitary; equal g/r loss rates factor out of RR.
    drive = torch.tensor(
        [
            [0.0, complex(math.cos(0.3), math.sin(0.3))],
            [complex(math.cos(0.3), -math.sin(0.3)), 0.0],
        ],
        dtype=torch.complex128,
    )
    n = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.complex128))
    eye = torch.eye(2, dtype=torch.complex128)
    laser = 0.5 * (torch.kron(drive, eye) + torch.kron(eye, drive))
    detuning = -0.2 * (torch.kron(n, eye) + torch.kron(eye, n))
    pair = torch.kron(n, n)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)
    for k in range(intervals):
        t = (k + 0.5) / intervals
        h = (
            amplitudes[k // (intervals // 2)] * laser
            + detuning
            + 64 / (2 + 2 * t) ** 6 * pair
        )
        state = torch.matrix_exp(-1j * h / intervals) @ state
    return math.exp(-0.2) * state[3].abs().square()


def test_native_pair_control_gradient_matches_independent_frozen_unitary() -> None:
    amplitudes = torch.tensor([0.8, 1.2], dtype=torch.float64, requires_grad=True)
    actual = fql.run(control_plan(amplitudes)).expectation("rr")[-1]
    (gradient,) = torch.autograd.grad(actual, amplitudes)
    oracle = amplitudes.detach().requires_grad_(True)
    expected = dense_rr(oracle)
    (reference,) = torch.autograd.grad(expected, oracle)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    torch.testing.assert_close(gradient, reference, atol=1e-10, rtol=0)


def test_native_pair_optimization_rebuilds_plans_and_reduces_loss() -> None:
    amplitudes = torch.tensor([0.8, 1.2], dtype=torch.float64, requires_grad=True)
    initial = fql.run(control_plan(amplitudes)).expectation("rr")[-1].detach()
    train_controls(amplitudes)
    final = fql.run(control_plan(amplitudes)).expectation("rr")[-1].detach()
    assert (final - 0.12).square() < (initial - 0.12).square() / 100
    assert torch.isfinite(amplitudes).all()


def test_native_training_reports_conditioning_and_replays_detached_plan() -> None:
    plan = control_plan(torch.tensor([0.8, 1.2], dtype=torch.float64))
    result = fql.run(plan)
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    assert torch.equal(result.populations, replay.populations)
    survival, no_flag, rr = (
        result.expectation(n) for n in ("survival", "no_flag", "rr")
    )
    t = result.times
    torch.testing.assert_close(survival, torch.exp(-0.2 * t), atol=1e-10, rtol=0)
    torch.testing.assert_close(
        no_flag, (1 - 0.8 * (1 - torch.exp(-0.1 * t))).square(), atol=1e-10, rtol=0
    )
    assert 0 < rr[-1] < rr[-1] / no_flag[-1] < rr[-1] / survival[-1] < 1
