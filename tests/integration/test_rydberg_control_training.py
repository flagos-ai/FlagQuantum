"""A native moving pair trains through existing parameter and optimizer APIs."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum.lindblad as fql
from examples.train_rydberg_controls import control_plan, train_controls
from flagquantum.operators import SuperOperator
from flagquantum.runtime import sample_terminal

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

    # Training stays analytic. Only the fresh detached export retains states
    # for observed terminal detection; no sampling estimator enters autograd.
    plan = control_plan(amplitudes.detach(), return_density_matrices=True)
    encoded = plan.to_json()
    evolution = fql.run(plan)
    replay = fql.run(fql.LindbladPlan.from_json(encoded))
    assert evolution.density_matrices is not None
    assert replay.density_matrices is not None
    assert not evolution.density_matrices.requires_grad
    quiet = torch.diag(torch.tensor([1.0, 1.0, math.sqrt(0.2)], dtype=torch.complex128))
    flag = torch.diag(torch.tensor([0.0, 0.0, math.sqrt(0.8)], dtype=torch.complex128))
    branches = {
        f"{left}/{right}": (torch.kron(a, b),)
        for left, a in (("no_flag", quiet), ("flag", flag))
        for right, b in (("no_flag", quiet), ("flag", flag))
    }
    rng = torch.get_rng_state()
    observed = sample_terminal(
        evolution.density_matrices[-1], branches, shots=4096, seed=7
    )
    repeated = sample_terminal(
        replay.density_matrices[-1], branches, shots=4096, seed=7
    )
    assert isinstance(observed.value, torch.Tensor)
    assert isinstance(repeated.value, torch.Tensor)
    assert torch.equal(observed.value, repeated.value)
    assert torch.equal(rng, torch.get_rng_state())

    # Equal g/r loss rates give independent exponential survival, even after
    # optimizing the interacting drive. These are observed flags, not lost truth.
    flag_probability = 0.8 * (1 - math.exp(-0.1))
    expected = torch.tensor(
        [
            (1 - flag_probability) ** 2,
            flag_probability * (1 - flag_probability),
            flag_probability * (1 - flag_probability),
            flag_probability**2,
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor(observed.metadata["probabilities"], dtype=torch.float64)
    torch.testing.assert_close(weights, expected, atol=1e-10, rtol=0)
    frequencies = torch.bincount(observed.value, minlength=4).to(torch.float64) / 4096
    standard_errors = torch.sqrt(expected * (1 - expected) / 4096)
    assert bool(((frequencies - expected).abs() < 6 * standard_errors).all())
    assert weights[0] > evolution.expectation("survival")[-1] + 1e-3
    conditional = SuperOperator.from_kraus(branches["no_flag/no_flag"]).apply(
        evolution.density_matrices[-1]
    )
    conditioned_rr = conditional[4, 4].real / torch.trace(conditional).real
    torch.testing.assert_close(
        conditioned_rr,
        evolution.expectation("rr")[-1] / evolution.expectation("no_flag")[-1],
        atol=1e-10,
        rtol=0,
    )
    assert encoded == plan.to_json()
    assert plan.to_dict()["version"] == "1.7"


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
