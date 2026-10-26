"""Saved presence plans join existing public planning and canonical execution."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from examples.train_rydberg_controls import moving_pair_program

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "method", ["runge-kutta", "crank-nicolson", "krylov-exponential"]
)
def test_saved_moving_pair_trajectory_replays_without_replanning(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = moving_pair_program().measure_presence("a", at=0.5, efficiency=0.8)
    # A full-rank input lets every existing integrator meet its unchanged PSD
    # requirement here; pure-state RK witnesses remain in frozen consumers.
    state = source.basis_state(("g", "g"))
    initial = (
        0.9 * torch.outer(state, state.conj())
        + torch.eye(9, dtype=torch.complex128) / 90
    )
    plan = fql.plan(
        source,
        initial,
        torch.linspace(0, 1, 129, dtype=torch.float64),
        parameter_values=[{"omega": 1.2}] * 64 + [{"omega": 0.4}] * 64,
        outputs=fq.expectation(source.survival_operator(), name="survival"),
        options=fq.ExecutionOptions(shots=1, seed=17),
        method=method,
        return_density_matrices=True,
    )
    saved = plan.to_json()
    restored = fql.LindbladPlan.from_json(saved)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("saved trajectory must not replan or consult caller input")

    monkeypatch.setattr(fql, "plan", forbidden)
    direct, replay = fq.run(plan), fq.run(restored)
    assert direct.state is not None and direct.state.shape == (1, 9, 9)
    assert direct.expectation("survival").shape == (129,)
    assert direct.measurement("density_matrix").value.shape == (129, 9, 9)
    assert torch.equal(direct.state, replay.state)
    assert torch.equal(direct.expectation("survival"), replay.expectation("survival"))
    assert torch.equal(direct.measurement(-1).value, replay.measurement(-1).value)
    assert plan.to_json() == restored.to_json() == saved
    assert direct.runtime["distribution_semantics"] == "single_device_fast_path"
    assert direct.runtime["exact_plan_execution"] is True
    assert direct.runtime["scalability_claim_allowed"] is False
