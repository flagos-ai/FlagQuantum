"""Independent loss, phase and detection references before native atom APIs."""

from __future__ import annotations

import pytest
import torch

import flagquantum.lindblad as fql
from examples.neutral_atom_pair_reference import reference_plan

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("loss_rate", [0.0, 0.2])
@pytest.mark.parametrize("efficiency", [0.0, 0.8, 1.0])
def test_loss_is_not_the_same_event_as_detection(
    loss_rate: float, efficiency: float
) -> None:
    plan = reference_plan(loss_rate=loss_rate, detection_efficiency=efficiency)
    result = fql.run(plan)
    times = torch.tensor([0.0, 0.25, 0.5], dtype=torch.float64)
    one_atom_survival = torch.exp(-loss_rate * times)
    survival = one_atom_survival.square()
    no_flag = (1 - efficiency * (1 - one_atom_survival)).square()
    phase = torch.tensor([0.0, 0.25, 0.25 + 0.25 / 64], dtype=torch.float64)
    for name, expected in (
        ("survival", survival),
        ("rr", 0.5 * survival),
        ("coherence", survival * torch.cos(phase)),
        ("no_flag", no_flag),
    ):
        torch.testing.assert_close(
            result.expectation(name), expected, atol=1e-10, rtol=0
        )
    torch.testing.assert_close(
        result.populations.sum(dim=-1), torch.ones_like(times), atol=1e-10, rtol=0
    )
    conditional = result.expectation("rr") / result.expectation("survival")
    torch.testing.assert_close(
        conditional, torch.full_like(times, 0.5), atol=1e-10, rtol=0
    )
    if loss_rate > 0 and efficiency < 1:
        assert (result.expectation("rr") / result.expectation("no_flag"))[-1] < 0.5
    restored = fql.LindbladPlan.from_json(plan.to_json())
    replay = fql.run(restored)
    for name in ("survival", "rr", "coherence", "no_flag"):
        assert torch.equal(replay.expectation(name), result.expectation(name))


def test_geometry_changes_phase_without_relabelling_loss_as_postselection() -> None:
    near = fql.run(reference_plan(distances=(2.0, 2.0)))
    far = fql.run(reference_plan(distances=(4.0, 4.0)))
    assert (
        abs(near.expectation("coherence")[-1] - far.expectation("coherence")[-1]) > 0.05
    )
    torch.testing.assert_close(
        near.expectation("survival"), far.expectation("survival"), atol=1e-10, rtol=0
    )


def test_reference_refuses_coincident_atoms() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        reference_plan(distances=(2.0, 0.0))
