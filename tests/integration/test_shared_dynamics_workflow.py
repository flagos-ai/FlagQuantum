"""Two physical models share outputs, batch axes, gradients and numeric replay."""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from examples.shared_dynamics_workflow import main, models

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("model_name", ["three-level", "spin-mode"])
def test_two_models_preserve_the_same_workflow(
    model_name: str, batch_size: int
) -> None:
    _, dimensions, transition, initial, observable = next(
        model for model in models() if model[0] == model_name
    )
    coupling = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    times = torch.linspace(0, 0.5, 51, dtype=torch.float64)
    plan = fql.plan(
        coupling * transition,
        initial[:batch_size],
        times,
        dimensions=dimensions,
        outputs=fq.expectation(observable, name="transfer"),
        options=fq.ExecutionOptions(batch_size=batch_size, require_gradients=True),
    )
    result = fql.run(plan)
    actual = result.expectation("transfer")
    assert actual.shape == (51, batch_size)
    # Both active states form an exact two-state Rabi block; the vacuum is dark.
    expected = torch.zeros((51, batch_size), dtype=torch.float64)
    expected[:, 0] = torch.sin(0.7 * times).square()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    (gradient,) = torch.autograd.grad(actual[-1].sum(), coupling)
    assert gradient.item() == pytest.approx(0.5 * math.sin(0.7), abs=1e-10)
    assert result.plan is plan
    assert torch.equal(result.expectation(0), actual)

    numeric = fql.plan(
        coupling.detach() * transition,
        initial[:batch_size],
        times,
        dimensions=dimensions,
        outputs=fq.expectation(observable, name="transfer"),
        options=fq.ExecutionOptions(batch_size=batch_size),
    )
    restored = fql.LindbladPlan.from_json(numeric.to_json())
    assert restored.to_json() == numeric.to_json()
    assert restored.dimensions == dimensions
    replay = fql.run(restored).expectation("transfer")
    assert not replay.requires_grad
    torch.testing.assert_close(replay, actual.detach(), atol=0, rtol=0)
    with pytest.raises(TypeError, match="semantic overrides"):
        fql.run(restored, outputs=fq.expectation(observable))


def test_user_example_trains_both_models(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    output = capsys.readouterr().out
    for name in ("three-level", "spin-mode"):
        assert f"{name}: axes=(51, 2)" in output
    assert output.count("Initial transfer: 0.11757891") == 2
    assert output.count("Coupling derivative: 0.32210884") == 2
    assert output.count("Loss decreased: True") == 2
    assert output.count("Exact numeric replay: True") == 2
