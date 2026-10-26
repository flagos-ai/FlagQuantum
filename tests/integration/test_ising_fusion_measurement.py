"""Acceptance of the native fusion-observation user workflow."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from examples.ising_fusion_measurement import fusion_state, main
from flagquantum.operators import SuperOperator

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("initial,sign", [("1", -1), ("psi", 1)])
def test_braid_coherence_and_unread_charge_measurement(
    initial: Literal["1", "psi"], sign: int
) -> None:
    state = fusion_state(initial)
    # In the declared gauge this word gives |1> +/- |psi>, up to global phase.
    expected = torch.tensor([[1, sign], [sign, 1]], dtype=torch.complex128) / 2
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    projectors = [torch.diag(row.to(torch.complex128)) for row in torch.eye(2)]
    unread = SuperOperator.from_kraus(projectors).apply(state)
    torch.testing.assert_close(
        unread, torch.eye(2, dtype=state.dtype) / 2, atol=1e-12, rtol=0
    )
    assert abs(state[0, 1]) > 0.49
    assert torch.trace(unread).real.item() == pytest.approx(1, abs=1e-12)
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_user_workflow_checks_sampling_back_action_and_rng() -> None:
    before = torch.random.get_rng_state().clone()
    main()
    assert torch.equal(before, torch.random.get_rng_state())
