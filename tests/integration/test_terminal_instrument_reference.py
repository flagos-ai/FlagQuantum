"""Independent physics oracles for two current-API instrument references."""

from __future__ import annotations

import math

import pytest
import torch

from examples.terminal_instrument_reference import fock_reference, presence_reference
from flagquantum.operators import SuperOperator

pytestmark = pytest.mark.integration


def test_presence_instrument_is_complete_and_preserves_the_declared_backaction() -> (
    None
):
    rho, branches = presence_reference()
    quiet, flag = (SuperOperator.from_kraus(k).apply(rho) for k in branches.values())
    effect = sum(k.mH @ k for operators in branches.values() for k in operators)
    torch.testing.assert_close(
        effect, torch.eye(3, dtype=rho.dtype), atol=1e-10, rtol=0
    )
    assert torch.trace(quiet).real.item() == pytest.approx(0.68, abs=1e-10)
    assert torch.trace(flag).real.item() == pytest.approx(0.32, abs=1e-10)
    assert (quiet[0, 0].real / 0.68).item() == pytest.approx(15 / 17, abs=1e-10)
    assert (quiet + flag)[0, 2].real.item() == pytest.approx(
        math.sqrt(0.048), abs=1e-10
    )
    assert not torch.allclose(quiet + flag, rho, atol=1e-10, rtol=0)


def test_finite_fock_instrument_has_number_statistics_and_a_dephasing_channel() -> None:
    rho, branches = fock_reference()
    states = [SuperOperator.from_kraus(k).apply(rho) for k in branches.values()]
    actual = torch.stack([torch.trace(s).real for s in states])
    torch.testing.assert_close(
        actual, torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64), atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        sum(states), torch.diag(torch.diagonal(rho)), atol=1e-10, rtol=0
    )
