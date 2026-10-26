"""Independent density and weighted-branch references for the unread channel."""

from __future__ import annotations

import cmath
import math
import random
from typing import Literal

import pytest
import torch

from flagquantum.core.ising import IsingBraidProgram
from flagquantum.simulation.ising_braid import evolve_ising_braid

pytestmark = pytest.mark.unit
Word = tuple[tuple[int, bool], ...]
Channel = Literal["1", "psi"]
DTYPE = torch.complex128


def _continue(state: torch.Tensor, word: Word) -> torch.Tensor:
    # Closed-form reference, independent of production F/R and Kraus assembly.
    outer = torch.diag(
        torch.tensor(
            [cmath.exp(-1j * math.pi / 8), cmath.exp(3j * math.pi / 8)], dtype=DTYPE
        )
    )
    inner = torch.tensor([[1, -1j], [-1j, 1]], dtype=DTYPE) * (
        cmath.exp(1j * math.pi / 8) / math.sqrt(2)
    )
    for left, clockwise in word:
        gate = inner if left == 1 else outer
        gate = gate.mH if clockwise else gate
        state = gate @ state @ gate.mH
    return state


@pytest.mark.parametrize("initial", ["1", "psi"])
@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ((), ()),
        (((0, False),), ((2, True),)),
        (((1, False),), ()),
        (((1, False),), ((1, True),)),
        (((1, False), (0, True)), ((1, False), (2, False))),
        (((2, True), (1, False)), ((0, False), (1, True))),
    ],
)
def test_full_density_matches_unread_and_weighted_conditional_continuations(
    initial: Channel, prefix: Word, suffix: Word
) -> None:
    rng = torch.random.get_rng_state().clone()
    python_rng = random.getstate()
    source = IsingBraidProgram(initial_channel=initial)
    for left, clockwise in prefix:
        source.exchange(left, clockwise=clockwise)
    source.dephase_pair()
    for left, clockwise in suffix:
        source.exchange(left, clockwise=clockwise)
    actual = evolve_ising_braid(source.to_ir())
    initial_density = torch.diag(
        torch.tensor([initial == "1", initial == "psi"], dtype=DTYPE)
    )
    before = _continue(initial_density, prefix)
    unread = torch.diag(torch.diag(before))
    expected = _continue(unread, suffix)
    weighted = torch.zeros((2, 2), dtype=DTYPE)
    for index in range(2):
        branch = torch.zeros_like(before)
        branch[index, index] = before[index, index]
        probability = torch.trace(branch).real
        if probability > 0:
            weighted += probability * _continue(branch / probability, suffix)
        else:
            assert torch.count_nonzero(branch) == 0
    assert actual.shape == (2, 2) and actual.dtype == DTYPE
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(actual, weighted, atol=1e-12, rtol=0)
    torch.testing.assert_close(actual, actual.mH, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.trace(actual), torch.tensor(1, dtype=DTYPE), atol=1e-12, rtol=0
    )
    assert torch.linalg.eigvalsh(actual).min() >= -1e-12
    if not suffix:
        assert actual[0, 1] == actual[1, 0] == 0
        torch.testing.assert_close(
            actual.diagonal(), before.diagonal(), atol=1e-12, rtol=0
        )
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert random.getstate() == python_rng


@pytest.mark.parametrize("initial", ["1", "psi"])
def test_unread_echo_is_mixed_while_braid_only_echo_stays_pure(
    initial: Channel,
) -> None:
    observed = IsingBraidProgram(initial_channel=initial)
    observed.exchange(1).dephase_pair().exchange(1, clockwise=True)
    control = IsingBraidProgram(initial_channel=initial)
    control.exchange(1).exchange(1, clockwise=True)
    mixed = evolve_ising_braid(observed.to_ir())
    pure = evolve_ising_braid(control.to_ir())
    torch.testing.assert_close(mixed, torch.eye(2, dtype=DTYPE) / 2, atol=1e-12, rtol=0)
    torch.testing.assert_close(pure, control.projector(initial), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.trace(mixed @ mixed), torch.tensor(0.5, dtype=DTYPE), atol=1e-12, rtol=0
    )
