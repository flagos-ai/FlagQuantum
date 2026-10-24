"""Independent conditional-density references for sampled Ising actions."""

from __future__ import annotations

import cmath
import math
import random
from typing import Any, Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.errors import CapabilityError, ExecutionError
from flagquantum.simulation.ising_braid import (
    _continue_ising_sampled,
    _ising_sampled_prefix,
    evolve_ising_braid,
)

pytestmark = pytest.mark.unit
Word = tuple[tuple[int, bool], ...]
Channel = Literal["1", "psi"]
DTYPE = torch.complex128


def _reference(state: torch.Tensor, word: Word) -> torch.Tensor:
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


def _source(
    initial: Channel, prefix: Word, suffix: Word, *, unread: bool = False
) -> IsingBraidProgram:
    source = IsingBraidProgram(initial_channel=initial)
    for left, clockwise in prefix:
        source.exchange(left, clockwise=clockwise)
    if unread:
        source.dephase_pair()
    else:
        source.measure_pair()
    for left, clockwise in suffix:
        source.exchange(left, clockwise=clockwise)
    return source


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
def test_conditional_density_and_weighted_unread_reference(
    initial: Channel, prefix: Word, suffix: Word
) -> None:
    source = _source(initial, prefix, suffix)
    record = source.to_ir()
    torch_rng = torch.random.get_rng_state().clone()
    python_rng = random.getstate()
    before = _ising_sampled_prefix(record)
    snapshot = before.clone()
    initial_density = torch.diag(
        torch.tensor([initial == "1", initial == "psi"], dtype=DTYPE)
    )
    expected_before = _reference(initial_density, prefix)
    torch.testing.assert_close(before, expected_before, atol=1e-12, rtol=0)
    weighted = torch.zeros_like(before)
    for index in range(2):
        probability = before[index, index].real
        if probability == 0:
            with pytest.raises(ExecutionError, match="positive"):
                _continue_ising_sampled(record, before, index)
            continue
        actual = _continue_ising_sampled(record, before, index)
        selected = torch.zeros_like(before)
        selected[index, index] = 1
        expected = _reference(selected, suffix)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(actual, actual.mH, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            torch.trace(actual), torch.tensor(1, dtype=DTYPE), atol=1e-12, rtol=0
        )
        torch.testing.assert_close(
            torch.trace(actual @ actual),
            torch.tensor(1, dtype=DTYPE),
            atol=1e-12,
            rtol=0,
        )
        assert torch.linalg.eigvalsh(actual).min() >= -1e-12
        assert (
            actual.shape == (2, 2)
            and actual.dtype == DTYPE
            and actual.device.type == "cpu"
        )
        weighted += probability * actual
    torch.testing.assert_close(
        weighted,
        _reference(torch.diag(expected_before.diagonal()), suffix),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        weighted,
        evolve_ising_braid(_source(initial, prefix, suffix, unread=True).to_ir()),
        atol=1e-12,
        rtol=0,
    )
    assert torch.equal(before, snapshot)
    assert source.to_ir() == record
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng


@pytest.mark.parametrize(
    "index", [-1, 2, True, False, 0.0, "psi", None, torch.tensor(0)]
)
def test_invalid_selected_index_fails_closed(index: Any) -> None:
    record = IsingBraidProgram().exchange(1).measure_pair().to_ir()
    with pytest.raises(ExecutionError):
        _continue_ising_sampled(record, _ising_sampled_prefix(record), index)


@pytest.mark.parametrize(
    "density",
    [
        torch.eye(3, dtype=DTYPE) / 3,
        torch.eye(2, dtype=torch.complex64) / 2,
        torch.eye(2, dtype=torch.float64) / 2,
        (torch.eye(2, dtype=DTYPE) / 2).requires_grad_(),
        torch.full((2, 2), complex(float("nan"), 0), dtype=DTYPE),
        torch.full((2, 2), complex(float("inf"), 0), dtype=DTYPE),
        torch.tensor([[0.5, 0.2], [0, 0.5]], dtype=DTYPE),
        torch.diag(torch.tensor([-0.1, 1.1], dtype=DTYPE)),
        torch.eye(2, dtype=DTYPE),
        torch.eye(2, dtype=DTYPE).unsqueeze(0) / 2,
        torch.empty((2, 2), dtype=DTYPE, device="meta"),
    ],
)
def test_invalid_prefix_density_is_rejected(density: torch.Tensor) -> None:
    with pytest.raises(ExecutionError):
        _continue_ising_sampled(IsingBraidProgram().measure_pair().to_ir(), density, 0)


@pytest.mark.parametrize("probabilities", [(0.4 + 5e-12, 0.6), (1e-15, 1 - 1e-15)])
@pytest.mark.parametrize("index", [0, 1])
def test_original_positive_trace_normalizes_selected_branch(
    probabilities: tuple[float, float], index: int
) -> None:
    record = IsingBraidProgram().measure_pair().exchange(1).to_ir()
    density = torch.diag(torch.tensor(probabilities, dtype=DTYPE))
    expected = torch.zeros_like(density)
    expected[index, index] = 1
    actual = _continue_ising_sampled(record, density, index)
    torch.testing.assert_close(
        actual, _reference(expected, ((1, False),)), atol=1e-12, rtol=0
    )


def test_full_kernel_and_root_admission_remain_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> torch.Tensor:
        pytest.fail("sampled source must be rejected before exchange arithmetic")

    source = IsingBraidProgram().exchange(1).measure_pair()
    monkeypatch.setattr(
        "flagquantum.simulation.ising_braid._exchange_unitary", unexpected
    )
    with pytest.raises(CapabilityError, match="selected"):
        evolve_ising_braid(source.to_ir())
    with pytest.raises(CapabilityError):
        fq.plan(source, options=fq.ExecutionOptions(shots=1, seed=17))


@pytest.mark.parametrize("unread", [False, True])
def test_missing_sampled_boundary_is_not_inferred(unread: bool) -> None:
    source = IsingBraidProgram().exchange(1)
    if unread:
        source.dephase_pair()
    with pytest.raises(CapabilityError, match="sampled"):
        _ising_sampled_prefix(source.to_ir())
    with pytest.raises(CapabilityError, match="sampled"):
        _continue_ising_sampled(source.to_ir(), torch.eye(2, dtype=DTYPE) / 2, 0)
