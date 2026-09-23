"""The MPS apply path must refuse a wire the state does not have.

`apply_one` indexed the site list directly, so a negative wire selected a
different site and the gate landed on the wrong wire with no error, while an
index past the last site raised a raw `IndexError` instead of the `ValueError`
this class already uses for an out-of-range observable wire.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
CX = torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=torch.complex64,
)
OUT_OF_RANGE = (3, 9, -1, -2)

Apply = Callable[[int], None]


def _apply_one(wire: int) -> None:
    MPSState.zero(3).apply_one(X, wire=wire)


def _apply_parametric_one(wire: int) -> None:
    MPSState.zero(3).apply_parametric_one("rx", 0.5, wire=wire)


def _apply_two(wire: int) -> None:
    MPSState.zero(3).apply_two(CX, wire)


def _apply_swap(wire: int) -> None:
    MPSState.zero(3).apply_swap(left_wire=wire)


def _apply_two_remote(wire: int) -> None:
    MPSState.zero(3).apply_two_remote(CX, [wire, 0])


OPERATIONS: tuple[tuple[str, Apply], ...] = (
    ("apply_one", _apply_one),
    ("apply_parametric_one", _apply_parametric_one),
    ("apply_two", _apply_two),
    ("apply_swap", _apply_swap),
    ("apply_two_remote", _apply_two_remote),
)


@pytest.mark.parametrize("wire", OUT_OF_RANGE)
@pytest.mark.parametrize(
    ("operation", "apply"), OPERATIONS, ids=[name for name, _ in OPERATIONS]
)
def test_apply_rejects_a_wire_outside_the_state(
    operation: str, apply: Apply, wire: int
) -> None:
    with pytest.raises(ValueError, match="wire index out of range"):
        apply(wire)


def test_a_negative_wire_no_longer_reaches_a_different_site() -> None:
    """The pre-change path applied the gate to the last wire for ``wire=-1``.

    Both calls used to leave the state well formed, so the substitution was
    invisible in the result. Asserting the refusal is what makes it visible.
    """
    state = MPSState.zero(3)
    with pytest.raises(ValueError, match="wire index out of range"):
        state.apply_one(X, wire=-1)

    # The state must be untouched by the refused call.
    assert state.expectation_z().tolist() == [[1.0, 1.0, 1.0]]


def test_the_last_wire_has_no_right_neighbour() -> None:
    """A two-site gate on the final wire is out of range, not an IndexError."""

    state = MPSState.zero(3)
    with pytest.raises(ValueError, match="wire index out of range"):
        state.apply_two(CX, 2)


def test_in_range_wires_still_apply() -> None:
    """The guard must not reject a wire the state does have."""

    single = MPSState.zero(3)
    single.apply_one(X, wire=2)
    assert single.expectation_z().tolist() == [[1.0, 1.0, -1.0]]

    parametric = MPSState.zero(3)
    parametric.apply_parametric_one("rx", 0.5, wire=0)
    assert parametric.n_wires == 3

    for n_wires in (2, 3, 4):
        for left_wire in range(n_wires - 1):
            two_site = MPSState.zero(n_wires)
            two_site.apply_two(CX, left_wire)
            assert two_site.expectation_z().shape[-1] == n_wires

        remote = MPSState.zero(n_wires)
        remote.apply_two_remote(CX, [0, n_wires - 1])
        assert remote.n_wires == n_wires
