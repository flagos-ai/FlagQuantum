"""The density-matrix helpers must refuse a wire the Hilbert space lacks.

Every function in this module addresses wires by indexing an axis of size
``2**n_wires``. A negative index is a valid Python index, so ``-1`` silently
addressed the last wire and ``apply_unitary_density`` returned the same density
matrix as the wire the caller did not name; an index past the last wire raised a
bare ``IndexError`` instead of naming the offending wire.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import flagquantum.simulation.density_matrix as density_matrix_module

pytestmark = pytest.mark.unit

N_WIRES = 3
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
OUT_OF_RANGE = (3, 9, -1, -2)

Operation = Callable[[int], object]


def _ground_state() -> torch.Tensor:
    rho = torch.zeros(2**N_WIRES, 2**N_WIRES, dtype=torch.complex64)
    rho[0, 0] = 1
    return rho


def _expand_operator(wire: int) -> object:
    return density_matrix_module.expand_operator(X, [wire], N_WIRES)


def _apply_unitary_density(wire: int) -> object:
    return density_matrix_module.apply_unitary_density(
        _ground_state(), X, [wire], N_WIRES
    )


def _apply_kraus_density(wire: int) -> object:
    return density_matrix_module.apply_kraus_density(
        _ground_state(), [X], [wire], N_WIRES
    )


def _expectation_z_density(wire: int) -> object:
    return density_matrix_module.expectation_z_density(_ground_state(), [wire])


OPERATIONS: tuple[tuple[str, Operation], ...] = (
    ("expand_operator", _expand_operator),
    ("apply_unitary_density", _apply_unitary_density),
    ("apply_kraus_density", _apply_kraus_density),
    ("expectation_z_density", _expectation_z_density),
)


@pytest.mark.parametrize("wire", OUT_OF_RANGE)
@pytest.mark.parametrize(
    ("operation", "run"), OPERATIONS, ids=[name for name, _ in OPERATIONS]
)
def test_wire_outside_the_hilbert_space_is_refused(
    operation: str, run: Operation, wire: int
) -> None:
    with pytest.raises(ValueError, match="wire index out of range"):
        run(wire)


def test_a_negative_wire_no_longer_reaches_the_last_wire() -> None:
    """`wire=-1` used to give the result for wire ``N_WIRES - 1``.

    The two calls produced byte-identical density matrices, so the substitution
    was invisible in the result. This asserts the refusal, and that a refusal
    leaves no output to be mistaken for the requested one.
    """

    requested_wire = N_WIRES - 1
    legitimate = _apply_unitary_density(requested_wire)

    with pytest.raises(ValueError, match="wire index out of range"):
        _apply_unitary_density(-1)

    # The legitimate call is unaffected by the refused one.
    assert torch.equal(legitimate, _apply_unitary_density(requested_wire))


def test_in_range_wires_still_expand_and_apply() -> None:
    """The guard must not reject a wire the Hilbert space does have."""

    for wire in range(N_WIRES):
        expanded = density_matrix_module.expand_operator(X, [wire], N_WIRES)
        assert expanded.shape == (2**N_WIRES, 2**N_WIRES)

        applied = density_matrix_module.apply_unitary_density(
            _ground_state(), X, [wire], N_WIRES
        )
        assert applied.shape == (1, 2**N_WIRES, 2**N_WIRES)
        # X is unitary, so the trace must survive.
        assert torch.allclose(applied[0].trace().real, torch.tensor(1.0), atol=1e-6)

    batched = density_matrix_module.expand_operator(torch.stack([X, X]), [0], N_WIRES)
    assert batched.shape == (2, 2**N_WIRES, 2**N_WIRES)


def test_two_wire_operator_still_expands() -> None:
    """A multi-wire operator keeps working; only invalid wires are refused."""

    cx = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=torch.complex64,
    )
    assert density_matrix_module.expand_operator(cx, [0, 1], N_WIRES).shape == (8, 8)
    with pytest.raises(ValueError, match="wire index out of range"):
        density_matrix_module.expand_operator(cx, [0, 1], 1)
