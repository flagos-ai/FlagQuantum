"""The density-matrix helpers must refuse an operator that cannot match its wires.

`expand_operator` reads ``matrix[sub_row, sub_col]`` with both indices drawn from
``2 ** len(wires)``, so it only ever consumed the top-left ``2**k x 2**k`` block
of whatever it was given. A 4x4 operator passed for a single wire was therefore
accepted and its three other quadrants discarded without a word; a 2x2 operator
passed for two wires raised a bare ``IndexError`` from inside the indexing
expression.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import flagquantum.simulation.density_matrix as density_matrix_module

pytestmark = pytest.mark.unit

N_WIRES = 2
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
CX = torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=torch.complex64,
)

Operation = Callable[[torch.Tensor, tuple[int, ...]], object]


def _expand_operator(matrix: torch.Tensor, wires: tuple[int, ...]) -> object:
    return density_matrix_module.expand_operator(matrix, list(wires), N_WIRES)


def _apply_unitary_density(matrix: torch.Tensor, wires: tuple[int, ...]) -> object:
    rho = torch.zeros(2**N_WIRES, 2**N_WIRES, dtype=torch.complex64)
    rho[0, 0] = 1
    return density_matrix_module.apply_unitary_density(
        rho, matrix, list(wires), N_WIRES
    )


def _apply_kraus_density(matrix: torch.Tensor, wires: tuple[int, ...]) -> object:
    rho = torch.zeros(2**N_WIRES, 2**N_WIRES, dtype=torch.complex64)
    rho[0, 0] = 1
    return density_matrix_module.apply_kraus_density(
        rho, [matrix], list(wires), N_WIRES
    )


OPERATIONS: tuple[tuple[str, Operation], ...] = (
    ("expand_operator", _expand_operator),
    ("apply_unitary_density", _apply_unitary_density),
    ("apply_kraus_density", _apply_kraus_density),
)


@pytest.mark.parametrize(
    ("matrix", "wires", "reason"),
    (
        (torch.eye(4, dtype=torch.complex64), (0,), "too large for one wire"),
        (X, (0, 1), "too small for two wires"),
        (torch.eye(3, dtype=torch.complex64), (0,), "not a power-of-two dimension"),
        (torch.zeros(2, 3, dtype=torch.complex64), (0, 1), "rectangular"),
    ),
    ids=("4x4-on-1-wire", "2x2-on-2-wires", "3x3-on-1-wire", "2x3-rectangular"),
)
@pytest.mark.parametrize(
    ("operation", "run"), OPERATIONS, ids=[name for name, _ in OPERATIONS]
)
def test_operator_that_cannot_match_its_wires_is_refused(
    operation: str,
    run: Operation,
    matrix: torch.Tensor,
    wires: tuple[int, ...],
    reason: str,
) -> None:
    with pytest.raises(ValueError):
        run(matrix, wires)


def test_an_oversized_operator_is_not_silently_truncated() -> None:
    """A 4x4 operator on one wire used to give the result for its top-left 2x2.

    The discarded quadrants were replaced by the values in ``X``, so a caller who
    passed the wrong matrix by mistake received a well-formed density matrix for
    an operator that was never applied.
    """

    padded = CX.clone()
    padded[2:, 2:] = 9.0

    with pytest.raises(ValueError, match="operator dimension does not match"):
        _expand_operator(padded, (0,))

    # The refusal is the point: the wrong-size call must not equal the right-size one.
    assert not torch.equal(
        density_matrix_module.expand_operator(X, [0], N_WIRES),
        density_matrix_module.expand_operator(CX, [0, 1], N_WIRES),
    )


def test_matching_operator_sizes_still_expand_and_apply() -> None:
    """The guard must not reject an operator that does match its wires."""

    for wire in range(N_WIRES):
        assert density_matrix_module.expand_operator(X, [wire], N_WIRES).shape == (
            2**N_WIRES,
            2**N_WIRES,
        )
        applied = _apply_unitary_density(X, (wire,))
        assert applied.shape == (1, 2**N_WIRES, 2**N_WIRES)
        assert torch.allclose(applied[0].trace().real, torch.tensor(1.0), atol=1e-6)

    assert density_matrix_module.expand_operator(CX, [0, 1], N_WIRES).shape == (
        2**N_WIRES,
        2**N_WIRES,
    )
    assert density_matrix_module.expand_operator(
        torch.stack([X, X]), [0], N_WIRES
    ).shape == (2, 2**N_WIRES, 2**N_WIRES)
