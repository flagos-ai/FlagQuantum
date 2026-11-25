"""Generate the four original structured matrices and their address rules."""

import random
from dataclasses import dataclass
from typing import Literal

import torch

Case = Literal["checkerboard", "toeplitz", "tridiagonal", "laplacian"]
CASES: tuple[Case, ...] = ("checkerboard", "toeplitz", "tridiagonal", "laplacian")
DTYPE = torch.complex128


@dataclass(frozen=True)
class Address:
    """A permutation and loading coefficients for a single selector address."""

    rows: tuple[int, ...]
    weights: tuple[float, ...]
    invalid: tuple[bool, ...]


@dataclass(frozen=True)
class Structure:
    """A complete bounded SELECT table and its separately generated matrix."""

    name: Case
    matrix: torch.Tensor
    addresses: tuple[tuple[int, Address], ...]
    selector_bits: int
    boundary_flag: bool


def _address(
    rows: tuple[int, ...], weights: tuple[float, ...], valid: tuple[bool, ...]
) -> Address:
    return Address(rows, weights, tuple(not value for value in valid))


def _checkerboard(rng: random.Random) -> Structure:
    size = 8
    coefficients = tuple(round(rng.uniform(-1, 1), 2) for _ in range(2))
    matrix = torch.tensor(
        [[coefficients[(row + col) % 2] for col in range(size)] for row in range(size)],
        dtype=DTYPE,
    )
    terms = tuple(
        (
            key,
            _address(
                tuple((col + key) % size for col in range(size)),
                (coefficients[key % 2],) * size,
                (True,) * size,
            ),
        )
        for key in range(size)
    )
    return Structure("checkerboard", matrix, terms, 3, False)


def _toeplitz(rng: random.Random) -> Structure:
    size = 16
    diagonal = tuple(rng.uniform(-1, 1) for _ in range(4))
    matrix = sum(
        (
            torch.diag(
                torch.full((size - abs(1 - key),), value, dtype=DTYPE), diagonal=1 - key
            )
            for key, value in enumerate(diagonal)
        ),
        torch.zeros((size, size), dtype=DTYPE),
    )
    terms = []
    for key, coefficient in enumerate(diagonal):
        raw = tuple(col + key - 1 for col in range(size))
        terms.append(
            (
                key,
                _address(
                    tuple(row % size for row in raw),
                    (coefficient,) * size,
                    tuple(0 <= row < size for row in raw),
                ),
            )
        )
    return Structure("toeplitz", matrix, tuple(terms), 2, True)


def _tridiagonal(rng: random.Random) -> Structure:
    size = 16
    diagonal = tuple(round(rng.uniform(-1, 1), 4) for _ in range(size))
    off_diagonal = tuple(round(rng.uniform(-1, 1), 4) for _ in range(size - 1))
    values = torch.tensor(off_diagonal, dtype=DTYPE)
    matrix = (
        torch.diag(torch.tensor(diagonal, dtype=DTYPE))
        + torch.diag(values, diagonal=1)
        + torch.diag(values, diagonal=-1)
    )
    terms = []
    for key, displacement in ((0, 0), (1, -1), (3, 1)):
        raw = tuple(col + displacement for col in range(size))
        weights = tuple(
            (
                (diagonal[col] if displacement == 0 else off_diagonal[min(col, row)])
                if 0 <= row < size
                else 0.0
            )
            for col, row in enumerate(raw)
        )
        terms.append(
            (key, _address(tuple(row % size for row in raw), weights, (True,) * size))
        )
    return Structure("tridiagonal", matrix, tuple(terms), 2, False)


def _laplacian() -> Structure:
    side = 4
    size = side * side
    # The source scales its Laplacian by the magnitude four of its diagonal.
    one_axis = torch.diag(torch.ones(side - 1, dtype=DTYPE), diagonal=1)
    one_axis = one_axis + one_axis.mH
    identity = torch.eye(side, dtype=DTYPE)
    matrix = (
        -torch.eye(size, dtype=DTYPE)
        + (torch.kron(identity, one_axis) + torch.kron(one_axis, identity)) / 4
    )
    terms = []
    for key, dx, dy in ((0, 0, 0), (2, -1, 0), (3, 1, 0), (4, 0, -1), (5, 0, 1)):
        coords = tuple((col % side + dx, col // side + dy) for col in range(size))
        rows = tuple(x % side + side * (y % side) for x, y in coords)
        valid = tuple(0 <= x < side and 0 <= y < side for x, y in coords)
        terms.append((key, _address(rows, (-1.0 if key == 0 else 0.25,) * size, valid)))
    return Structure("laplacian", matrix, tuple(terms), 3, True)


def make_structures(seed: int = 7) -> tuple[Structure, ...]:
    """Regenerate the random coefficients locally and retain two signed sectors."""
    rng = random.Random(seed)
    return _checkerboard(rng), _toeplitz(rng), _tridiagonal(rng), _laplacian()
