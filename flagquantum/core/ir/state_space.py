"""Finite product and explicit-basis semantics shared by execution paths.

This is internal mathematical data, not a program IR or a physical-system model.
CircuitIR 1.0 continues to describe qubit circuits only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, init=False)
class ProductStateSpace:
    """An ordered finite tensor product with subsystem 0 most significant."""

    dimensions: tuple[int, ...]

    def __init__(self, dimensions: Sequence[int]) -> None:
        if not isinstance(dimensions, Sequence) or isinstance(dimensions, (str, bytes)):
            raise TypeError("dimensions must be an ordered sequence of integers")
        resolved = tuple(dimensions)
        if not resolved or any(type(item) is not int or item < 1 for item in resolved):
            raise ValueError("dimensions must contain positive integers")
        object.__setattr__(self, "dimensions", resolved)

    @property
    def dimension(self) -> int:
        return math.prod(self.dimensions)

    @property
    def is_qubit_space(self) -> bool:
        return all(dimension == 2 for dimension in self.dimensions)

    def basis_index(self, occupations: Sequence[int]) -> int:
        """Return the flat index of one in-range mixed-radix basis tuple."""

        if len(occupations) != len(self.dimensions):
            raise ValueError("one occupation is required per subsystem")
        index = 0
        for occupation, dimension in zip(occupations, self.dimensions, strict=True):
            if type(occupation) is not int or not 0 <= occupation < dimension:
                raise ValueError("occupation is outside the subsystem basis")
            index = index * dimension + occupation
        return index


def _basis_label(value: Sequence[int | str]) -> tuple[int | str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("a basis label must be an ordered sequence")
    label = tuple(value)
    if not label or any(
        type(atom) not in (int, str) or (type(atom) is str and not atom)
        for atom in label
    ):
        raise ValueError(
            "basis labels require nonempty rows of integer or string atoms"
        )
    return label


@dataclass(frozen=True, slots=True, init=False)
class ExplicitBasisSpace:
    """Ordered identifiers for a finite orthonormal, non-factorized basis.

    The caller owns the physical model and orthonormal-basis convention. Labels
    identify coordinates; they do not certify charge sectors or fusion rules.
    """

    labels: tuple[tuple[int | str, ...], ...]

    def __init__(self, labels: Sequence[Sequence[int | str]]) -> None:
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise TypeError("basis labels must be an ordered sequence")
        rows = tuple(_basis_label(row) for row in labels)
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("basis labels must be nonempty and have a common arity")
        if len(set(rows)) != len(rows):
            raise ValueError("basis labels must be unique")
        object.__setattr__(self, "labels", rows)

    @property
    def dimension(self) -> int:
        return len(self.labels)

    @property
    def is_qubit_space(self) -> bool:
        return False

    def basis_index(self, label: Sequence[int | str]) -> int:
        """Return the index of an exact label without inferring an embedding."""

        return self.labels.index(_basis_label(label))
