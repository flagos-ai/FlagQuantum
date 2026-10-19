"""Finite tensor-product basis semantics shared by existing execution paths.

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
