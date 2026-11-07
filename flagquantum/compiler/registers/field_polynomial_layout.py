# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GFPolySplit and GFPolyJoin.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_poly_arithmetic/gf_poly_split_and_join.py

"""Gate-free views of packed finite-field polynomial registers."""

from dataclasses import dataclass

from .partition import RegisterLayout, partition_register, unpartition_register
from .register_bits import join_integer, split_integer


@dataclass(frozen=True)
class FieldPolynomialLayout:
    """Bind coefficient words in descending degree, with big-endian field bits.

    The coefficient width comes from the caller's field encoding. A degree
    bound includes the constant coefficient, and leading zero words retain
    their positions. These views reuse the same wires and require no scratch.
    """

    degree: int
    coefficient_width: int

    def __post_init__(self) -> None:
        if type(self.degree) is not int or self.degree < 0:
            raise ValueError("Polynomial layout requires a concrete nonnegative degree")
        if type(self.coefficient_width) is not int or self.coefficient_width < 1:
            raise ValueError("Polynomial layout requires a positive coefficient width")

    @property
    def logical_width(self) -> int:
        return (self.degree + 1) * self.coefficient_width

    @property
    def _layout(self) -> RegisterLayout:
        return tuple(
            (f"a{power}", self.coefficient_width)
            for power in range(self.degree, -1, -1)
        )

    def split(self, wires: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        """View a flat register as coefficient wire words, highest power first."""
        return tuple(partition_register(wires, self._layout).values())

    def join(self, coefficients: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Restore the flat view in the same coefficient and bit order."""
        layout = self._layout
        if len(coefficients) != len(layout):
            raise ValueError("Polynomial join requires degree plus one coefficients")
        return unpartition_register(
            {name: word for (name, _), word in zip(layout, coefficients, strict=True)},
            layout,
        )

    def unpack_basis(self, packed: int) -> tuple[int, ...]:
        """Read encoded coefficient words, including padding, from a basis index."""
        bits = split_integer(packed, self.logical_width)
        width = self.coefficient_width
        return tuple(
            join_integer(bits[offset : offset + width])
            for offset in range(0, self.logical_width, width)
        )

    def pack_basis(self, coefficients: tuple[int, ...]) -> int:
        """Pack encoded coefficient words without applying finite-field arithmetic."""
        if len(coefficients) != self.degree + 1:
            raise ValueError("Polynomial join requires degree plus one coefficients")
        return join_integer(
            tuple(
                bit
                for coefficient in coefficients
                for bit in split_integer(coefficient, self.coefficient_width)
            )
        )
