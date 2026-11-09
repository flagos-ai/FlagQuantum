# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Register metadata adapted from Qualtran v0.7.0 Register and Side.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/registers.py

"""Neutral register metadata, separate from numerical codecs and executable IR.

Proposal 103 metadata prerequisite 0.1 owns this internal boundary. These records
do not extend Program IR's admitted types, serialize a new IR or certify that an
encoding or symbolic dimension can be lowered.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Flag, auto
from typing import TYPE_CHECKING, TypeAlias, cast

from ...value_math import SymbolicInt as RegisterWidth
from ...value_math import prod
from ...value_parameters import is_symbolic_value

if TYPE_CHECKING:
    from sympy import Expr

TypeParameter: TypeAlias = "int | str | float | Expr"


@dataclass(frozen=True)
class RegisterType:
    """Identity and exact widths of a prevalidated quantum/classical encoding.

    Encoding-specific constructors and numerical codecs are separate owners.
    Named parameters distinguish types such as bounded integers; they do not
    become callbacks or authorize execution of an unknown encoding.
    """

    encoding: str
    quantum_bits: RegisterWidth
    classical_bits: RegisterWidth = 0
    parameters: tuple[tuple[str, TypeParameter], ...] = ()

    @property
    def num_qubits(self) -> RegisterWidth:
        return self.quantum_bits

    @property
    def num_cbits(self) -> RegisterWidth:
        return self.classical_bits

    @property
    def num_bits(self) -> RegisterWidth:
        return self.num_qubits + self.num_cbits

    def is_symbolic(self) -> bool:
        # Source Montgomery classification examines width, ignoring modulus.
        if self.encoding == "montgomery":
            return is_symbolic_value(self.quantum_bits, self.classical_bits)
        if self.encoding == "fixed_point":
            return is_symbolic_value(
                self.quantum_bits,
                self.classical_bits,
                *(
                    value
                    for name, value in self.parameters
                    if name == "fractional_width"
                ),
            )
        return is_symbolic_value(
            self.quantum_bits,
            self.classical_bits,
            *(value for _, value in self.parameters),
        )


class RegisterSide(Flag):
    """Input-only, output-only, or shared input/output register ports."""

    LEFT = auto()
    RIGHT = auto()
    THRU = LEFT | RIGHT


@dataclass(frozen=True, init=False)
class Register:
    """One named scalar or row-major array of typed logical register elements.

    A scalar zero-width register has one logical index, ``()``. An array with
    a zero-length dimension has no elements. Register metadata preserves that
    distinction independently of its eventual physical wire representation.
    """

    name: str
    dtype: RegisterType
    _shape: tuple[RegisterWidth, ...]
    side: RegisterSide

    def __init__(
        self,
        name: str,
        dtype: RegisterType,
        shape: int | Iterable[RegisterWidth] = (),
        side: RegisterSide = RegisterSide.THRU,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self, "_shape", (shape,) if isinstance(shape, int) else tuple(shape)
        )
        object.__setattr__(self, "side", side)
        if not isinstance(dtype, RegisterType):
            raise ValueError(f"dtype must be a RegisterType: found {type(dtype)}")

    def is_symbolic(self) -> bool:
        return is_symbolic_value(self.dtype, *self._shape)

    @property
    def shape_symbolic(self) -> tuple[RegisterWidth, ...]:
        return self._shape

    @property
    def shape(self) -> tuple[int, ...]:
        if is_symbolic_value(*self._shape):
            raise ValueError(f"{self} is symbolic. Cannot get real-valued shape.")
        return cast(tuple[int, ...], self._shape)

    @property
    def bitsize(self) -> RegisterWidth:
        return self.dtype.num_bits

    def all_idxs(self) -> Iterator[tuple[int, ...]]:
        """Yield element indices in the locked source's Cartesian-product order."""
        yield from itertools.product(*(range(size) for size in self.shape))

    def total_bits(self) -> RegisterWidth:
        return self.bitsize * prod(self.shape_symbolic)

    def total_qubits(self) -> RegisterWidth:
        return self.dtype.num_qubits * prod(self.shape_symbolic)

    def total_cbits(self) -> RegisterWidth:
        return self.dtype.num_cbits * prod(self.shape_symbolic)

    def adjoint(self) -> Register:
        """Exchange input/output direction; a through register retains identity."""
        if self.side is RegisterSide.THRU:
            return self
        if self.side is RegisterSide.LEFT:
            return type(self)(self.name, self.dtype, self._shape, RegisterSide.RIGHT)
        if self.side is RegisterSide.RIGHT:
            return type(self)(self.name, self.dtype, self._shape, RegisterSide.LEFT)
        raise ValueError(f"Unknown side {self.side}")
