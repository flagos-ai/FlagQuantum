# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Cast in Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/cast.py

"""Register metadata casts that preserve the stored bit pattern."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..bits.bitwise import _validate_wires
from .register_bits import join_integer, split_integer


@dataclass(frozen=True)
class RegisterCast:
    """A width-preserving view over a register or a shaped array of words."""

    source_width: SymbolicInt
    target_width: SymbolicInt
    shape: tuple[SymbolicInt, ...] = ()
    source_quantum: bool = True
    target_quantum: bool = True
    allow_quantum_to_classical: bool = False

    def __post_init__(self) -> None:
        if (
            self.source_quantum != self.target_quantum
            and not self.allow_quantum_to_classical
        ):
            raise ValueError("A quantum/classical cast requires an explicit opt-in")
        for width in (self.source_width, self.target_width, *self.shape):
            if isinstance(width, (int, float)) and (
                type(width) is not int or width < 0
            ):
                raise ValueError(
                    "Register cast dimensions must be nonnegative integers"
                )
        if type(self.source_width) is int and self.source_width != self.target_width:
            raise ValueError("Register cast must preserve the word width")

    @property
    def logical_width(self) -> SymbolicInt:
        """Report the source peak quantum width, including symbolic shapes."""
        size: SymbolicInt = 1
        for extent in self.shape:
            size *= extent
        return cast(
            SymbolicInt,
            smax(
                self.source_width * size if self.source_quantum else 0,
                self.target_width * size if self.target_quantum else 0,
            ),
        )

    def adjoint(self) -> RegisterCast:
        """Reverse the native view while retaining the layout and explicit opt-in."""
        return replace(
            self,
            source_width=self.target_width,
            target_width=self.source_width,
            source_quantum=self.target_quantum,
            target_quantum=self.source_quantum,
        )

    def _concrete_width(self) -> int:
        if not isinstance(self.source_width, int) or not isinstance(
            self.target_width, int
        ):
            raise ValueError("Register cast execution requires concrete widths")
        return self.source_width

    def bind(self, wires: tuple[int, ...]) -> tuple[int, ...]:
        """Keep the same ordered slots in the new view, without emitting a gate."""
        count = self._concrete_width()
        for extent in self.shape:
            if not isinstance(extent, int):
                raise ValueError("Register cast execution requires a concrete shape")
            count *= extent
        if len(wires) != count:
            raise ValueError("Register cast binding must match the declared layout")
        if wires:
            _validate_wires(wires)
        return wires

    def reinterpret_integer(
        self, value: int, *, source_signed: bool = False, target_signed: bool = False
    ) -> int:
        """Reinterpret one stored integer, including a fixed-point word."""
        bits = split_integer(value, self._concrete_width(), signed=source_signed)
        return join_integer(bits, signed=target_signed)
