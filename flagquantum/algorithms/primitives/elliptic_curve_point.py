# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 elliptic curve point arithmetic.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_point.py

"""Affine point arithmetic shared by elliptic-curve algorithms."""

from dataclasses import dataclass
from operator import index
from typing import SupportsIndex, cast

from ...core.value_math import SymbolicInt
from ...core.value_parameters import is_symbolic_value


@dataclass(frozen=True)
class ECPoint:
    """A point in the source affine coordinates and prime-field convention.

    The pair (0, 0) represents infinity. Callers supply coordinates from the
    same short Weierstrass curve; the point stores its modulus and a coefficient.
    """

    x: SymbolicInt
    y: SymbolicInt
    mod: SymbolicInt
    curve_a: SymbolicInt = 0

    @classmethod
    def inf(cls, mod: SymbolicInt, curve_a: SymbolicInt) -> "ECPoint":
        """Return the source identity point."""
        return cls(0, 0, mod, curve_a)

    def __neg__(self) -> "ECPoint":
        return ECPoint(self.x, (-self.y) % self.mod, self.mod, self.curve_a)

    def __add__(self, other: "ECPoint") -> "ECPoint":
        if self.mod != other.mod or self.curve_a != other.curve_a:
            raise ValueError("Use consistent mod and curve")
        identity = ECPoint.inf(self.mod, self.curve_a)
        if self == -other:
            return identity
        if self == identity:
            return other
        if other == identity:
            return self

        if self == other:
            numerator = 3 * self.x**2 + self.curve_a
            denominator = 2 * self.y
        else:
            numerator = other.y - self.y
            denominator = other.x - self.x

        inverse = pow(int(denominator % self.mod), -1, int(self.mod))
        slope = ((numerator % self.mod) * inverse) % self.mod
        x = (slope**2 - self.x - other.x) % self.mod
        y = (slope * (self.x - x) - self.y) % self.mod
        return ECPoint(x, y, self.mod, self.curve_a)

    def __mul__(self, scalar: SymbolicInt) -> "ECPoint":
        if scalar == 0:
            return ECPoint.inf(self.mod, self.curve_a)
        if is_symbolic_value(self.x, self.y):
            from sympy import Function

            return ECPoint(
                cast(SymbolicInt, Function("ec_times_x")(self.x, scalar)),
                cast(SymbolicInt, Function("ec_times_y")(self.y, scalar)),
                self.mod,
                self.curve_a,
            )

        count = index(cast(SupportsIndex, scalar))
        addend = -self if count < 0 else self
        count = abs(count)
        result = ECPoint.inf(self.mod, self.curve_a)
        while count:
            if count & 1:
                result = result + addend
            count >>= 1
            if count:
                addend = addend + addend
        return result

    def __rmul__(self, scalar: SymbolicInt) -> "ECPoint":
        return self * scalar

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
