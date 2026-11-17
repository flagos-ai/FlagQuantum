# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 rotation_synthesis/rings/_zw.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/rings/_zw.py

"""The ring Z[w] where w = e^{i pi/4}"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ._rotation_math import MathConfig
from ._rotation_types import Complex, Integral, Real, is_int
from ._zsqrt2 import ZSqrt2

CoordinateInput = Sequence[Integral] | NDArray[np.integer[Any]]

_Z8 = np.exp(1j * np.pi / 4)
_Z8_POWERS = np.array([_Z8**i for i in range(4)])


def _coords_converter(coords: CoordinateInput) -> tuple[int, int, int, int]:
    assert all(is_int(c) for c in coords)
    res = tuple(int(c) for c in coords)
    assert len(res) == 4
    return res


@dataclass(frozen=True, eq=False, init=False)
class ZW:
    r"""Elements of the ring $\mathbb{Z}[\omega]$ where $\omega=\zeta_8=e^{i\pi/4}$.

    Elements of $\mathbb{Z}[\omega]$ can be represented by 4 integers as
    $$
        \sum_{i=0}^3 a_i \omega^i
    $$

    Attributes:
        coords: The 4 integers defining the element.
    """

    coords: tuple[int, int, int, int]

    def __init__(self, coords: CoordinateInput) -> None:
        object.__setattr__(self, "coords", _coords_converter(coords))

    def __complex__(self) -> complex:
        return complex(np.dot(self.coords, _Z8_POWERS))

    def value(self, sqrt2: Real) -> Complex:
        """Return the floating-point value of the element.

        The method uses the given value of sqrt2 (e.g. np.sqrt(2), mpmath.sqrt(2), ...etc)
        to control the precision of the result.
        """
        a, b, c, d = self.coords
        real = a + (b - d) / sqrt2
        imag = c + (b + d) / sqrt2
        return real + 1j * imag

    def arg(self, config: MathConfig) -> Real:
        """Return the angle of the complex number representing the element.

        Note: The precision of the result depends on the provided config.

        Args:
            config: A MathConfig object.
        """
        v = self.value(config.sqrt2)
        return config.arctan2(v.imag, v.real)

    def mag(self, sqrt2: Real) -> Real:
        """Return the magnitude of the complex number representing the element.

        Note: The precision of the result depends on the precision of the provided `sqrt2`.

        Args:
            sqrt2: The value of sqrt2 to use.
        """
        return abs(self.value(sqrt2))

    def polar(self, config: MathConfig) -> tuple[Real, Real]:
        """Returns the polar form of the complex number representing the element.

        Note: The precision of the result depends on the provided config.

        Args:
            config: A MathConfig object.
        """
        return self.mag(config.sqrt2), self.arg(config)

    def __mul__(self, other: object) -> ZW:
        if isinstance(other, ZW):
            c = [0] * 4
            for i in range(4):
                for j in range(4):
                    sgn = -1 if i + j >= 4 else 1
                    c[(i + j) & 3] += sgn * self.coords[i] * other.coords[j]
            return ZW(c)
        if is_int(other):
            return self * ZW([int(other), 0, 0, 0])
        return NotImplemented

    def __add__(self, other: object) -> ZW:
        if isinstance(other, ZW):
            return ZW([a + b for a, b in zip(self.coords, other.coords, strict=True)])
        if is_int(other):
            return self + ZW([int(other), 0, 0, 0])
        return NotImplemented

    def __sub__(self, other: object) -> ZW:
        if isinstance(other, ZW):
            return ZW([a - b for a, b in zip(self.coords, other.coords, strict=True)])
        if is_int(other):
            return self - ZW([int(other), 0, 0, 0])
        return NotImplemented

    def __rmul__(self, other: object) -> ZW:
        return self * other

    def __radd__(self, other: object) -> ZW:
        return self + other

    def __rsub__(self, other: object) -> ZW:
        return -(self - other)

    def __pow__(self, exponent: object) -> ZW:
        if not is_int(exponent):
            raise TypeError(f"{exponent=} is not integral")
        exponent = int(exponent)
        if exponent < 0:
            raise ValueError(f"negative {exponent=} is not supported")
        if exponent == 0:
            return One
        other = One
        cur = self
        while exponent > 1:
            if exponent & 1:
                other = other * cur
            cur = cur * cur
            exponent >>= 1
        return cur * other

    def __neg__(self) -> ZW:
        return ZW([-c for c in self.coords])

    def __hash__(self) -> int:
        return hash(self.coords)

    def __eq__(self, other: object) -> bool:
        return cast(bool, self.coords == cast(Any, other).coords)

    def conjugate(self) -> ZW:
        """Return the complex-conjugate of the element."""
        a, b, c, d = self.coords
        return ZW([a, -d, -c, -b])

    def conj(self) -> ZW:
        """Return the complex-conjugate of the element."""
        return self.conjugate()

    def sqrt2_conjugate(self) -> ZW:
        """Return the sqrt2-conjugate of the element."""
        a, b, c, d = self.coords
        return ZW([a, -b, c, -d])

    def sqrt2_conj(self) -> ZW:
        """Return the sqrt2-conjugate of the element."""
        return self.sqrt2_conjugate()

    @staticmethod
    def from_pair(a: ZSqrt2, b: ZSqrt2, include_w: bool = False) -> ZW:
        r"""Constructs a $\mathbb{Z}[\omega]$ element from its $\mathbb{\sqrt{2}}$ representation.

        Constructs the element defined by $a + i b + \mathrm{include\_w} * \omega$
        where $a, b \in \mathbb{Z}[\sqrt{2}]$.

        Args:
            a: the real part of the element.
            b: the imaginary part of the element.
            include_w: whether to add $\omega=e^{i \pi/4}$ to the element.
        """
        return ZW([a.a, a.b + b.b + int(include_w), b.a, b.b - a.b])

    def to_zsqrt2(self) -> tuple[ZSqrt2, ZSqrt2, bool]:
        r"""Writes an elements of $\mathbb{Z}[\omega=e^{i \pi/4}]$ in terms of $\mathbb{Z}[\sqrt{2}]

        Every element of $\mathbb{Z}[e^{i \pi/4}]$ can be written in one of two forms, either
        $\alpha + \beta i$ or $\alpha + \beta i + \omega$.

        Returns:
            - $\alpha$, $\beta$, and a boolean indicating whether $\omega$ is needed or not.
        """
        m0, m1, m2, m3 = self.coords
        r = (m1 + m3) % 2
        return (
            ZSqrt2(m0, (m1 - m3 - r) // 2),
            ZSqrt2(m2, (m1 + m3 - r) // 2),
            r == 1,
        )

    def norm(self) -> int:
        res = self * self.conj() * self.sqrt2_conj() * self.sqrt2_conj().conj()
        assert all(c == 0 for c in res.coords[1:])
        assert res.coords[0] >= 0
        return res.coords[0]

    def __floordiv__(self, other: ZW) -> ZW:
        z = self * other.conj() * other.sqrt2_conj() * other.sqrt2_conj().conj()
        x_norm = other.norm()
        return ZW([v // x_norm for v in z.coords])

    def is_divisible_by(self, other: ZW) -> bool:
        z = self * other.conj() * other.sqrt2_conj() * other.sqrt2_conj().conj()
        x_norm = other.norm()
        return all(c % x_norm == 0 for c in z.coords)

    def gcd(self, other: ZW) -> ZW:
        """Returns the gcd between the operands using the Euclidean algorithm."""
        x = self
        y = other
        while True:
            if x.norm() > y.norm():
                x, y = y, x
            if x == Zero:
                break
            c = y // x
            best = (y.norm(), y)
            for dc in itertools.product(range(2), repeat=4):
                r = c + ZW(dc)
                ny = y - r * x
                if ny.norm() < best[0]:
                    best = ny.norm(), ny
            assert best[0] < y.norm()
            y = best[1]
        return y

    @staticmethod
    def factor_prime(p: int) -> dict[ZW, int]:
        """Returns the factorization of a prime into the prime ideals of the ring.

        Note: The method assumes the input is a prime.

        Args:
            p: A prime number.
        Returns:
            The factorization in terms of prime ideals.
        """
        if p == 2:
            # Special case 2 = (1 + e^{i pi/4})^4
            return {One + Omega: 4}
        from sympy import ntheory

        if p % 8 == 1:
            # roots of x^4 + 1 = 0 mod p
            roots = [
                r
                for r1 in ntheory.sqrt_mod_iter(-1, p)
                for r in ntheory.sqrt_mod_iter(r1, p)
            ]
            return {Omega - r: 1 for r in roots}
        if p % 8 == 5:
            # x^4 + 1 = (x^2 - a) (x^2 + a) mod p
            # a^2 = -1 mod p
            a = ntheory.sqrt_mod(-1, p)
            return {
                -(Omega**2 - a): 1,
                Omega**2 + a: 1,
            }  # Multiply by -1 to cancel the phase
        if p % 8 == 3:
            # x^4 + 1 = (x^2 + a - 1) (x^2 - a - 1) mod p
            # a^2 = -2 mod p
            a = ntheory.sqrt_mod(-2, p)
            return {
                Omega**2 + a * Omega - 1: 1,
                J
                * (Omega**2 - a * Omega - 1): 1,  # Multiply by 1j to cancel the phase.
            }
        assert p % 8 == 7, f"{p=}"
        # x^4 + 1 = (x^2 + a + 1) (x^2 - a + 1) mod p
        # a^2 = 2 mod p
        a = ntheory.sqrt_mod(2, p)
        return {
            Omega**2 + a * Omega + 1: 1,
            J * (Omega**2 - a * Omega + 1): 1,  # Multiply by 1j to cancel the phase.
        }


Zero = ZW([0, 0, 0, 0])
One = ZW([1, 0, 0, 0])
J = ZW([0, 0, 1, 0])  # 1j
Omega = ZW([0, 1, 0, 0])  # e^{i pi/4}
SQRT_2 = ZW([0, 1, 0, -1])
LAMBDA_KLIUCHNIKOV = ZW([2, 1, 0, -1])  # 2 + sqrt(2)
LAMBDA_KLIUCHNIKOV_SQRT2_CONJ = LAMBDA_KLIUCHNIKOV.sqrt2_conj()  # 2 - sqrt(2)
