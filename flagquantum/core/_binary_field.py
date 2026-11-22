# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/gf2_field.cc and gf2_field.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.
# Immutable integers/matrices replace native polynomial values and inline storage.

"""Private concrete binary field values with source construction and arithmetic."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ._binary_field_modulus import PolynomialModulus, _check_degree, default_polynomial
from ._binary_matrix import BinaryMatrix
from ._binary_polynomial import invert_mod, parse, remainder


def _polynomial(value: int | str) -> int:
    if isinstance(value, str):
        return parse(value)
    if type(value) is not int or value < 0 or value.bit_length() > 1024:
        raise ValueError("Binary polynomial must fit 1024 nonnegative coefficient bits")
    return value


@dataclass(frozen=True, init=False)
class BinaryField:
    """Concrete field construction shared by native metadata and synthesis.

    Specify degree with an optional modulus or infer degree from a modulus.
    Coefficient k is bit k. Default degree construction searches a primitive
    polynomial and selects x. Explicit moduli select the smallest generator.
    The existing Qualtran numerical codec keeps its independent galois defaults.
    """

    degree: int
    modulus: int
    primitive_element: int
    _arithmetic: PolynomialModulus = field(repr=False, compare=False)

    def __init__(
        self,
        degree: int | None = None,
        modulus: int | str | None = None,
        primitive_element: int | str | None = None,
    ) -> None:
        default = modulus is None
        if default:
            if degree is None:
                raise ValueError("Binary field requires a degree or modulus")
            modulus = default_polynomial(degree)
        else:
            assert modulus is not None
            modulus = _polynomial(modulus)
            if degree is None:
                degree = modulus.bit_length() - 1
        _check_degree(degree)
        if modulus.bit_length() <= degree:
            modulus |= 1 << degree
        if modulus.bit_length() != degree + 1 or not modulus & 1:
            raise ValueError(
                "Binary field modulus requires the declared degree and constant one"
            )
        arithmetic = PolynomialModulus(degree, modulus)
        if degree >= 2 and not arithmetic.is_irreducible():
            raise ValueError("Binary field modulus must be irreducible")
        primitive = primitive_element
        if primitive is None:
            generator = 1 if degree == 1 else 2
            if not default:
                while not arithmetic.is_primitive(generator):
                    generator += 1
        else:
            generator = _polynomial(primitive)
            if not 0 < generator < 1 << degree or not arithmetic.is_primitive(
                generator
            ):
                raise ValueError(
                    "Binary field generator must be a nonzero primitive element"
                )
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "modulus", modulus)
        object.__setattr__(self, "primitive_element", generator)
        object.__setattr__(self, "_arithmetic", arithmetic)

    def is_element(self, value: int) -> bool:
        return 0 <= value < 1 << self._arithmetic.degree

    def is_primitive(self, value: int) -> bool:
        return (
            value != 0
            and self.is_element(value)
            and self._arithmetic.is_primitive(value)
        )

    def mod(self, value: int) -> int:
        return remainder(_polynomial(value), self._arithmetic.modulus)

    def reduce(self, value: int) -> int:
        """Retain the source fast reduction for already bounded products."""
        return self._arithmetic.reduce(_polynomial(value))

    def add(self, left: int, right: int) -> int:
        return _polynomial(left) ^ _polynomial(right)

    def multiply(self, left: int, right: int) -> int:
        return self._arithmetic.multiply(self.mod(left), self.mod(right))

    def square(self, value: int) -> int:
        return self._arithmetic.square(self.mod(value))

    def frobenius(self, value: int, count: int) -> int:
        if type(count) is not int or count < 0:
            raise ValueError("Frobenius count must be a nonnegative integer")
        result = self.mod(value)
        for _ in range(count % self._arithmetic.degree):
            result = self._arithmetic.square(result)
        return result

    def power(self, value: int, exponent: int) -> int:
        return self._arithmetic.power(self.mod(value), exponent)

    def inverse(self, value: int) -> int:
        return invert_mod(self.mod(value), self._arithmetic.modulus)

    def divide(self, left: int, right: int) -> int:
        return self.multiply(left, self.inverse(right))

    def frobenius_matrix(self, count: int) -> BinaryMatrix:
        if type(count) is not int or count < 0:
            raise ValueError("Frobenius count must be a nonnegative integer")
        degree = self._arithmetic.degree
        root = self.frobenius(2, count)
        columns = []
        value = 1
        for _ in range(degree):
            columns.append(value)
            value = self.multiply(value, root)
        return BinaryMatrix(tuple(columns), degree).transposed()

    def constant_matrix(self, constant: int) -> BinaryMatrix:
        value = self.mod(constant)
        if not value:
            raise ValueError("Binary field constant matrix requires a nonzero constant")
        degree = self._arithmetic.degree
        columns = []
        for _ in range(degree):
            columns.append(value)
            value <<= 1
            if value >> degree & 1:
                value ^= 1 << degree
                for exponent in self._arithmetic.terms:
                    value ^= 1 << exponent
        return BinaryMatrix(tuple(columns), degree).transposed()

    def linear_map_matrix(self, function: Callable[[int], int]) -> BinaryMatrix:
        degree = self._arithmetic.degree
        mask = (1 << degree) - 1
        return BinaryMatrix(
            tuple(_polynomial(function(1 << index)) & mask for index in range(degree)),
            degree,
        ).transposed()
