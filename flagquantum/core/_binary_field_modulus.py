# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/gf2_field.cc, GF2PolyMod and factor search.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.
# Python integers replace fixed-width integer limbs and mutable polynomial values.

"""Private polynomial modulus arithmetic and exact binary field construction."""

from dataclasses import dataclass, field
from functools import lru_cache

from ._binary_field_factor_offsets import CYCLOTOMIC_U64_OFFSETS
from ._binary_field_factors import CYCLOTOMIC_BIG_FACTORS, CYCLOTOMIC_U64_FACTORS
from ._binary_polynomial import gcd, multiply, remainder


def _check_degree(degree: int) -> None:
    if type(degree) is not int or not 1 <= degree <= 512:
        raise ValueError("Binary field degree must be an integer between 1 and 512")


@lru_cache(maxsize=512)
def mersenne_prime_factors(degree: int) -> tuple[int, ...]:
    """Reconstruct ordered distinct factors using the original cyclotomic tables."""
    _check_degree(degree)
    divisors = [value for value in range(1, degree + 1) if degree % value == 0]
    phi = [(1 << divisor) - 1 for divisor in divisors]
    for index, divisor in enumerate(divisors):
        for later in range(index + 1, len(divisors)):
            if divisors[later] % divisor == 0 and phi[later] % phi[index] == 0:
                phi[later] //= phi[index]
    factors: list[int] = []
    for index, divisor in enumerate(divisors):
        if divisor == 1:
            continue
        remaining = phi[index]
        for prime in factors:
            if prime <= divisor:
                while remaining % prime == 0:
                    remaining //= prime
        start, stop = CYCLOTOMIC_U64_OFFSETS[divisor - 1 : divisor + 1]
        candidates = list(CYCLOTOMIC_U64_FACTORS[start:stop])
        candidates.extend(
            low | (middle << 64) | (high << 128)
            for field_degree, low, middle, high in CYCLOTOMIC_BIG_FACTORS
            if field_degree == divisor
        )
        for prime in candidates:
            if remaining % prime == 0:
                factors.append(prime)
                while remaining % prime == 0:
                    remaining //= prime
        if remaining > 1:
            factors.append(remaining)
    return tuple(factors)


def _spread_bits_32(value: int) -> int:
    value &= 0xFFFFFFFF
    value = (value | value << 16) & 0x0000FFFF0000FFFF
    value = (value | value << 8) & 0x00FF00FF00FF00FF
    value = (value | value << 4) & 0x0F0F0F0F0F0F0F0F
    value = (value | value << 2) & 0x3333333333333333
    return (value | value << 1) & 0x5555555555555555


@dataclass(frozen=True)
class PolynomialModulus:
    """Source unchecked modular algebra on bounded polynomial coefficient values.

    Callers own admission and reduced-operand preconditions. This value is also
    used to test candidates before a validated field can exist.
    """

    degree: int
    modulus: int
    terms: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terms",
            tuple(
                index
                for index in reversed(range(self.degree))
                if self.modulus >> index & 1
            ),
        )

    @property
    def sparse_reduction(self) -> bool:
        return len(self.terms) <= 16

    def reduce(self, value: int) -> int:
        if value.bit_length() <= self.degree:
            return value
        if not self.sparse_reduction:
            return remainder(value, self.modulus)
        for index in reversed(range(self.degree, value.bit_length())):
            if value >> index & 1:
                value ^= 1 << index
                base = index - self.degree
                for exponent in self.terms:
                    value ^= 1 << (base + exponent)
        return value

    def multiply(self, left: int, right: int) -> int:
        return self.reduce(multiply(left, right))

    def square(self, value: int) -> int:
        result = 0
        # Source low/high 32-bit halves map to successive 64-bit result words.
        for word in range(min(8, (value.bit_length() + 63) // 64)):
            low = _spread_bits_32(value >> (word * 64))
            high = _spread_bits_32(value >> (word * 64 + 32))
            result |= (low | (high << 64)) << (word * 128)
        return self.reduce(result)

    def power(self, value: int, exponent: int) -> int:
        if type(exponent) is not int or exponent < 0:
            raise ValueError("Polynomial modular exponent must be nonnegative")
        result = 1
        for index in range(exponent.bit_length()):
            if exponent >> index & 1:
                result = self.multiply(result, value)
            value = self.square(value)
        return result

    def is_irreducible(self) -> bool:
        value = 2
        for _ in range(self.degree // 2):
            value = self.square(value)
            if gcd(value ^ 2, self.modulus) != 1:
                return False
        return True

    def is_primitive(self, value: int) -> bool:
        if self.degree == 1:
            return value == 1
        if value == 1:
            return False
        order = (1 << self.degree) - 1
        return all(
            self.power(value, order // prime) != 1
            for prime in mersenne_prime_factors(self.degree)
        )


def is_irreducible(polynomial: int) -> bool:
    """Retain original degree/constant-bit check order before Ben-Or testing."""
    if type(polynomial) is not int or polynomial < 0 or polynomial.bit_length() > 1024:
        raise ValueError("Binary polynomial must fit 1024 nonnegative coefficient bits")
    degree = polynomial.bit_length() - 1
    if degree < 1:
        return False
    if degree == 1:
        return True
    if not polynomial & 1:
        return False
    if degree > 512:
        raise ValueError("Binary polynomial degree exceeds 512")
    return PolynomialModulus(degree, polynomial).is_irreducible()


@lru_cache(maxsize=512)
def default_polynomial(degree: int) -> int:
    """Search trinomials then pentanomials in the exact source candidate order."""
    _check_degree(degree)
    base = (1 << degree) | 1
    if degree == 1:
        return base
    for first in range(1, degree):
        candidate = base | (1 << first)
        modulus = PolynomialModulus(degree, candidate)
        if modulus.is_irreducible() and modulus.is_primitive(2):
            return candidate
    for first in range(3, degree):
        for second in range(2, first):
            for third in range(1, second):
                candidate = base | (1 << first) | (1 << second) | (1 << third)
                modulus = PolynomialModulus(degree, candidate)
                if modulus.is_irreducible() and modulus.is_primitive(2):
                    return candidate
    raise RuntimeError(f"No primitive binary polynomial found for degree {degree}")
