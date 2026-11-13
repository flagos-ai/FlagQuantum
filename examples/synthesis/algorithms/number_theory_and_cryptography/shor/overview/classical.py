"""Exact continued fractions and checked mathematical Shor postprocessing."""

from __future__ import annotations

import math
from fractions import Fraction


def convergents(value: Fraction) -> tuple[Fraction, ...]:
    """Enumerate exact continued-fraction convergents in source order."""
    if not isinstance(value, Fraction) or value < 0:
        raise ValueError("use an exact nonnegative fraction")
    numerator, denominator = value.numerator, value.denominator
    previous_p, p, previous_q, q = 0, 1, 1, 0
    result = []
    while denominator:
        quotient, remainder = divmod(numerator, denominator)
        previous_p, p = p, quotient * p + previous_p
        previous_q, q = q, quotient * q + previous_q
        result.append(Fraction(p, q))
        numerator, denominator = denominator, remainder
    return tuple(result)


def postprocess_phase(value: Fraction, modulus: int) -> Fraction:
    """Preserve the source's last convergent with denominator strictly below n."""
    if type(modulus) is not int or modulus < 2:
        raise ValueError("modulus must be an integer greater than one")
    if not isinstance(value, Fraction) or not 0 <= value < 1:
        raise ValueError("phase must be an exact fraction in [0,1)")
    return tuple(row for row in convergents(value) if row.denominator < modulus)[-1]


def factors_from_order(base: int, order: int, modulus: int) -> tuple[int, int] | None:
    """Use mathematical modular powers, correcting the source helper's XOR typo."""
    if any(type(value) is not int for value in (base, order, modulus)):
        raise ValueError("base, order and modulus must be integers")
    if modulus < 3 or not 1 < base < modulus or math.gcd(base, modulus) != 1:
        raise ValueError("base must be a nontrivial invertible residue")
    if order < 1 or pow(base, order, modulus) != 1:
        raise ValueError("order must be a positive exponent returning one")
    if order % 2:
        return None
    halfway = pow(base, order // 2, modulus)
    if halfway in (1, modulus - 1):
        return None
    factors = sorted((math.gcd(halfway - 1, modulus), math.gcd(halfway + 1, modulus)))
    if factors[0] <= 1 or factors[1] >= modulus or math.prod(factors) != modulus:
        return None
    return factors[0], factors[1]


def recover_order(
    observations: tuple[tuple[Fraction, int], ...], modulus: int
) -> tuple[tuple[Fraction, ...], int]:
    """Apply count>10, retain source fractions and derive the denominator LCM."""
    for value, count in observations:
        if type(count) is not int or count < 1:
            raise ValueError("observation counts must be positive integers")
        postprocess_phase(value, modulus)
    fractions = tuple(
        sorted(
            {
                postprocess_phase(value, modulus)
                for value, count in observations
                if count > 10
            }
        )
    )
    if not fractions:
        raise ValueError("no phase passed the source count threshold")
    order = math.lcm(*(value.denominator for value in fractions))
    if order == 1:
        raise ValueError("zero phase alone does not identify a nontrivial period")
    return fractions, order
