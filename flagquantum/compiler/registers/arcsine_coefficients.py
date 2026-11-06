# SPDX-License-Identifier: Apache-2.0

"""Reproducible Chebyshev projection for the Appendix D arcsine factor."""

from fractions import Fraction
from math import comb


def arcsine_polynomial_coefficients(
    fraction_bits: int, *, degree: int = 4
) -> tuple[tuple[int, ...], Fraction]:
    """Return signed Horner coefficient words and a uniform factor error bound.

    The approximated factor is asin(sqrt(z))/sqrt(z) on [0, 1/4], with
    value one at zero. Coefficients are in descending degree order and
    rounded to the nearest output ulp, with ties to even. The bound
    covers this quantized polynomial before circuit arithmetic.
    """
    if type(fraction_bits) is not int or fraction_bits < 0:
        raise ValueError("Arcsine coefficient precision must be nonnegative")
    if type(degree) is not int or degree < 0:
        raise ValueError("Arcsine polynomial degree must be nonnegative")
    order = max(degree, (fraction_bits + 1) // 2 + 4)
    chebyshev = [Fraction(0)] * (degree + 1)
    endpoint = Fraction(0)
    for n in range(order + 1):
        # z**n = (1+u)**(2*n)/(16**n*u**n) for u=exp(i*theta).
        weight = Fraction(comb(2 * n, n), (2 * n + 1) * 64**n)
        endpoint += weight * 4**n
        for k in range(min(n, degree) + 1):
            chebyshev[k] += weight * comb(2 * n, n + k) * (2 if k else 1)

    power = [Fraction(0)] * (degree + 1)
    previous: list[int] = []
    current = [1]
    for k, value in enumerate(chebyshev):
        for i, term in enumerate(current):
            power[i] += value * term
        if k == degree:
            break
        following = [0] * (len(current) + 1)
        for i, term in enumerate(current):
            following[i] -= 2 * term
            following[i + 1] += 16 * term
        for i, term in enumerate(previous):
            following[i] -= term
        if not previous:
            # T_1(t)=t; the recurrence applies from T_2 onward.
            following = [-1, 8]
        previous, current = current, following

    scale = 1 << fraction_bits
    words = tuple(round(value * scale) for value in power)
    rounding = sum(
        (
            abs(value - Fraction(word, scale)) / 4**i
            for i, (value, word) in enumerate(zip(power, words, strict=True))
        ),
        Fraction(0),
    )
    projection = endpoint - sum(chebyshev)
    series_tail = Fraction(1, 3 * 4**order)
    return words[::-1], projection + series_tail + rounding
