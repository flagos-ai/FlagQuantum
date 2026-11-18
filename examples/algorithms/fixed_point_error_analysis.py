# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 error_analysis_for_fxp_arithmetic.ipynb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/error_analysis_for_fxp_arithmetic.ipynb

"""Notebook D4/D5/D6 arithmetic with explicit unsigned stored integers.

This is scalar error analysis. It neither emits gates nor predicts a quantum
circuit's primitive cost. The original fxpmath object stays at the reference
boundary. Shifts and each fixed-width operation retain truncation/saturation.
"""

import numpy as np

from flagquantum.simulation.fixed_point_encoding import fixed_point_real_value


def fxp(x: float, d: int, *, d_word: int | None = None) -> int:
    """Return the unsigned stored word with the original scalar input guard."""
    assert 0 <= x < 1
    width = d if d_word is None else d_word
    if type(d) is not int or type(width) is not int or d < 0 or width < 1:
        raise ValueError(
            "Unsigned analysis requires a positive word and nonnegative fraction"
        )
    return min(int(x * 2**d), (1 << width) - 1)


def assert_allclose(x: float, y: float, eps: float) -> None:
    """Retain NumPy's original default relative tolerance alongside absolute eps."""
    np.testing.assert_allclose(x, y, atol=eps)


def get_bitsize_for_fxp_mul_with_integer(eps: float, integer_width: int) -> int:
    """Return the original Equation D7 bound."""
    return integer_width + int(np.ceil(np.log2(integer_width / eps)))


def mul_with_int_via_repeated_add(
    a: float, b: int, real_width: int, integer_width: int
) -> float:
    """Retain shifted inputs and per-addend conversion to real_width-integer_width fractions."""
    word = fxp(a, real_width, d_word=real_width + integer_width - 1)
    result = 0
    maximum = (1 << real_width) - 1
    for i in range(integer_width):
        term = (word << i) >> integer_width
        result = min(result + term * ((b >> i) & 1), maximum)
    return float(fixed_point_real_value(result, real_width, real_width - integer_width))


def get_bitsize_for_fxp_mul(eps: float) -> int:
    """Keep Equation D17's original ceil placement and final integer truncation."""
    return int(np.ceil(1 + np.log2(1 / eps)) + np.log2(1 + np.log2(1 / eps)))


def mul_via_repeated_add(a: float, b: float, d: int) -> float:
    """Retain the original MSB-first traversal, excluding b's final bit."""
    left, right = fxp(a, d), fxp(b, d)
    result = 0
    maximum = (1 << d) - 1
    for i in range(d - 1):
        left >>= 1
        if (right >> (d - 1 - i)) & 1:
            result = min(result + left, maximum)
    return float(fixed_point_real_value(result, d, d))


def get_bitsize_for_fxp_square(eps: float) -> int:
    """Return the original Equation D36 bound, including its failing samples."""
    return int(np.ceil(np.log2(1 / eps) + np.log2(11 / 3 + np.log2(1 / eps))))


def _multiply_bit(word: int, bit: int, width: int) -> int:
    # Same-width constant conversion saturates integer one to 1 - 2**(-width).
    constant = min(bit << width, (1 << width) - 1)
    return (word * constant) >> width


def square_via_repeated_add(a: float, d: int) -> float:
    """Follow the original two loops, evolving mask and even-width tail."""
    word = fxp(a, d)
    bits = [(word >> (d - 1 - i)) & 1 for i in range(d)]
    result = 0
    one = fxp(0.5, d)
    maximum = (1 << d) - 1
    for n in range(d // 2, d - 1):
        result = min(result + _multiply_bit(word >> n, bits[n], d), maximum)
    mask = 0
    for n in range((d - 1) // 2):
        right = _multiply_bit(one >> (2 * n + 1), bits[n], d)
        term = _multiply_bit(min(((word & mask) >> n) + right, maximum), bits[n], d)
        result = min(result + term, maximum)
        mask |= one >> n
    if not d & 1:
        n = d // 2 - 1
        result = min(result + _multiply_bit((word & mask) >> n, bits[n], d), maximum)
    return float(fixed_point_real_value(result, d, d))
