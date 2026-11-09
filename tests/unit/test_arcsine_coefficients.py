"""Exact coefficient reproduction and encoded signed ArcSin verification."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from math import asin, comb, sqrt

import pytest

from flagquantum.compiler.registers.fixed_point.arcsine_coefficients import (
    arcsine_polynomial_coefficients,
)
from flagquantum.compiler.registers.fixed_point.signed_arcsine import signed_arcsine_ir
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_fixed_point_arcsine import oracle as unsigned_oracle
from tests.unit.test_fixed_point_polynomial import oracle as polynomial_oracle
from tests.unit.test_signed_arcsine import layout, options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "degree,p,expected,limit",
    [
        (0, 0, (1,), Fraction(5, 100)),
        (3, 24, (1090080, 1199891, 2799260, 16777191), Fraction(17, 10**7)),
        (4, 18, (13240, 10413, 19783, 43687, 262144), Fraction(54, 10**8)),
        (4, 24, (847339, 666410, 1266089, 2795951, 16777217), Fraction(112, 10**9)),
        (
            5,
            24,
            (713931, 401133, 764018, 1257374, 2796223, 16777216),
            Fraction(13, 10**9),
        ),
    ],
)
def test_arcsine_factor_profile_and_exact_series_check(degree, p, expected, limit):
    words, bound = arcsine_polynomial_coefficients(p, degree=degree)
    assert words == expected
    assert 0 < bound < limit
    coefficients = tuple(Fraction(word, 1 << p) for word in reversed(words))
    for j in range(129):
        z = Fraction(j, 512)
        # Check against both ends of an independent rational series enclosure.
        reference = sum(
            (Fraction(comb(2 * n, n), (2 * n + 1) * 4**n) * z**n for n in range(33)),
            Fraction(0),
        )
        tail = z**33 / (1 - z)
        value = sum((c * z**n for n, c in enumerate(coefficients)), Fraction(0))
        assert max(abs(value - reference), abs(value - reference - tail)) <= bound


def test_generated_polynomial_in_signed_arcsine():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    f, p, out, b, steps = 3, 12, 10, 7, 2
    words, factor_error = arcsine_polynomial_coefficients(p)
    bound = layout(f, p, out, len(words) - 1, b, steps)
    source, _, work, _ = bound
    gates = signed_arcsine_ir(*bound, **options(p, out, words, b, steps))
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    assert counts["reset"] == events
    assert counts["t"] + counts["tdg"] == 4 * events

    def bits(word, width):
        return tuple(word >> j & 1 for j in reversed(range(width)))

    before, after = [], []
    fixed_point_error = 0.0
    angle_error = 0.0
    for value in range(-(1 << f), (1 << f) + 1):
        x = Fraction(abs(value), 1 << f)
        reduced = x * x if x < Fraction(1, 2) else (1 - x) / 2
        word = int(reduced * (1 << (2 * f)))
        poly = polynomial_oracle(word, 2 * f, p + 2, words)
        polynomial_value = sum(
            (Fraction(c, 1 << p) * reduced**j for j, c in enumerate(reversed(words))),
            Fraction(0),
        )
        fixed_point_error = max(
            fixed_point_error, float(abs(polynomial_value - Fraction(poly, 1 << p)))
        )
        result = unsigned_oracle(abs(value), f, p, out - 1, words, b, steps, False)
        signed = -result if value < 0 else result
        angle_error = max(
            angle_error, abs(signed / (1 << (out - 2)) - asin(value / (1 << f)))
        )
        occupied = (1 << out) - 1 if value % 2 else 0
        prefix = bits(value, len(source))
        before.append(prefix + bits(occupied, out) + (0,) * len(work))
        after.append(prefix + bits(occupied ^ signed, out) + (0,) * len(work))
    assert float(factor_error) < 2.75e-5
    assert fixed_point_error < 0.002
    assert angle_error < 0.02
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)

    def basis(word):
        return int("".join(map(str, word)), 2)

    n = len(before[0])
    ir = DynamicCircuit(n).append(FieldOperation(n, gates)).to_ir()
    selected = (0, 3, 7, 8, 11, 13, 16)
    initial = {basis(before[j]): exp(0.31j * j) / sqrt(len(selected)) for j in selected}
    expected = {basis(after[j]): exp(0.31j * j) / sqrt(len(selected)) for j in selected}
    actual = execute_sparse_register(ir, initial, seed=71)
    assert all(
        abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
        for i in actual.keys() | expected.keys()
    )


@pytest.mark.parametrize(
    "precision,degree,match",
    [
        (-1, 4, "precision"),
        (True, 4, "precision"),
        (12, -1, "degree"),
        (12, True, "degree"),
    ],
)
def test_arcsine_coefficient_invalid_format(precision, degree, match):
    with pytest.raises(ValueError, match=match):
        arcsine_polynomial_coefficients(precision, degree=degree)
