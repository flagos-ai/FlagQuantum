"""Appendix D nonnegative dataflow and coherent branch cleanup."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import asin, floor, pi, sqrt

import pytest

from flagquantum.compiler.registers.fixed_point.fixed_point_arcsine import (
    fixed_point_arcsine_ir,
    fixed_point_arcsine_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_fixed_point_polynomial import oracle as poly_oracle
from tests.unit.test_fixed_point_sqrt import oracle as sqrt_oracle
from tests.unit.test_inverse_sqrt_tuned import tuned_oracle

pytestmark = pytest.mark.unit


def layout(f, p, out, degree, b, s):
    work, slots = fixed_point_arcsine_requirements(
        f, p, out, degree, precision=b, iterations=s
    )
    return (
        tuple(range(f + 1)),
        tuple(range(f + 1, f + out + 1)),
        tuple(range(f + out + 1, f + out + 1 + work)),
        tuple(range(slots)),
    )


def oracle(value, f, p, out, coeffs, b, s, tuned):
    x = Fraction(value, 1 << f)
    low = x < Fraction(1, 2)
    reduced = x * x if low else (1 - x) / 2
    reduced_word = int(reduced * (1 << (2 * f)))
    poly = Fraction(poly_oracle(reduced_word, 2 * f, p + 2, coeffs), 1 << p)
    assert 0 <= poly < 2
    q = max(f, b)
    if low:
        factor = x
    elif tuned:
        seed = Fraction(tuned_oracle(reduced_word, 2 * f, b, s), 1 << b)
        factor = Fraction(floor(reduced * seed * (1 << q)), 1 << q)
    else:
        factor = Fraction(sqrt_oracle(reduced_word, 2 * f, 2 * f, b, s, q), 1 << q)
    scale = 1 << (out - 1)
    term = floor((1 if low else 2) * factor * poly * scale)
    return term if low else floor(pi / 2 * scale) - term


def emit(bound, p, out, coeffs, b, s, tuned=False):
    return fixed_point_arcsine_ir(
        *bound,
        coefficients=coeffs,
        polynomial_fraction_bits=p,
        half_pi_word=floor(pi / 2 * (1 << (out - 1))),
        precision=b,
        iterations=s,
        tuned=tuned,
    )


@pytest.mark.parametrize(
    "f,p,out,coeffs,b,s,tuned",
    [
        (1, 1, 3, (2,), 1, 0, False),
        (2, 2, 5, (1, 4), 2, 1, False),
        (3, 3, 7, (1, 8), 2, 0, False),
        (3, 4, 8, (3, 16), 4, 1, True),
        (4, 4, 12, (1, 3, 16), 3, 1, False),
    ],
)
def test_arcsine_all_nonnegative_inputs_and_histories(f, p, out, coeffs, b, s, tuned):
    bound = layout(f, p, out, len(coeffs) - 1, b, s)
    gates = emit(bound, p, out, coeffs, b, s, tuned)
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    assert events == counts["reset"]
    assert counts["t"] + counts["tdg"] == 4 * events

    def bits(word, width):
        return tuple(word >> j & 1 for j in reversed(range(width)))

    before, after = [], []
    for value in range((1 << f) + 1):
        expect = oracle(value, f, p, out, coeffs, b, s, tuned)
        assert 0 <= expect < 1 << out
        assert abs(expect / (1 << (out - 1)) - asin(value / (1 << f))) < 0.33
        occupied = ((1 << out) - 1) if value % 2 else 0
        prefix, clean = bits(value, f + 1), (0,) * len(bound[2])
        before.append(prefix + bits(occupied, out) + clean)
        after.append(prefix + bits(occupied ^ expect, out) + clean)
    assert oracle(0, f, p, out, coeffs, b, s, tuned) == 0
    assert oracle(1 << f, f, p, out, coeffs, b, s, tuned) == floor(
        pi / 2 * (1 << (out - 1))
    )
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


@pytest.mark.parametrize("tuned", [False, True])
def test_arcsine_both_branches_coherent_with_spectator(tuned):
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    f, p, out, coeffs, b, s = 2, 3, 6, (1, 8), 3, 1
    bound = layout(f, p, out, len(coeffs) - 1, b, s)
    gates = emit(bound, p, out, coeffs, b, s, tuned)
    work = len(bound[2])
    wires = f + out + work + 2
    ir = DynamicCircuit(wires).append(FieldOperation(wires, gates)).to_ir()
    assert ir.instructions == gates
    before, after = {}, {}
    for word, spectator in product(range((1 << f) + 1), (0, 1)):
        output = oracle(word, f, p, out, coeffs, b, s, tuned)
        i = (word << (out + work + 1)) | spectator
        j = i | (output << (work + 1))
        before[i] = after[j] = exp(0.3j * word * (1 + spectator)) / sqrt(10)
    for initial, expected in ((before, after), (after, before)):
        for seed in (7, 37):
            actual = execute_sparse_register(ir, initial, seed=seed)
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in actual.keys() | expected.keys()
            )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"polynomial_fraction_bits": -1}, "polynomial precision"),
        ({"result": ()}, "result integer bit"),
        ({"half_pi_word": 16}, "half-pi word"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": ()}, "work wires"),
        ({"result": (0, 5, 6, 7)}, "distinct nonnegative wires"),
    ],
)
def test_arcsine_invalid_layout(change, match):
    names = ("source", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(2, 2, 4, 0, 2, 0), strict=True))
    with pytest.raises(ValueError, match=match):
        fixed_point_arcsine_ir(
            **(
                bound
                | {
                    "coefficients": (4,),
                    "polynomial_fraction_bits": 2,
                    "half_pi_word": 12,
                    "precision": 2,
                    "iterations": 0,
                }
                | change
            )
        )


def test_arcsine_rejects_duplicate_slots():
    source, result, work, slots = layout(2, 2, 4, 0, 2, 0)
    with pytest.raises(ValueError, match="distinct nonnegative slots"):
        emit((source, result, work, (0,) * len(slots)), 2, 4, (4,), 2, 0)
