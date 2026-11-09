"""Polynomial-size ArcTan approximation and native cleanup evidence."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import atan, floor, pi, sqrt

import pytest

from flagquantum.compiler.registers.fixed_point.arctan_arithmetic import (
    _normalized_arctan_coefficients,
    arctan_arithmetic_ir,
    arctan_arithmetic_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_fixed_point_polynomial import oracle as polynomial_oracle
from tests.unit.test_reciprocal import oracle as reciprocal_oracle

pytestmark = pytest.mark.unit


def layout(n, p, b, degree):
    work, slots = arctan_arithmetic_requirements(n, b, degree=degree)
    return (
        tuple(range(n)),
        n,
        tuple(range(n + 1, n + 1 + p)),
        tuple(range(n + 1 + p, n + 1 + p + work)),
        tuple(range(slots)),
    )


def oracle(x, p, b, degree):
    if x < 2:
        return x * (1 << (p - 1))
    r = reciprocal_oracle(x, 0, b, b)
    coefficients = tuple(
        round((2 / pi) * (-1) ** j * (1 << b) / (2 * j + 1))
        for j in range(degree, -1, -1)
    )
    poly = polynomial_oracle(r * r, 2 * b, b + 1, coefficients)
    assert 0 < poly < 1 << b
    complement = 1 - Fraction(poly * r, 1 << (2 * b))
    return floor(complement * (1 << p))


@pytest.mark.parametrize(
    "n,p,b,degree",
    [(1, 1, 1, 0), (2, 2, 2, 0), (3, 4, 5, 2), (4, 5, 7, 3), (5, 6, 8, 4)],
)
def test_arctan_arithmetic_all_inputs_clean_history(n, p, b, degree):
    bound = layout(n, p, b, degree)
    gates = arctan_arithmetic_ir(*bound, precision=b, degree=degree)
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    assert counts["reset"] == events
    assert counts["t"] + counts["tdg"] == 4 * events

    def bits(word, width):
        return tuple(word >> j & 1 for j in reversed(range(width)))

    before, after = [], []
    for x in range(1 << n):
        out = oracle(x, p, b, degree)
        for sign, target in product((0, 1), (0, (1 << p) - 1)):
            prefix, clean = bits(x, n), (0,) * len(bound[3])
            before.append(prefix + (sign,) + bits(target, p) + clean)
            after.append(prefix + (sign ^ bool(x),) + bits(target ^ out, p) + clean)
        if x >= 2:
            error = abs(out / (1 << p) - 2 * atan(x) / pi)
            assert error < 2.5 / (1 << p)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


def test_arctan_arithmetic_append_complex_trajectories():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    n, p, b, degree = 2, 3, 4, 2
    bound = layout(n, p, b, degree)
    gates = arctan_arithmetic_ir(*bound, precision=b, degree=degree)
    work = len(bound[3])
    qubits = n + 1 + p + work + 1
    ir = DynamicCircuit(qubits).append(FieldOperation(qubits, gates)).to_ir()
    before, after = {}, {}
    for x, s in product(range(1 << n), (0, 1)):
        occupied = (s << p) | (3 * x % (1 << p))
        i = ((x << (p + 1)) | occupied) << (work + 1) | s
        j = i ^ (((bool(x) << p) | oracle(x, p, b, degree)) << (work + 1))
        before[i] = after[j] = exp(0.3j * x * (1 + s)) / sqrt(8)
    for initial, desired in ((before, after), (after, before)):
        actual = execute_sparse_register(ir, initial, seed=19)
        assert all(
            abs(actual.get(i, 0) - desired.get(i, 0)) < 2e-12
            for i in actual.keys() | desired.keys()
        )


@pytest.mark.parametrize("p,degree", [(2, 0), (5, 3), (12, 6), (24, 8)])
def test_reproducible_normalized_coefficients(p, degree):
    assert _normalized_arctan_coefficients(p, degree) == tuple(
        round((2 / pi) * (-1) ** j * (1 << p) / (2 * j + 1))
        for j in range(degree, -1, -1)
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"target": ()}, "working precision"),
        ({"precision": 1}, "cover every"),
        ({"workspace": ()}, "work qubits"),
        ({"classical_bits": (0,) * 7}, "distinct nonnegative slots"),
        ({"sign": 0}, "distinct nonnegative wires"),
    ],
)
def test_arctan_arithmetic_invalid_layout(change, match):
    names = ("source", "sign", "target", "workspace", "classical_bits")
    bound = dict(zip(names, layout(2, 3, 3, 1), strict=True))
    with pytest.raises(ValueError, match=match):
        arctan_arithmetic_ir(**(bound | {"precision": 3, "degree": 1} | change))


@pytest.mark.ftqc_reference
def test_arithmetic_rounding_boundaries_against_locked_lookup():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    source = ArcTan(3, 4)
    differences = {}
    for x in range(8):
        _, sign, word = source.apply(x, 0, 0)
        assert sign == bool(x)
        delta = oracle(x, 4, 5, 2) - word
        if delta:
            differences[x] = delta
    assert differences == {2: 1, 3: 1, 5: 1}
