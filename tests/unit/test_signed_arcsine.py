"""Coherent signed arcsine wrapping and full measured uncomputation."""

from cmath import exp
from collections import Counter
from itertools import product
from math import floor, pi, sqrt

import pytest

from flagquantum.compiler.registers.fixed_point_arcsine import fixed_point_arcsine_ir
from flagquantum.compiler.registers.signed_arcsine import (
    signed_arcsine_ir,
    signed_arcsine_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_fixed_point_arcsine import layout as unsigned_layout
from tests.unit.test_fixed_point_arcsine import oracle as unsigned_oracle

pytestmark = pytest.mark.unit


def layout(f, p, out, degree, b, steps):
    n = f + 2
    work, slots = signed_arcsine_requirements(
        f, p, out, degree, precision=b, iterations=steps
    )
    return (
        tuple(range(n)),
        tuple(range(n, n + out)),
        tuple(range(n + out, n + out + work)),
        tuple(range(slots)),
    )


def options(p, out, coeffs, b, steps, tuned=False):
    return {
        "coefficients": coeffs,
        "polynomial_fraction_bits": p,
        "half_pi_word": floor(pi / 2 * (1 << (out - 2))),
        "precision": b,
        "iterations": steps,
        "tuned": tuned,
    }


@pytest.mark.parametrize(
    "f,p,out,coeffs,b,steps,tuned",
    [
        (1, 1, 4, (2,), 1, 0, False),
        (1, 1, 20, (2,), 1, 0, False),
        (2, 2, 6, (1, 4), 2, 1, False),
        (3, 4, 9, (3, 16), 4, 1, True),
    ],
)
def test_signed_endpoints_oddness_and_every_measurement_history(
    f, p, out, coeffs, b, steps, tuned
):
    config = options(p, out, coeffs, b, steps, tuned)
    bound = layout(f, p, out, len(coeffs) - 1, b, steps)
    gates = signed_arcsine_ir(*bound, **config)
    unsigned = fixed_point_arcsine_ir(
        *unsigned_layout(f, p, out - 1, len(coeffs) - 1, b, steps), **config
    )
    counts, base = (Counter(g.name for g in stream) for stream in (gates, unsigned))
    events = base["measure"] + 2 * (f + out - 1)
    assert counts["measure"] == counts["reset"] == events
    assert counts["ccx"] == base["ccx"] + 2 * (out - 1)
    assert counts["t"] + counts["tdg"] == 4 * events

    def bits(word, size):
        return tuple(word >> i & 1 for i in reversed(range(size)))

    before, after = [], []
    bound_word = 1 << f
    for x in range(-bound_word, bound_word + 1):
        magnitude = unsigned_oracle(abs(x), f, p, out - 1, coeffs, b, steps, tuned)
        if x == 0:
            assert magnitude == 0
        if abs(x) == bound_word:
            assert magnitude == config["half_pi_word"]
        y = -magnitude if x < 0 else magnitude
        occupied = ((1 << out) - 1) if x % 2 else 0
        prefix, clean = bits(x, f + 2), (0,) * len(bound[2])
        before.append(prefix + bits(occupied, out) + clean)
        after.append(prefix + bits(occupied ^ y, out) + clean)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


def test_signed_arcsine_append_preserves_cross_sign_complex_amplitudes():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    f, p, out, coeffs, b, steps = 2, 2, 6, (1, 4), 2, 1
    bound = layout(f, p, out, 1, b, steps)
    work = len(bound[2])
    gates = signed_arcsine_ir(*bound, **options(p, out, coeffs, b, steps))
    qubits = f + 2 + out + work + 1
    ir = DynamicCircuit(qubits).append(FieldOperation(qubits, gates)).to_ir()
    assert ir.instructions == gates
    before, after = {}, {}
    for x, spectator in product(range(-4, 5), (0, 1)):
        magnitude = unsigned_oracle(abs(x), f, p, out - 1, coeffs, b, steps, False)
        y = (-magnitude if x < 0 else magnitude) % (1 << out)
        i = ((x % (1 << (f + 2))) << (out + work + 1)) | spectator
        j = i | (y << (work + 1))
        before[i] = after[j] = exp(0.3j * x * (1 + spectator)) / sqrt(18)
    for initial, expected in ((before, after), (after, before)):
        for seed in (7, 51, 89):
            actual = execute_sparse_register(ir, initial, seed=seed)
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in actual.keys() | expected.keys()
            )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"result": (7,)}, "result integer bit"),
        ({"workspace": ()}, "work qubits"),
        ({"classical_bits": ()}, "work qubits"),
        ({"result": (0, 4, 5, 6)}, "distinct nonnegative wires"),
    ],
)
def test_signed_invalid_layout(change, match):
    names = ("source", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(1, 1, 4, 0, 1, 0), strict=True))
    with pytest.raises(ValueError, match=match):
        signed_arcsine_ir(**(bound | options(1, 4, (2,), 1, 0) | change))


def test_signed_arcsine_rejects_duplicate_slots():
    source, result, work, slots = layout(1, 1, 4, 0, 1, 0)
    with pytest.raises(ValueError, match="distinct nonnegative slots"):
        signed_arcsine_ir(
            source, result, work, (0,) * len(slots), **options(1, 4, (2,), 1, 0)
        )
