"""Appendix C2 initial constants and the tuned square-root paths."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from math import floor, sqrt

import pytest

from flagquantum.compiler.registers.fixed_point_sqrt import fixed_point_sqrt_ir
from flagquantum.compiler.registers.inverse_sqrt import inverse_sqrt_ir
from flagquantum.compiler.registers.inverse_sqrt_seed import inverse_sqrt_seed_ir
from flagquantum.core.ir import CircuitIR
from tests.unit.algorithms.primitives.multiplexers.test_thc_qrom import _execute_sparse
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_fixed_point_sqrt import layout as sqrt_layout
from tests.unit.test_inverse_sqrt import layout as inverse_layout
from tests.unit.test_inverse_sqrt_seed import layout as seed_layout

pytestmark = pytest.mark.unit


def tuned_oracle(word, f, b, steps=0):
    if not word:
        return 0
    a = Fraction(word, 1 << f)
    k = (f - word.bit_length() + 1) // 2
    constant = Fraction("1.613" if k < 0 else "1.62" if k > 0 else "1.5")
    power = Fraction(2) ** k
    scale = 1 << b
    estimate = floor(scale * power * (constant - a * power**2 / 2))
    assert a * Fraction(estimate, scale) ** 2 <= Fraction(81, 64)
    for index in range(steps):
        next_word = floor(estimate * (3 - a * Fraction(estimate, scale) ** 2) / 2)
        assert next_word > 0
        assert a * Fraction(next_word, scale) ** 2 <= 1
        if index:
            assert next_word >= estimate
        estimate = next_word
    return estimate


def assert_channel(bound, gates, outputs):
    source, result, workspace, _ = bound

    def bits(word, width):
        return tuple(word >> i & 1 for i in reversed(range(width)))

    before, after = [], []
    for word, out in enumerate(outputs):
        occupied = (1 << len(result)) - 1 if word % 2 else 0
        prefix, clean = bits(word, len(source)), (0,) * len(workspace)
        before.append(prefix + bits(occupied, len(result)) + clean)
        after.append(prefix + bits(out ^ occupied, len(result)) + clean)
    count = Counter(g.name for g in gates)
    assert count["measure"] == count["reset"]
    assert certified_and_channel_map(gates, tuple(before)) == (
        tuple(after),
        count["measure"],
    )
    assert certified_and_channel_map(gates, tuple(after)) == (
        tuple(before),
        count["measure"],
    )


@pytest.mark.parametrize(
    "n,f,b", [(1, 0, 0), (2, 2, 0), (3, 0, 5), (5, 2, 4), (7, 5, 8)]
)
def test_tuned_seed_quantization_all_inputs_and_gate_delta(n, f, b):
    bound = seed_layout(n, f, b)
    settings = {"source_fraction_bits": f, "result_fraction_bits": b}
    gates = inverse_sqrt_seed_ir(*bound, **settings, tuned=True)
    default = inverse_sqrt_seed_ir(*bound, **settings)
    assert default == inverse_sqrt_seed_ir(*bound, **settings, tuned=False)
    assert tuple(g for g in gates if g.name != "cx") == tuple(
        g for g in default if g.name != "cx"
    )
    assert_channel(bound, gates, (tuned_oracle(w, f, b) for w in range(1 << n)))


@pytest.mark.parametrize("root", [False, True])
@pytest.mark.parametrize("n,f,b,steps,o", [(3, 0, 5, 1, 5), (3, 3, 4, 2, 5)])
def test_tuned_iterations_all_words_and_clean_history(root, n, f, b, steps, o):
    bind = sqrt_layout if root else inverse_layout
    emit = fixed_point_sqrt_ir if root else inverse_sqrt_ir
    bound = bind(n, f, b, steps, o)
    settings = {"source_fraction_bits": f, "precision": b, "iterations": steps}
    gates = emit(*bound, **settings, tuned=True)
    default = emit(*bound, **settings)
    assert tuple(g for g in gates if g.name != "cx") == tuple(
        g for g in default if g.name != "cx"
    )

    def output(word):
        estimate = tuned_oracle(word, f, b, steps)
        if not root:
            return estimate >> (b + f // 2 + 1 - o)
        a = Fraction(word, 1 << f)
        out_scale = 1 << (o - (n - f + 1) // 2)
        result = floor(a * estimate * out_scale / (1 << b))
        assert Fraction(result, out_scale) ** 2 <= a
        return result

    outputs = tuple(output(word) for word in range(1 << n))
    assert_channel(bound, gates, outputs)
    work = len(bound[2])
    ir = CircuitIR(n + o + work + 1, gates)
    before, after = {}, {}
    for word, out in enumerate(outputs):
        for spectator in (0, 1):
            i = (word << (o + work + 1)) | spectator
            j = i | (out << (work + 1))
            before[i] = after[j] = exp(0.3j * word * (1 + spectator)) / sqrt(
                2 ** (n + 1)
            )
    for initial, expected in ((before, after), (after, before)):
        actual = _execute_sparse(ir, initial, seed=31)
        assert all(
            abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
            for i in actual.keys() | expected.keys()
        )


def test_tuned_overshoot_witnesses_return_to_lower_bound():
    assert tuned_oracle(4, 0, 5) == 17
    assert tuned_oracle(4, 0, 5, 1) == 15
    assert tuned_oracle(2, 3, 5) == 71
    assert tuned_oracle(2, 3, 5, 1) == 62


@pytest.mark.parametrize(
    "emit,bind", [(inverse_sqrt_ir, inverse_layout), (fixed_point_sqrt_ir, sqrt_layout)]
)
def test_tuned_wrapper_requires_one_newton_update(emit, bind):
    with pytest.raises(ValueError, match="at least one Newton update"):
        emit(
            *bind(1, 0, 1, 0, 1),
            source_fraction_bits=0,
            precision=1,
            iterations=0,
            tuned=True,
        )
    with pytest.raises(ValueError, match="tuning must be boolean"):
        emit(
            *bind(1, 0, 1, 1, 1),
            source_fraction_bits=0,
            precision=1,
            iterations=1,
            tuned=1,
        )


def test_seed_rejects_nonboolean_tuning():
    with pytest.raises(ValueError, match="tuning must be boolean"):
        inverse_sqrt_seed_ir(
            *seed_layout(1, 0, 0),
            source_fraction_bits=0,
            result_fraction_bits=0,
            tuned=1,
        )
