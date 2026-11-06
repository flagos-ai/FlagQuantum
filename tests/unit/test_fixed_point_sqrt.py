"""Square root multiply-back, full channel and shared scratch checks."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import floor, sqrt

import pytest
import torch

from flagquantum.compiler.registers.fixed_point_sqrt import (
    fixed_point_sqrt_ir,
    fixed_point_sqrt_requirements,
)
from flagquantum.compiler.registers.inverse_sqrt import (
    inverse_sqrt_ir,
    inverse_sqrt_requirements,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.algorithms.primitives.multiplexers.test_thc_qrom import _execute_sparse
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_inverse_sqrt import oracle as inverse_oracle

pytestmark = pytest.mark.unit


def layout(n, f, b, s, o):
    work, slots = fixed_point_sqrt_requirements(
        n, source_fraction_bits=f, precision=b, iterations=s
    )
    return (
        tuple(range(n)),
        tuple(range(n, n + o)),
        tuple(range(n + o, n + o + work)),
        tuple(range(slots)),
    )


def oracle(word, n, f, b, s, o):
    a = Fraction(word, 1 << f)
    inv = Fraction(inverse_oracle(word, f, b, s, b + f // 2 + 1), 1 << b)
    scale = 1 << (o - (n - f + 1) // 2)
    out = floor(a * inv * scale)
    assert Fraction(out, scale) ** 2 <= a
    assert 0 <= a * inv - Fraction(out, scale) < Fraction(1, scale)
    return out


@pytest.mark.parametrize(
    "n,f,b,s,o",
    [
        (1, 0, 1, 0, 1),
        (2, 2, 2, 1, 2),
        (3, 1, 2, 2, 2),
        (4, 0, 3, 2, 5),
        (4, 3, 3, 0, 9),
        (5, 0, 4, 0, 3),
    ],
)
def test_multiply_back_preserves_history_schedule_and_all_branches(n, f, b, s, o):
    bound = layout(n, f, b, s, o)
    source, result, workspace, bits = bound
    settings = {"source_fraction_bits": f, "precision": b, "iterations": s}
    gates = fixed_point_sqrt_ir(*bound, **settings)
    estimate_width = b + f // 2 + 1
    history_width = (s + 1) * estimate_width
    extra_events = 2 * n * estimate_width - max(n, estimate_width)
    inv_work, inv_slots = inverse_sqrt_requirements(n, **settings)
    assert len(workspace) == max(inv_work, history_width + extra_events)
    assert len(bits) == inv_slots
    inverse_result = tuple(
        range(n + o + len(workspace), n + o + len(workspace) + estimate_width)
    )
    inverse = inverse_sqrt_ir(
        source, inverse_result, workspace[:inv_work], bits, **settings
    )
    copy_start = next(
        i for i, gate in enumerate(inverse) if gate.wires[-1] in inverse_result
    )
    after_copy = inverse[copy_start + estimate_width :]
    assert gates[:copy_start] == inverse[:copy_start]
    assert gates[-len(after_copy) :] == after_copy
    count = Counter(g.name for g in gates)
    base_count = Counter(g.name for g in inverse)
    assert count["measure"] - base_count["measure"] == extra_events
    assert count["measure"] == count["reset"]
    assert count["t"] + count["tdg"] == 4 * count["measure"]
    assert count["ccx"] == base_count["ccx"]

    def word(value, width):
        return tuple(value >> i & 1 for i in reversed(range(width)))

    before, after = [], []
    for value in range(1 << n):
        occupied = (1 << o) - 1 if value % 2 else 0
        prefix, clean = word(value, n), (0,) * len(workspace)
        before.append(prefix + word(occupied, o) + clean)
        after.append(prefix + word(occupied ^ oracle(value, n, f, b, s, o), o) + clean)
    assert certified_and_channel_map(gates, tuple(before)) == (
        tuple(after),
        count["measure"],
    )
    assert certified_and_channel_map(gates, tuple(after)) == (
        tuple(before),
        count["measure"],
    )


@pytest.mark.parametrize(
    "n,f,b,s,o", [(1, 0, 1, 1, 2), (2, 2, 2, 2, 3), (3, 1, 2, 1, 2)]
)
def test_square_root_complex_trajectory_and_clean_history(n, f, b, s, o):
    bound = layout(n, f, b, s, o)
    work = len(bound[2])
    gates = fixed_point_sqrt_ir(
        *bound, source_fraction_bits=f, precision=b, iterations=s
    )
    ir = CircuitIR(n + o + work + 1, gates)
    before, after = {}, {}
    for word, spectator in product(range(1 << n), (0, 1)):
        i = (word << (o + work + 1)) | spectator
        j = i | (oracle(word, n, f, b, s, o) << (work + 1))
        before[i] = after[j] = exp(1j * word * (1 + 0.3 * spectator)) / sqrt(
            2 ** (n + 1)
        )
    for initial, expected in ((before, after), (after, before)):
        for seed in (7, 21, 91):
            actual = _execute_sparse(ir, initial, seed=seed)
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in actual.keys() | expected.keys()
            )


def test_square_root_append_and_restore():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(1, 0, 1, 0, 1)
    work = len(bound[2])
    width = 2 + work
    gates = fixed_point_sqrt_ir(
        *bound, source_fraction_bits=0, precision=1, iterations=0
    )
    state = torch.zeros(1 << width, dtype=torch.complex128)
    i = 1 << (work + 1)
    state[0], state[i] = 0.8, 0.6j
    expected = state.clone()
    expected[i], expected[i | (1 << work)] = 0, 0.6j
    circuit = DynamicCircuit(width, inputs=state, dtype=torch.complex128)
    op = FieldOperation(width, gates)
    circuit.append(op)
    forward = run_dynamic(circuit, shots=4, seed=11, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, expected.expand(4, -1), atol=2e-12, rtol=0
    )
    circuit.append(op)
    reverse = run_dynamic(circuit, shots=4, seed=11, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, state.expand(4, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"result": ()}, "keep all integer bits"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": ()}, "work wires"),
        ({"classical_bits": (0,) * 8}, "distinct nonnegative slots"),
        ({"classical_bits": tuple(range(-1, 7))}, "distinct nonnegative slots"),
        ({"result": (0, 3)}, "distinct nonnegative wires"),
    ],
)
def test_square_root_rejects_invalid_layout(change, match):
    names = ("source", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(3, 1, 2, 1, 2), strict=True))
    settings = {"source_fraction_bits": 1, "precision": 2, "iterations": 1}
    with pytest.raises(ValueError, match=match):
        fixed_point_sqrt_ir(**(bound | settings | change))
