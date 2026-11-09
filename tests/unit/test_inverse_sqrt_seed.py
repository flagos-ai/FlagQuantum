"""Exact initial estimates, clean measured channels, and append round trips."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import floor, sqrt

import pytest
import torch

from flagquantum.compiler.registers.fixed_point.inverse_sqrt_seed import (
    inverse_sqrt_seed_ir,
    inverse_sqrt_seed_requirements,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n, f, b, extra=0):
    o = b + f // 2 + 1 + extra
    work, slots = inverse_sqrt_seed_requirements(
        n, source_fraction_bits=f, result_fraction_bits=b
    )
    return (
        tuple(range(n)),
        tuple(range(n, n + o)),
        tuple(range(n + o, n + o + work)),
        tuple(range(slots)),
    )


def oracle(word, f, b):
    if word == 0:
        return 0
    value = Fraction(word, 1 << f)
    power = Fraction(1)
    while value * power**2 < Fraction(1, 2):
        power *= 2
    while value * power**2 >= 2:
        power /= 2
    scaled = value * power**2
    seed = power * (3 - scaled) / 2
    assert Fraction(1, 2) <= value * seed**2 <= 1
    return floor(seed * (1 << b))


@pytest.mark.parametrize(
    "n,f,b,extra",
    [
        (1, 0, 0, 0),
        (2, 2, 1, 0),
        (3, 1, 0, 1),
        (4, 2, 3, 0),
        (5, 0, 7, 0),
        (7, 4, 5, 0),
    ],
)
def test_seed_all_words_histories_and_resource_counts(n, f, b, extra):
    bound = layout(n, f, b, extra)
    width, work = len(bound[1]), len(bound[2])
    gates = inverse_sqrt_seed_ir(*bound, source_fraction_bits=f, result_fraction_bits=b)
    # An exact controlled add/subtract computes one AND per suffix bit
    # and per carry of the shifted accumulator.
    k_min, k_max = (f - n + 1) // 2, f // 2
    shifted_widths = (2 + f + k_max - 3 * ((f - n + 1 + i) // 2) for i in range(n))
    events = 2 * sum(n - i + width - 1 for i, width in enumerate(shifted_widths))
    assert work == 2 * n + 2 * (max(b, f + 1 - 3 * k_min) + k_max + 1)
    counts = Counter(g.name for g in gates)
    assert counts["ccx"] == 2 * n
    assert counts["measure"] == counts["reset"] == events
    assert counts["t"] + counts["tdg"] == 4 * events

    def bits(word, width):
        return tuple(word >> i & 1 for i in reversed(range(width)))

    before, after = [], []
    for word in range(1 << n):
        initial = (1 << width) - 1 if word % 2 else 0
        seed = oracle(word, f, b)
        prefix, clean = bits(word, n), (0,) * work
        before.append(prefix + bits(initial, width) + clean)
        after.append(prefix + bits(initial ^ seed, width) + clean)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


@pytest.mark.parametrize("n,f,b", [(1, 0, 0), (2, 2, 2), (3, 1, 2), (4, 0, 5)])
def test_seed_coherent_spectator_and_clean_erase(n, f, b):
    bound = layout(n, f, b)
    output, work = len(bound[1]), len(bound[2])
    width = n + output + work + 1
    gates = inverse_sqrt_seed_ir(*bound, source_fraction_bits=f, result_fraction_bits=b)
    before, after = {}, {}
    for word, spectator in product(range(1 << n), (0, 1)):
        i = (word << (output + work + 1)) | spectator
        j = i | (oracle(word, f, b) << (work + 1))
        before[i] = after[j] = exp(1j * (word - 0.3 * word * spectator)) / sqrt(
            2 ** (n + 1)
        )
    for initial, expected in ((before, after), (after, before)):
        for seed in (11, 67, 91):
            actual = execute_sparse_register(
                CircuitIR(width, gates), initial, seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in actual.keys() | expected.keys()
            )
        if n == 1:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            target = torch.zeros_like(dense)
            for i, amplitude in initial.items():
                dense[i] = amplitude
            for i, amplitude in expected.items():
                target[i] = amplitude
            events = sum(g.name == "measure" for g in gates)
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history),
                    target / sqrt(2**events),
                    atol=2e-12,
                    rtol=0,
                )


def test_seed_append_round_trip():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(1, 0, 0)
    work = len(bound[2])
    width = 2 + work
    gates = inverse_sqrt_seed_ir(*bound, source_fraction_bits=0, result_fraction_bits=0)
    state = torch.zeros(1 << width, dtype=torch.complex128)
    i = 1 << (work + 1)
    state[0], state[i] = 0.8, 0.6j
    expected = state.clone()
    expected[i], expected[i | (1 << work)] = 0, 0.6j
    circuit = DynamicCircuit(width, inputs=state, dtype=torch.complex128)
    operation = FieldOperation(width, gates)
    circuit.append(operation)
    forward = run_dynamic(circuit, shots=4, seed=9, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, expected.expand(4, -1), atol=2e-12, rtol=0
    )
    circuit.append(operation)
    reverse = run_dynamic(circuit, shots=4, seed=9, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, state.expand(4, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"source": ()}, "valid source format"),
        ({"source_fraction_bits": 4}, "valid source format"),
        ({"result_fraction_bits": -1}, "nonnegative result precision"),
        ({"result": (8,)}, "more result integer bits"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": ()}, "work wires"),
        ({"classical_bits": tuple(range(-1, 7))}, "distinct nonnegative slots"),
        ({"classical_bits": (0,) * 8}, "distinct nonnegative slots"),
        ({"result": (0, 3, 4)}, "distinct nonnegative wires"),
    ],
)
def test_seed_rejects_invalid_layout(change, match):
    names = ("source", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(3, 1, 2), strict=True))
    config = {"source_fraction_bits": 1, "result_fraction_bits": 2}
    with pytest.raises(ValueError, match=match):
        inverse_sqrt_seed_ir(**(bound | config | change))
