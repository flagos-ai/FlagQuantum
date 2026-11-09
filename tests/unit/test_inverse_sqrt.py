"""Complete inverse square root schedule and coherent cleanup."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import floor, sqrt

import pytest
import torch

from flagquantum.compiler.registers.fixed_point.inverse_sqrt import (
    inverse_sqrt_ir,
    inverse_sqrt_requirements,
)
from flagquantum.compiler.registers.fixed_point.inverse_sqrt_newton import (
    inverse_sqrt_newton_requirements,
    inverse_sqrt_newton_step_ir,
)
from flagquantum.compiler.registers.fixed_point.inverse_sqrt_seed import (
    inverse_sqrt_seed_ir,
    inverse_sqrt_seed_requirements,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n, f, b, s, o):
    work, slots = inverse_sqrt_requirements(
        n, source_fraction_bits=f, precision=b, iterations=s
    )
    return (
        tuple(range(n)),
        tuple(range(n, n + o)),
        tuple(range(n + o, n + o + work)),
        tuple(range(slots)),
    )


def oracle(word, f, b, steps, output_width):
    if not word:
        return 0
    a = Fraction(word, 1 << f)
    exponent = (f - word.bit_length() + 1) // 2
    power = Fraction(2) ** exponent
    exact = power * (3 - a * power**2) / 2
    scale = 1 << b
    current = Fraction(floor(exact * scale), scale)
    budget = Fraction(1, scale)
    assert current > 0
    for _ in range(steps):
        exact = exact * (3 - a * exact**2) / 2
        following = Fraction(floor(current * (3 - a * current**2) * scale / 2), scale)
        assert following >= current
        assert a * following**2 <= 1
        current = following
        budget = Fraction(3, 2) * budget + Fraction(1, scale)
        assert 0 <= exact - current < budget
    output_fraction = output_width - (f // 2 + 1)
    result = floor(current * (1 << output_fraction))
    shortened = Fraction(result, 1 << output_fraction)
    assert a * shortened**2 <= 1
    assert 0 <= exact - shortened < budget + Fraction(1, 1 << output_fraction)
    return result


@pytest.mark.parametrize(
    "n,f,b,s,o",
    [
        (1, 0, 1, 0, 1),
        (2, 2, 2, 2, 3),
        (3, 1, 2, 2, 2),
        (4, 0, 3, 3, 3),
        (5, 3, 4, 3, 5),
    ],
)
def test_inverse_sqrt_component_schedule_all_inputs_and_histories(n, f, b, s, o):
    bound = layout(n, f, b, s, o)
    source, result, work, slots = bound
    gates = inverse_sqrt_ir(*bound, source_fraction_bits=f, precision=b, iterations=s)
    width = f // 2 + 1 + b
    seed_work, seed_slots = inverse_sqrt_seed_requirements(
        n, source_fraction_bits=f, result_fraction_bits=b
    )
    step_work, step_slots = (
        inverse_sqrt_newton_requirements(
            n, width, source_fraction_bits=f, estimate_fraction_bits=b
        )
        if s
        else (0, 0)
    )
    history = tuple(work[i * width : (i + 1) * width] for i in range(s + 1))
    scratch = work[(s + 1) * width :]
    assert len(scratch) == max(seed_work, step_work)
    assert len(slots) == max(seed_slots, step_slots)
    seed = inverse_sqrt_seed_ir(
        source,
        history[0],
        scratch[:seed_work],
        slots[:seed_slots],
        source_fraction_bits=f,
        result_fraction_bits=b,
    )
    updates = tuple(
        inverse_sqrt_newton_step_ir(
            source,
            history[i],
            history[i + 1],
            scratch[:step_work],
            slots[:step_slots],
            source_fraction_bits=f,
            estimate_fraction_bits=b,
        )
        for i in range(s)
    )
    copy = tuple(
        Instruction("cx", (a, z)) for a, z in zip(history[-1][:o], result, strict=True)
    )
    stages = (*updates, copy, *reversed(updates))
    assert gates == (*seed, *(g for stage in stages for g in stage), *seed)

    counts = Counter(gate.name for gate in gates)
    events = counts["measure"]
    assert counts["reset"] == events
    assert counts["t"] + counts["tdg"] == 4 * events
    assert counts["ccx"] == 4 * n

    def bits(word, size):
        return tuple(word >> i & 1 for i in reversed(range(size)))

    before, after = [], []
    clean = (0,) * len(work)
    for word in range(1 << n):
        initial = (1 << o) - 1 if word % 2 else 0
        before.append(bits(word, n) + bits(initial, o) + clean)
        after.append(
            bits(word, n) + bits(initial ^ oracle(word, f, b, s, o), o) + clean
        )
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


@pytest.mark.parametrize(
    "n,f,b,s,o", [(1, 0, 1, 1, 2), (2, 2, 2, 2, 3), (3, 1, 2, 2, 2)]
)
def test_inverse_sqrt_complex_amplitudes_and_clean_history(n, f, b, s, o):
    bound = layout(n, f, b, s, o)
    work = len(bound[2])
    gates = inverse_sqrt_ir(*bound, source_fraction_bits=f, precision=b, iterations=s)
    ir = CircuitIR(n + o + work + 1, gates)
    before, after = {}, {}
    for word, spectator in product(range(1 << n), (0, 1)):
        i = (word << (o + work + 1)) | spectator
        j = i | (oracle(word, f, b, s, o) << (work + 1))
        before[i] = after[j] = exp(1j * word * (1 + 0.3 * spectator)) / sqrt(
            2 ** (n + 1)
        )
    for initial, expected in ((before, after), (after, before)):
        for seed in (7, 21, 91):
            actual = execute_sparse_register(ir, initial, seed=seed)
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in actual.keys() | expected.keys()
            )


def test_inverse_sqrt_seed_only_append_round_trip():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(1, 0, 1, 0, 1)
    work = len(bound[2])
    width = 2 + work
    gates = inverse_sqrt_ir(*bound, source_fraction_bits=0, precision=1, iterations=0)
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
        ({"iterations": -1}, "nonnegative iteration count"),
        ({"iterations": True}, "nonnegative iteration count"),
        ({"precision": 0}, "positive seeds nonzero"),
        ({"result": ()}, "keep all integer bits"),
        ({"result": (9,) * 5}, "keep all integer bits"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": ()}, "work wires"),
        ({"classical_bits": (-1,) + tuple(range(7))}, "distinct nonnegative slots"),
        ({"result": (0, 3)}, "distinct nonnegative wires"),
    ],
)
def test_inverse_sqrt_rejects_invalid_layout(change, match):
    names = ("source", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(3, 1, 2, 1, 2), strict=True))
    settings = {"source_fraction_bits": 1, "precision": 2, "iterations": 1}
    with pytest.raises(ValueError, match=match):
        inverse_sqrt_ir(**(bound | settings | change))
