"""Signed Horner step with per-partial truncation and measured adjoints."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import floor, sqrt

import pytest
import torch

from flagquantum.compiler.registers.fixed_point_horner import (
    fixed_point_horner_step_ir,
    fixed_point_horner_step_requirements,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n, width):
    work, slots = fixed_point_horner_step_requirements(width)
    return (
        tuple(range(n)),
        tuple(range(n, n + width)),
        tuple(range(n + width, n + 2 * width)),
        tuple(range(n + 2 * width, n + 2 * width + work)),
        tuple(range(slots)),
    )


def oracle(a, y, f, width, coefficient):
    total = coefficient
    exact = Fraction(coefficient)
    for bit in range(width):
        if y >> bit & 1:
            term = Fraction(a << bit, 1 << f)
            sign = -1 if bit == width - 1 else 1
            total += sign * floor(term)
            exact += sign * term
    assert abs(total - exact) < width
    return total


@pytest.mark.parametrize(
    "n,width,f,c",
    [
        (1, 1, 0, -1),
        (1, 1, 1, 0),
        (2, 3, 2, -4),
        (4, 3, 1, 3),
        (2, 5, 0, -3),
        (4, 4, 3, 0),
    ],
)
def test_horner_all_input_pairs_history_and_costs(n, width, f, c):
    bound = layout(n, width)
    work = len(bound[3])
    settings = {"argument_fraction_bits": f, "coefficient": c}
    forward = fixed_point_horner_step_ir(*bound, **settings)
    inverse = fixed_point_horner_step_ir(*bound, **settings, adjoint=True)
    events = width - 1 if c else 0
    for bit in range(width):
        shift = bit - f
        destination = width - max(0, shift)
        source = min(destination, n + min(0, shift))
        if source > 0:
            events += source + destination - 1
    for stream in (forward, inverse):
        counts = Counter(gate.name for gate in stream)
        assert counts["measure"] == counts["reset"] == events
        assert counts["t"] + counts["tdg"] == 4 * events
        assert counts["ccx"] == 0
    assert work == 2 * width - 1

    def word(value, size):
        return tuple(value >> i & 1 for i in reversed(range(size)))

    before, after = [], []
    mask = (1 << width) - 1
    for a, y in product(range(1 << n), range(1 << width)):
        for occupied in (0, mask, (a + 2 * y) & mask):
            prefix, clean = word(a, n) + word(y, width), (0,) * work
            before.append(prefix + word(occupied, width) + clean)
            update = (occupied + oracle(a, y, f, width, c)) & mask
            after.append(prefix + word(update, width) + clean)
    assert certified_and_channel_map(forward, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(inverse, tuple(after)) == (tuple(before), events)


@pytest.mark.parametrize("n,width,f,c", [(1, 1, 0, -1), (2, 3, 2, 1), (3, 3, 1, -2)])
def test_horner_complex_states_and_exact_adjoint(n, width, f, c):
    bound = layout(n, width)
    work = len(bound[3])
    settings = {"argument_fraction_bits": f, "coefficient": c}
    forward = fixed_point_horner_step_ir(*bound, **settings)
    inverse = fixed_point_horner_step_ir(*bound, **settings, adjoint=True)
    wire_count = n + 2 * width + work + 1
    norm = sqrt(2 ** (n + width + 1))
    before, after = {}, {}
    for a, y, s in product(range(1 << n), range(1 << width), (0, 1)):
        occupied = (a + y) % (1 << width)
        updated = (occupied + oracle(a, y, f, width, c)) % (1 << width)
        prefix = ((a << width) | y) << (width + work + 1)
        i = prefix | (occupied << (work + 1)) | s
        j = prefix | (updated << (work + 1)) | s
        before[i] = after[j] = exp(0.3j * (a - y) * (1 + s)) / norm
    for stream, initial, expected in (
        (forward, before, after),
        (inverse, after, before),
    ):
        for seed in (9, 51, 89):
            actual = execute_sparse_register(
                CircuitIR(wire_count, stream), initial, seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in actual.keys() | expected.keys()
            )
        if width == 1:
            dense = torch.zeros(1 << wire_count, dtype=torch.complex128)
            target = torch.zeros_like(dense)
            for i, amplitude in initial.items():
                dense[i] = amplitude
            for i, amplitude in expected.items():
                target[i] = amplitude
            for event in (0, 1):
                torch.testing.assert_close(
                    forced_branch(dense, stream, (event,)),
                    target / sqrt(2),
                    atol=2e-12,
                    rtol=0,
                )


def test_horner_truncates_before_each_signed_contribution():
    assert oracle(3, 3, 2, 3, 0) == 1
    assert floor(Fraction(9, 4)) == 2
    assert oracle(3, 7, 2, 3, 0) == -2
    assert floor(Fraction(-3, 4)) == -1


def test_horner_append_round_trip():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(1, 1)
    settings = {"argument_fraction_bits": 0, "coefficient": 0}
    state = torch.zeros(16, dtype=torch.complex128)
    state[0], state[12] = 0.8, 0.6j
    expected = state.clone()
    expected[12], expected[14] = 0, 0.6j
    circuit = DynamicCircuit(4, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(4, fixed_point_horner_step_ir(*bound, **settings)))
    forward = run_dynamic(circuit, shots=4, seed=11, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, expected.expand(4, -1), atol=2e-12, rtol=0
    )
    circuit.append(
        FieldOperation(4, fixed_point_horner_step_ir(*bound, **settings, adjoint=True))
    )
    reverse = run_dynamic(circuit, shots=4, seed=11, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, state.expand(4, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"result": ()}, "positive result width"),
        ({"argument": ()}, "valid argument format"),
        ({"argument_fraction_bits": 4}, "valid argument format"),
        ({"previous": ()}, "matching previous and result widths"),
        ({"coefficient": 4}, "signed result width"),
        ({"coefficient": True}, "signed result width"),
        ({"adjoint": 1}, "adjoint must be boolean"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": (0,) * 5}, "distinct nonnegative slots"),
        ({"result": (0, 6, 7)}, "distinct nonnegative wires"),
    ],
)
def test_horner_invalid_layout(change, match):
    names = ("argument", "previous", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(2, 3), strict=True))
    with pytest.raises(ValueError, match=match):
        fixed_point_horner_step_ir(
            **(bound | {"argument_fraction_bits": 2, "coefficient": 0} | change)
        )


@pytest.mark.parametrize(
    "argument_width,width,fraction", [(1, 1, 1), (2, 3, 2), (3, 2, 1)]
)
def test_register_horner_exhaustive_words_and_branches(argument_width, width, fraction):
    from flagquantum.compiler.registers.fixed_point_horner import (
        register_horner_step_ir,
        register_horner_step_requirements,
    )

    work, slots = register_horner_step_requirements(argument_width, width)
    argument = tuple(range(argument_width))
    prev, coeff, result = (
        tuple(
            range(argument_width + index * width, argument_width + (index + 1) * width)
        )
        for index in range(3)
    )
    scratch = tuple(
        range(argument_width + 3 * width, argument_width + 3 * width + work)
    )
    bound = (argument, prev, coeff, result, scratch, tuple(range(slots)))
    forward = register_horner_step_ir(*bound, argument_fraction_bits=fraction)
    inverse = register_horner_step_ir(
        *bound, argument_fraction_bits=fraction, adjoint=True
    )
    events = sum(gate.name == "measure" for gate in forward)
    before, after = [], []
    mask = (1 << width) - 1
    for a, y, c in product(
        range(1 << argument_width), range(1 << width), range(1 << width)
    ):
        occupied = (a + y) & mask
        updated = (occupied + c - ((a * y) >> fraction)) & mask
        bits = f"{a:0{argument_width}b}{y:0{width}b}{c:0{width}b}"
        before.append(tuple(map(int, bits + f"{occupied:0{width}b}")) + (0,) * work)
        after.append(tuple(map(int, bits + f"{updated:0{width}b}")) + (0,) * work)
    assert certified_and_channel_map(forward, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(inverse, tuple(after)) == (tuple(before), events)
    norm = sqrt(len(before))
    amplitudes = [exp(0.3j * i) / norm for i in range(len(before))]

    def state(rows):
        return {
            int("".join(map(str, row)), 2): a
            for row, a in zip(rows, amplitudes, strict=True)
        }

    initial, expected = state(before), state(after)
    for seed in (3, 17):
        actual = execute_sparse_register(
            CircuitIR(len(before[0]), forward), initial, seed=seed
        )
        assert all(
            abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
            for i in actual.keys() | expected.keys()
        )
