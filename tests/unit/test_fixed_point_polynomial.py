"""Literal Horner schedule, bit-exact values and reversible cleanup."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import floor, sqrt

import pytest
import torch

from flagquantum.compiler.registers.bits.bitwise import xor_constant_ir, xor_ir
from flagquantum.compiler.registers.fixed_point.fixed_point_horner import (
    fixed_point_horner_step_ir,
)
from flagquantum.compiler.registers.fixed_point.fixed_point_polynomial import (
    fixed_point_polynomial_ir,
    fixed_point_polynomial_requirements,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n, w, d):
    work, slots = fixed_point_polynomial_requirements(w, d)
    return (
        tuple(range(n)),
        tuple(range(n, n + w)),
        tuple(range(n + w, n + w + work)),
        tuple(range(slots)),
    )


def oracle(word, f, width, coeffs):
    argument = Fraction(word, 1 << f)
    modulus = 1 << width
    current = coeffs[0] % modulus
    for coefficient in coeffs[1:]:
        terms = tuple(floor(argument * (1 << j)) for j in range(width))
        weighted = sum(value for j, value in enumerate(terms[:-1]) if current >> j & 1)
        if current & (modulus >> 1):
            weighted -= terms[-1]
        current = (coefficient + weighted) % modulus
    return current


@pytest.mark.parametrize(
    "n,w,f,coeffs",
    [
        (1, 1, 0, (-1,)),
        (1, 1, 0, (-1, -1)),
        (2, 3, 2, (1, 1, -4)),
        (4, 3, 1, (3, -4, 0, 3)),
        (5, 4, 4, (0, 0, -3, 2, -7)),
    ],
)
def test_polynomial_all_inputs_and_literal_reverse_schedule(n, w, f, coeffs):
    bound = layout(n, w, len(coeffs) - 1)
    source, result, workspace, bits = bound
    gates = fixed_point_polynomial_ir(
        *bound, argument_fraction_bits=f, coefficients=coeffs
    )
    if len(coeffs) == 1:
        assert gates == xor_constant_ir(result, coeffs[0] % (1 << w))
        assert not workspace and not bits
    else:
        history = tuple(workspace[j * w : (j + 1) * w] for j in range(len(coeffs)))
        scratch = workspace[len(coeffs) * w :]
        assert len(scratch) == len(bits) == 2 * w - 1
        load = xor_constant_ir(history[0], coeffs[0] % (1 << w))
        forward, reverse = [], []
        for a, b, c in zip(history[:-1], history[1:], coeffs[1:], strict=True):
            args = (source, a, b, scratch, bits)
            forward.append(
                fixed_point_horner_step_ir(
                    *args, argument_fraction_bits=f, coefficient=c
                )
            )
            reverse.append(
                fixed_point_horner_step_ir(
                    *args, argument_fraction_bits=f, coefficient=c, adjoint=True
                )
            )
        expected = (*forward, xor_ir(history[-1], result), *reversed(reverse))
        assert gates == (*load, *(gate for stage in expected for gate in stage), *load)

    count = Counter(gate.name for gate in gates)
    assert count["measure"] == count["reset"]
    assert count["t"] + count["tdg"] == 4 * count["measure"]
    assert count["ccx"] == 0

    def bits_of(value, width):
        return tuple(value >> j & 1 for j in reversed(range(width)))

    before, after = [], []
    for x in range(1 << n):
        for y in range(1 << w):
            prefix, clean = bits_of(x, n), (0,) * len(workspace)
            before.append(prefix + bits_of(y, w) + clean)
            after.append(prefix + bits_of(y ^ oracle(x, f, w, coeffs), w) + clean)
    assert certified_and_channel_map(gates, tuple(before)) == (
        tuple(after),
        count["measure"],
    )
    assert certified_and_channel_map(gates, tuple(after)) == (
        tuple(before),
        count["measure"],
    )


@pytest.mark.parametrize(
    "n,w,f,coeffs", [(1, 1, 0, (-1, -1)), (2, 3, 2, (1, 1, -4)), (3, 2, 1, (-2, 1, 0))]
)
def test_polynomial_preserves_complex_amplitudes_and_cleans_history(n, w, f, coeffs):
    bound = layout(n, w, len(coeffs) - 1)
    gates = fixed_point_polynomial_ir(
        *bound, argument_fraction_bits=f, coefficients=coeffs
    )
    work = len(bound[2])
    wire_count = n + w + work + 1
    before, after = {}, {}
    for x, s in product(range(1 << n), (0, 1)):
        y = x % (1 << w)
        update = oracle(x, f, w, coeffs)
        i = (x << (w + work + 1)) | (y << (work + 1)) | s
        j = i ^ (update << (work + 1))
        before[i] = after[j] = exp(0.3j * x * (1 + s)) / sqrt(2 ** (n + 1))
    for initial, expected in ((before, after), (after, before)):
        for seed in (9, 43, 81):
            actual = execute_sparse_register(
                CircuitIR(wire_count, gates), initial, seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in actual.keys() | expected.keys()
            )
        if w == 1:
            dense = torch.zeros(1 << wire_count, dtype=torch.complex128)
            target = torch.zeros_like(dense)
            for i, amplitude in initial.items():
                dense[i] = amplitude
            for i, amplitude in expected.items():
                target[i] = amplitude
            for history in product((0, 1), repeat=2):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history), target / 2, atol=2e-12, rtol=0
                )


def test_cross_level_truncation_witness():
    x = Fraction(3, 4)
    assert oracle(3, 2, 3, (1, 1, -4)) == 4
    assert floor(x**2 + x - 4) % 8 == 5


def test_polynomial_append_and_uncompute():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(1, 1, 1)
    width = 2 + len(bound[2])
    gates = fixed_point_polynomial_ir(
        *bound, argument_fraction_bits=0, coefficients=(-1, -1)
    )
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[0], state[1 << (width - 1)] = 0.8, 0.6j
    expected = state.clone()
    expected[0], expected[1 << (width - 2)] = 0, 0.8
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
        ({"coefficients": ()}, "nonnegative degree"),
        ({"argument": ()}, "valid argument format"),
        ({"coefficients": (1, True)}, "signed result width"),
        ({"coefficients": (1, 4)}, "signed result width"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": (0,) * 5}, "distinct nonnegative slots"),
        ({"result": (0, 6, 7)}, "distinct nonnegative wires"),
    ],
)
def test_polynomial_invalid_layout(change, match):
    names = ("argument", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(2, 3, 1), strict=True))
    with pytest.raises(ValueError, match=match):
        fixed_point_polynomial_ir(
            **(bound | {"argument_fraction_bits": 2, "coefficients": (1, 0)} | change)
        )
