"""Verify coherent coefficient arithmetic independently of fixed lookup tables."""

from cmath import exp
from itertools import product
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.grid_polynomial import GridInverseSqrtPolynomial
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width,fraction,out_width", [(1, 0, 1), (2, 1, 3), (2, 1, 1)])
def test_coherent_polynomial_preserves_inputs_and_cleans_scratch(
    width, fraction, out_width
):
    argument = tuple(range(width))
    coeff = tuple(tuple(range((i + 1) * width, (i + 2) * width)) for i in range(4))
    out = tuple(range(5 * width, 5 * width + out_width))
    logical = 5 * width + out_width
    work, slots = GridInverseSqrtPolynomial.requirements(width, width)
    scratch = tuple(range(logical, logical + work))
    op = GridInverseSqrtPolynomial(argument, coeff, out, scratch, fraction)
    circuit = DynamicCircuit(logical + work + 1).append(op)
    ir = circuit.to_ir()
    counts = count_emitted_instructions(ir)
    assert (work, slots, counts.t_count, counts.unconditional["measure"]) == (
        (6, 1, 48, 12) if width == 1 else (16, 1, 336, 84)
    )
    rows = tuple(product(range(1 << width), repeat=5))
    if width == 2:
        rows = rows[::17]
    incoming, expected = {}, {}
    cases = tuple(product(rows, range(1 << out_width), (0, 1)))
    for index, (row, target, spectator) in enumerate(cases):
        x, *words = row
        result = words[-1]
        for coefficient in reversed(words[:-1]):
            result = (coefficient - ((x * result) >> fraction)) % (1 << width)
        packed = 0
        for word in row:
            packed = (packed << width) | word
        before = (((packed << out_width) | target) << (work + 1)) | spectator
        result &= (1 << out_width) - 1
        after = (((packed << out_width) | (target ^ result)) << (work + 1)) | spectator
        amplitude = exp(0.37j * index) / sqrt(len(cases))
        incoming[before] = amplitude
        expected[after] = amplitude
    for seed in (3, 29):
        observed = execute_sparse_register(ir, incoming, seed=seed)
        assert observed == pytest.approx(expected, abs=4e-12)

    def bits(index):
        return tuple(index >> bit & 1 for bit in reversed(range(logical + work + 1)))

    assert certified_and_channel_map(ir.instructions, tuple(map(bits, incoming))) == (
        tuple(map(bits, expected)),
        counts.unconditional["measure"],
    )
    paired = DynamicCircuit(logical + work + 1).append(op).append(op).to_ir()
    assert execute_sparse_register(paired, incoming, seed=13) == pytest.approx(
        incoming, abs=4e-12
    )
