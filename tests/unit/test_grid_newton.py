"""Verify coherent Newton inputs, every certified branch and exact XOR recovery."""

from cmath import exp
from itertools import product
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.grid_newton import GridInverseSqrtNewton
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "n,m,f,b,o,delta", [(1, 1, 0, 0, 2, 0), (2, 2, 0, 1, 3, 0), (2, 2, 1, 1, 3, 2)]
)
def test_native_grid_newton_complete_valid_domain(n, m, f, b, o, delta):
    work, slots = GridInverseSqrtNewton.requirements(
        n, m, source_fraction_bits=f, estimate_fraction_bits=b
    )
    logical = n + m + o + delta
    width = logical + work + 1
    op = GridInverseSqrtNewton(
        tuple(range(n)),
        tuple(range(n, n + m)),
        tuple(range(n + m, n + m + o)),
        tuple(range(logical, logical + work)),
        f,
        b,
        tuple(range(n + m + o, logical)),
    )
    circuit = DynamicCircuit(width).append(op)
    ir = circuit.to_ir()
    counts = count_emitted_instructions(ir)
    if (n, m, f, b) == (2, 2, 0, 1):
        assert (work, slots, counts.t_count, counts.unconditional["measure"]) == (
            33,
            5,
            224,
            56,
        )
    scale = 1 << (f + 2 * b)
    rows = []
    for source, estimate, correction in product(
        range(1 << n), range(1 << m), range(1 << delta)
    ):
        if source * estimate**2 > 3 * scale:
            continue
        update = (
            estimate * (3 * scale + correction - source * estimate**2) // (2 * scale)
        )
        if update < 1 << o:
            rows.append((source, estimate, correction, update))
    cases = tuple(product(rows, range(1 << o), (0, 1)))
    incoming, expected = {}, {}
    for index, ((source, estimate, correction, update), target, spectator) in enumerate(
        cases
    ):
        prefix = (source << m) | estimate
        before = (((prefix << o) | target) << delta) | correction
        after = (((prefix << o) | (target ^ update)) << delta) | correction
        amplitude = exp(0.37j * index) / sqrt(len(cases))
        incoming[(before << (work + 1)) | spectator] = amplitude
        expected[(after << (work + 1)) | spectator] = amplitude

    def bits(index):
        return tuple(index >> bit & 1 for bit in reversed(range(width)))

    # Every actual And erasure must consume its exact conjunction. The
    # existing certificate verifies both outcomes for each primitive block.
    assert certified_and_channel_map(ir.instructions, tuple(map(bits, incoming))) == (
        tuple(map(bits, expected)),
        counts.unconditional["measure"],
    )
    assert certified_and_channel_map(ir.instructions, tuple(map(bits, expected))) == (
        tuple(map(bits, incoming)),
        counts.unconditional["measure"],
    )
    for seed in (3, 29):
        assert execute_sparse_register(ir, incoming, seed=seed) == pytest.approx(
            expected, abs=4e-12
        )
    paired = circuit.append(op).to_ir()
    assert execute_sparse_register(paired, incoming, seed=13) == pytest.approx(
        incoming, abs=4e-12
    )
