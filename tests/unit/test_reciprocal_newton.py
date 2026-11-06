"""Exact widened Newton arithmetic and measurement-assisted uncomputation."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import floor, sqrt

import pytest
import torch

from flagquantum.compiler.registers.reciprocal_newton import (
    reciprocal_newton_requirements,
    reciprocal_newton_step_ir,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.algorithms.primitives.multiplexers.test_thc_qrom import _execute_sparse
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch

pytestmark = pytest.mark.unit


def layout(n, b):
    work, slots = reciprocal_newton_requirements(n, b)
    return (
        tuple(range(n)),
        tuple(range(n, n + b)),
        tuple(range(n + b, n + 2 * b)),
        tuple(range(n + 2 * b, n + 2 * b + work)),
        tuple(range(slots)),
    )


def oracle(word, value, f, b):
    a = Fraction(word, 1 << f)
    x = Fraction(value, 1 << b)
    return floor((2 * x - a * x * x) * (1 << b))


def domain(n, f, b):
    return (
        (a, x)
        for a, x in product(range(1 << f, 1 << n), range(1 << b))
        if a * x <= 1 << (f + b)
    )


@pytest.mark.parametrize(
    "n,f,b", [(1, 0, 1), (1, 0, 5), (2, 0, 3), (3, 1, 3), (4, 2, 4)]
)
def test_newton_all_reachable_inputs_and_every_history(n, f, b):
    bound = layout(n, b)
    a, x, result, work, slots = bound
    gates = reciprocal_newton_step_ir(*bound, source_fraction_bits=f)
    square = b * (b - 1)
    product_work = 4 * n * b - max(n, 2 * b)
    carries = n + 2 * b - 1
    events = 2 * (square + product_work + carries)
    assert len(slots) == carries
    assert len(work) == 2 * b + n + 2 * b + max(square, product_work) + carries
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == 4 * events
    assert counts["measure"] == counts["reset"] == events
    assert not any(set(g.wires) & set(result) for g in gates if g.name != "cx")
    assert all(g.wires[-1] in result for g in gates if set(g.wires) & set(result))

    def word(value, bits):
        return tuple(value >> i & 1 for i in reversed(range(bits)))

    before, after = [], []
    for w, value in domain(n, f, b):
        update = oracle(w, value, f, b)
        assert value <= update < 1 << b
        dirty = (1 << b) - 1 if (w + value) & 1 else 0
        prefix = word(w, n) + word(value, b)
        before.append(prefix + word(dirty, b) + (0,) * len(work))
        after.append(prefix + word(update ^ dirty, b) + (0,) * len(work))
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


@pytest.mark.parametrize("n,f,b", [(1, 0, 1), (2, 0, 3), (3, 1, 2)])
def test_newton_primitive_complex_trajectories(n, f, b):
    bound = layout(n, b)
    gates = reciprocal_newton_step_ir(*bound, source_fraction_bits=f)
    work = len(bound[3])
    width = n + 2 * b + work + 1
    points = tuple(domain(n, f, b))
    before, after = {}, {}
    assert oracle(3, 2, 0, 3) == 2
    assert 2 * 2 - (3 * 2 * 2 >> 3) == 3
    for (w, value), spectator in product(points, (0, 1)):
        i = (((w << b) | value) << (b + work + 1)) | spectator
        j = i | (oracle(w, value, f, b) << (work + 1))
        before[i] = after[j] = exp(1j * (w - 0.31 * value + 0.2 * spectator)) / sqrt(
            2 * len(points)
        )

    for initial, expected in ((before, after), (after, before)):
        for seed in (3, 19, 72):
            actual = _execute_sparse(CircuitIR(width, gates), initial, seed=seed)
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in actual.keys() | expected.keys()
            )
        if b == 1:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            target = torch.zeros_like(dense)
            for i, value in initial.items():
                dense[i] = value
            for i, value in expected.items():
                target[i] = value
            events = sum(g.name == "measure" for g in gates)
            for history in (
                (0,) * events,
                (1,) * events,
                tuple(i % 2 for i in range(events)),
            ):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history),
                    target / sqrt(2**events),
                    atol=2e-12,
                    rtol=0,
                )


def test_locked_reciprocal_graph_records_mixed_format_gap():
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.multiplication import (
        InvertRealNumber,
        MultiplyTwoReals,
        ScaleIntByReal,
        SquareRealNumber,
    )
    from qualtran.resource_counting import SympySymbolAllocator

    source = InvertRealNumber(4, 2)
    assert source.signature.get_left("a").dtype.num_frac == 2
    graph = source.build_call_graph(SympySymbolAllocator())
    multiply = MultiplyTwoReals(4)
    assert (
        graph[multiply]
        == graph[SquareRealNumber(4)]
        == graph[ScaleIntByReal(4, 2)]
        == 2
    )
    assert multiply.signature.get_left("a").dtype.num_frac == 4
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"source_fraction_bits": 2}, "integer part"),
        ({"result": ()}, "matching estimate"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": (0, 0, 0)}, "distinct nonnegative"),
        ({"result": (0,)}, "distinct nonnegative"),
    ],
)
def test_reciprocal_newton_rejects_invalid_binding(overrides, error):
    names = ("source", "estimate", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(2, 1), strict=True)) | {"source_fraction_bits": 0}
    with pytest.raises(ValueError, match=error):
        reciprocal_newton_step_ir(**(bound | overrides))


def test_newton_append_and_erase():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(1, 1)
    work = len(bound[3])
    width = 3 + work
    gates = reciprocal_newton_step_ir(*bound, source_fraction_bits=0)
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[0], state[6 << work] = 0.8, 0.6j
    target = state.clone()
    target[6 << work], target[7 << work] = 0, 0.6j
    circuit = DynamicCircuit(width, inputs=state, dtype=torch.complex128)
    op = FieldOperation(width, gates)
    circuit.append(op)
    forward = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, target.expand(8, -1), atol=2e-12, rtol=0
    )
    circuit.append(op)
    reverse = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, state.expand(8, -1), atol=2e-12, rtol=0
    )
