"""Widened inverse-square-root arithmetic and measured channel checks."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import floor, sqrt

import pytest
import torch

from flagquantum.compiler.registers.inverse_sqrt_newton import (
    inverse_sqrt_newton_requirements,
    inverse_sqrt_newton_step_ir,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n, m, f, b, o):
    work, slots = inverse_sqrt_newton_requirements(
        n, m, source_fraction_bits=f, estimate_fraction_bits=b
    )
    return (
        tuple(range(n)),
        tuple(range(n, n + m)),
        tuple(range(n + m, n + m + o)),
        tuple(range(n + m + o, n + m + o + work)),
        tuple(range(slots)),
    )


def points(n, m, f, b, o):
    for w, x in product(range(1 << n), range(1 << m)):
        source = Fraction(w, 1 << f)
        estimate = Fraction(x, 1 << b)
        update = floor(x * (3 - source * estimate**2) / 2)
        if 0 <= update < 1 << o:
            yield w, x, update


@pytest.mark.parametrize(
    "n,m,f,b,o",
    [
        (1, 1, 0, 0, 1),
        (1, 2, 1, 2, 2),
        (2, 3, 0, 2, 3),
        (3, 3, 1, 2, 3),
        (5, 2, 0, 1, 4),
        (4, 4, 4, 2, 8),
    ],
)
def test_inverse_sqrt_all_legal_inputs_and_measurement_histories(n, m, f, b, o):
    bound = layout(n, m, f, b, o)
    work = len(bound[3])
    gates = inverse_sqrt_newton_step_ir(
        *bound, source_fraction_bits=f, estimate_fraction_bits=b
    )
    factor = max(n + 2 * m, f + 2 * b + 2)
    inner_product = 4 * n * m - max(n, 2 * m)
    final_product = (2 * m - 1) * factor
    events = 2 * (m * (m - 1) + inner_product + factor - 1) + final_product
    assert len(bound[4]) == factor - 1
    assert work == 2 * m + factor + max(inner_product, final_product) + factor - 1
    counts = Counter(gate.name for gate in gates)
    assert counts["t"] + counts["tdg"] == 4 * events
    assert counts["measure"] == counts["reset"] == events

    def bits(word, width):
        return tuple(word >> i & 1 for i in reversed(range(width)))

    before, after = [], []
    for w, x, update in points(n, m, f, b, o):
        prefix = bits(w, n) + bits(x, m)
        occupied = (1 << o) - 1 if (w + x) % 2 else 0
        before.append(prefix + bits(occupied, o) + (0,) * work)
        after.append(prefix + bits(occupied ^ update, o) + (0,) * work)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


@pytest.mark.parametrize(
    "n,m,f,b,o", [(1, 1, 0, 0, 1), (1, 2, 1, 2, 2), (3, 2, 1, 1, 2)]
)
def test_inverse_sqrt_complex_trajectories(n, m, f, b, o):
    bound = layout(n, m, f, b, o)
    gates = inverse_sqrt_newton_step_ir(
        *bound, source_fraction_bits=f, estimate_fraction_bits=b
    )
    work = len(bound[3])
    width = n + m + o + work + 1
    valid = tuple(points(n, m, f, b, o))
    before, after = {}, {}
    assert (1, 2, 2) in tuple(points(1, 2, 1, 2, 2))
    # In this witness the inner product falls below the two-bit fractional ulp.
    assert floor(Fraction(1, 16) * 4) == 0
    assert floor(2 * Fraction(3, 2)) == 3
    for (w, x, update), spectator in product(valid, (0, 1)):
        i = (((w << m) | x) << (o + work + 1)) | spectator
        j = i | (update << (work + 1))
        before[i] = after[j] = exp(1j * (w - 0.7 * x + 0.2 * spectator)) / sqrt(
            2 * len(valid)
        )
    for initial, expected in ((before, after), (after, before)):
        for seed in (7, 41, 92):
            actual = execute_sparse_register(
                CircuitIR(width, gates), initial, seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in actual.keys() | expected.keys()
            )
        if m == 1:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            target = torch.zeros_like(dense)
            for i, amplitude in initial.items():
                dense[i] = amplitude
            for i, amplitude in expected.items():
                target[i] = amplitude
            count = sum(gate.name == "measure" for gate in gates)
            for history in (
                (0,) * count,
                (1,) * count,
                tuple(i % 2 for i in range(count)),
            ):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history),
                    target / sqrt(2**count),
                    atol=2e-12,
                    rtol=0,
                )


def test_inverse_sqrt_append_and_restore():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(1, 1, 0, 0, 1)
    work = len(bound[3])
    width = 3 + work
    gates = inverse_sqrt_newton_step_ir(
        *bound, source_fraction_bits=0, estimate_fraction_bits=0
    )
    state = torch.zeros(1 << width, dtype=torch.complex128)
    index = 3 << (work + 1)
    state[0], state[index] = 0.8, 0.6j
    expected = state.clone()
    expected[index], expected[index | (1 << work)] = 0, 0.6j
    circuit = DynamicCircuit(width, inputs=state, dtype=torch.complex128)
    op = FieldOperation(width, gates)
    circuit.append(op)
    actual = run_dynamic(circuit, shots=4, seed=7, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states, expected.expand(4, -1), atol=2e-12, rtol=0
    )
    circuit.append(op)
    actual = run_dynamic(circuit, shots=4, seed=7, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states, state.expand(4, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"source_fraction_bits": 4}, "valid fractional widths"),
        ({"estimate_fraction_bits": True}, "valid fractional widths"),
        ({"result": (10,)}, "fit the fractional width"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": ()}, "work wires"),
        ({"classical_bits": (0,) * 8}, "distinct nonnegative slots"),
        ({"classical_bits": tuple(range(-1, 7))}, "distinct nonnegative slots"),
        ({"result": (0, 7, 8)}, "distinct nonnegative wires"),
    ],
)
def test_inverse_sqrt_invalid_layout(overrides, match):
    names = ("source", "estimate", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(3, 3, 1, 2, 3), strict=True))
    settings = {"source_fraction_bits": 1, "estimate_fraction_bits": 2}
    with pytest.raises(ValueError, match=match):
        inverse_sqrt_newton_step_ir(**(bound | settings | overrides))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "n,m,o,expected",
    [(2, 3, 3, (15, 2)), (5, 15, 22, (1140, 21)), (7, 8, 12, (345, 11))],
)
def test_locked_chemistry_update_is_a_distinct_resource_model(n, m, o, expected):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, QAny, QInt
    from qualtran.bloqs.arithmetic import (
        Add,
        MultiplyTwoReals,
        ScaleIntByReal,
        SquareRealNumber,
    )
    from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
        NewtonRaphsonApproxInverseSquareRoot,
    )
    from qualtran.resource_counting import (
        QECGatesCost,
        SympySymbolAllocator,
        get_cost_value,
    )

    model = NewtonRaphsonApproxInverseSquareRoot(n, m, o)
    assert model.signature.get_left("x_sq").dtype == QAny(n)
    assert model.signature.get_left("poly").dtype == QAny(m)
    assert model.signature.get_left("target").dtype == QAny(o)
    assert model.build_call_graph(SympySymbolAllocator()) == {
        SquareRealNumber(m): 1,
        ScaleIntByReal(m, n): 1,
        MultiplyTwoReals(o): 2,
        Add(QInt(o)): 1,
    }
    counts = get_cost_value(model, QECGatesCost())
    assert (counts.toffoli, counts.measurement) == expected
    with pytest.raises(DecomposeNotImplementedError):
        model.decompose_bloq()


def test_newton_quantum_correction_preserves_word_and_changes_update():
    source, estimate, result, work, bits = layout(1, 2, 0, 2, 3)
    correction = tuple(range(6 + len(work), 10 + len(work)))
    gates = inverse_sqrt_newton_step_ir(
        source,
        estimate,
        result,
        work,
        bits,
        source_fraction_bits=0,
        estimate_fraction_bits=2,
        correction=correction,
    )
    base = (1, 1, 0)
    cases = (
        ((0, 0, 0, 0), (0, 1, 0)),
        ((1, 0, 0, 0), (0, 1, 1)),
    )
    before = tuple(base + (0,) * (3 + len(work)) + word for word, _ in cases)
    after = tuple(base + updated + (0,) * len(work) + word for word, updated in cases)
    events = sum(gate.name == "measure" for gate in gates)
    assert certified_and_channel_map(gates, before) == (after, events)
    assert certified_and_channel_map(gates, after) == (before, events)
