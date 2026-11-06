"""Exact polynomial argument, clean branches and pinned ArcSin boundaries."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.arcsine_reduction import (
    arcsine_reduction_ir,
    arcsine_reduction_requirements,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(f):
    work, slots = arcsine_reduction_requirements(f)
    return (
        tuple(range(f + 1)),
        tuple(range(f + 1, 3 * f + 1)),
        tuple(range(3 * f + 1, 3 * f + 1 + work)),
        tuple(range(slots)),
    )


def expected_word(value, f):
    x = Fraction(value, 1 << f)
    reduced = x * x if x < Fraction(1, 2) else (1 - x) / 2
    word = reduced * (1 << (2 * f))
    assert word.denominator == 1
    return int(word)


@pytest.mark.parametrize("f", (1, 2, 3, 5, 8))
def test_arcsine_reduction_all_inputs_and_histories(f):
    bound = layout(f)
    work = len(bound[2])
    gates = arcsine_reduction_ir(*bound)
    counts = Counter(g.name for g in gates)
    erasures = 2 * f + 2 * f * (f - 1)
    assert counts["measure"] == counts["reset"] == erasures
    assert counts["t"] + counts["tdg"] == 4 * erasures
    assert counts["ccx"] == 3 * f + 3

    def word(value, width):
        return tuple(value >> i & 1 for i in reversed(range(width)))

    before, after = [], []
    for value in range((1 << f) + 1):
        prefix = word(value, f + 1)
        occupied = (1 << (2 * f)) - 1 if value % 2 else 0
        before.append(prefix + word(occupied, 2 * f) + (0,) * work)
        after.append(
            prefix + word(occupied ^ expected_word(value, f), 2 * f) + (0,) * work
        )
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), erasures)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), erasures)


@pytest.mark.parametrize("f", (1, 2, 3))
def test_arcsine_reduction_full_complex_trajectories(f):
    bound = layout(f)
    work = len(bound[2])
    width = 3 * f + 2 + work
    gates = arcsine_reduction_ir(*bound)
    norm = sqrt(2 * ((1 << f) + 1))
    before, after = {}, {}
    for value in range((1 << f) + 1):
        for spectator in (0, 1):
            src = (value << (2 * f + work + 1)) | spectator
            dest = src | (expected_word(value, f) << (work + 1))
            before[src] = after[dest] = exp(1j * (value + spectator / 5)) / norm
    for initial, expected in ((before, after), (after, before)):
        for seed in (7, 21, 49):
            observed = execute_sparse_register(
                CircuitIR(width, gates), initial, seed=seed
            )
            assert all(
                abs(observed.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in observed.keys() | expected.keys()
            )

        if f == 1:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            target = torch.zeros_like(dense)
            for i, amplitude in initial.items():
                dense[i] = amplitude
            for i, amplitude in expected.items():
                target[i] = amplitude
            for history in product((0, 1), repeat=2):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history), target / 2, atol=2e-12, rtol=0
                )


def test_arcsine_reduction_append_round_trip():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(1)
    gates = arcsine_reduction_ir(*bound)
    work = len(bound[2])
    width = 4 + work
    state = torch.zeros(1 << width, dtype=torch.complex128)
    source = 1 << (2 + work)
    state[0], state[source] = 0.8, 0.6j
    expected = state.clone()
    expected[source], expected[source | (1 << work)] = 0, 0.6j
    circuit = DynamicCircuit(width, inputs=state, dtype=torch.complex128)
    operation = FieldOperation(width, gates)
    circuit.append(operation)
    actual = run_dynamic(circuit, shots=4, seed=7, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states, expected.expand(4, -1), atol=2e-12, rtol=0
    )
    circuit.append(operation)
    actual = run_dynamic(circuit, shots=4, seed=7, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states, state.expand(4, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize("f,model", [(2, 2826), (3, 2520), (4, 1908)])
def test_pinned_arcsine_encoding_and_resource_boundary(f, model):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.trigonometric.arcsin import ArcSin
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    reference = ArcSin(4, f)
    assert reference.signature.get_left("result").dtype.num_frac == f
    assert reference.on_classical_vals(x=15, result=0) == {"x": 15, "result": 19}
    assert get_cost_value(reference, QECGatesCost()).toffoli == model
    with pytest.raises(DecomposeNotImplementedError):
        reference.decompose_bloq()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"source": (0,)}, "positive input width"),
        ({"result": (5,)}, "twice"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": (0, 0)}, "distinct nonnegative slots"),
        ({"classical_bits": (-1, 0)}, "distinct nonnegative slots"),
        ({"result": (0, 4, 5, 6)}, "distinct nonnegative wires"),
    ],
)
def test_arcsine_reduction_rejects_invalid_bindings(change, match):
    names = ("source", "result", "workspace", "classical_bits")
    with pytest.raises(ValueError, match=match):
        arcsine_reduction_ir(**(dict(zip(names, layout(2), strict=True)) | change))
