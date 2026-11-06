"""Complete reciprocal schedule, precision and clean measured reversal."""

from cmath import exp
from collections import Counter
from fractions import Fraction
from itertools import product
from math import floor, sqrt

import pytest
import torch

from flagquantum.compiler.registers.reciprocal import (
    reciprocal_ir,
    reciprocal_requirements,
)
from flagquantum.compiler.registers.reciprocal_newton import reciprocal_newton_step_ir
from flagquantum.compiler.registers.reciprocal_seed import reciprocal_seed_ir
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n, f, b, output):
    work, bits = reciprocal_requirements(n, b, source_fraction_bits=f)
    return (
        tuple(range(n)),
        tuple(range(n, n + output)),
        tuple(range(n + output, n + output + work)),
        tuple(range(bits)),
    )


def oracle(word, fraction, precision, output):
    a = Fraction(word, 1 << fraction)
    if a < 1:
        return 0
    steps = (precision - 1).bit_length()
    scale = 1 << precision
    x = Fraction(1, 1 << floor(a).bit_length())
    for _ in range(steps):
        x = Fraction(floor((2 * x - a * x * x) * scale), scale)
    encoded = floor(x * (1 << output))
    error = 1 / a - Fraction(encoded, 1 << output)
    bound = Fraction(1, a * (1 << (1 << steps))) + Fraction(steps, scale)
    if output < precision:
        bound += Fraction(1, 1 << output)
    assert 0 <= error <= bound
    return encoded


@pytest.mark.parametrize(
    "n,f,b,o",
    [
        (1, 0, 1, 1),
        (3, 2, 1, 1),
        (2, 0, 2, 2),
        (3, 1, 3, 2),
        (4, 2, 5, 4),
        (5, 0, 5, 5),
    ],
)
def test_reciprocal_schedule_all_inputs_and_measurement_histories(n, f, b, o):
    bound = layout(n, f, b, o)
    source, result, work, bits = bound
    gates = reciprocal_ir(*bound, source_fraction_bits=f, precision=b)
    steps = (b - 1).bit_length()
    history = tuple(work[i * b : (i + 1) * b] for i in range(steps + 1))
    scratch = work[(steps + 1) * b :]
    m = n - f
    seed_args = (source, history[0], scratch[0], scratch[1:m], bits[: m - 1])
    seed = reciprocal_seed_ir(*seed_args, source_fraction_bits=f)
    erase = reciprocal_seed_ir(*seed_args, source_fraction_bits=f, adjoint=True)
    forward = tuple(
        reciprocal_newton_step_ir(
            source, history[i], history[i + 1], scratch, bits, source_fraction_bits=f
        )
        for i in range(steps)
    )
    copy = tuple(
        Instruction("cx", (a, z)) for a, z in zip(history[-1][:o], result, strict=True)
    )
    all_updates = (*forward, copy, *reversed(forward))
    assert gates == (*seed, *(g for stage in all_updates for g in stage), *erase)
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    assert counts["reset"] == events
    assert counts["ccx"] == 2 * (m - 1)

    def word(value, width):
        return tuple(value >> j & 1 for j in reversed(range(width)))

    before, after = [], []
    clean = (0,) * len(work)
    for value in range(1 << n):
        prefix = word(value, n)
        update = oracle(value, f, b, o)
        occupied = ((1 << o) - 1) if value % 2 else 0
        before.append(prefix + word(occupied, o) + clean)
        after.append(prefix + word(update ^ occupied, o) + clean)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


@pytest.mark.parametrize("n,f,b,o", [(1, 0, 1, 1), (2, 0, 2, 2), (3, 1, 3, 2)])
def test_reciprocal_complex_amplitudes_and_cleanup(n, f, b, o):
    bound = layout(n, f, b, o)
    gates = reciprocal_ir(*bound, source_fraction_bits=f, precision=b)
    work = len(bound[2])
    width = n + o + work + 1
    before, after = {}, {}
    for value, s in product(range(1 << n), (0, 1)):
        i = (value << (o + work + 1)) | s
        j = i | (oracle(value, f, b, o) << (work + 1))
        before[i] = after[j] = exp(1j * (value - 0.31 * s)) / sqrt(2 ** (n + 1))
    for initial, expected in ((before, after), (after, before)):
        for seed in (3, 19, 72):
            observed = execute_sparse_register(
                CircuitIR(width, gates), initial, seed=seed
            )
            assert all(
                abs(observed.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in observed.keys() | expected.keys()
            )
        if b == 1:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            desired = torch.zeros_like(dense)
            for i, value in initial.items():
                dense[i] = value
            for i, value in expected.items():
                desired[i] = value
            torch.testing.assert_close(
                forced_branch(dense, gates, ()), desired, atol=2e-12, rtol=0
            )


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"source_fraction_bits": 2}, "integer part"),
        ({"precision": 1}, "cover every"),
        ({"result": ()}, "nonempty result"),
        ({"workspace": ()}, "work wires"),
        ({"classical_bits": (0, 0, 0, 0, 0)}, "distinct nonnegative"),
        ({"result": (0, 1)}, "distinct nonnegative"),
    ],
)
def test_reciprocal_rejects_invalid_bindings(changes, error):
    names = ("source", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(2, 0, 2, 2), strict=True)) | {
        "source_fraction_bits": 0,
        "precision": 2,
    }
    with pytest.raises(ValueError, match=error):
        reciprocal_ir(**(bound | changes))


def test_reciprocal_append_round_trip():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(1, 0, 1, 1)
    gates = reciprocal_ir(*bound, source_fraction_bits=0, precision=1)
    state = torch.zeros(16, dtype=torch.complex128)
    state[0], state[8] = 0.8, 0.6j
    expected = state.clone()
    expected[8], expected[12] = 0, 0.6j
    circuit = DynamicCircuit(4, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(4, gates))
    forward = run_dynamic(circuit, shots=4, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, expected.expand(4, -1), atol=2e-12, rtol=0
    )
    circuit.append(FieldOperation(4, gates))
    reverse = run_dynamic(circuit, shots=4, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, state.expand(4, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize(
    "n,f,expected", [(1, 0, None), (2, 0, None), (3, 1, (21, 5)), (4, 2, (47, 7))]
)
def test_pinned_reciprocal_model_boundary(n, f, expected):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.multiplication import InvertRealNumber
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    reference = InvertRealNumber(n, f)
    assert reference.signature.get_left("result").dtype.num_frac == f
    if expected is None:
        with pytest.raises(ValueError, match="bitsize must be at least 3"):
            get_cost_value(reference, QECGatesCost())
    else:
        cost = get_cost_value(reference, QECGatesCost())
        assert (cost.toffoli, cost.measurement) == expected
