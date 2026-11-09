"""Shared square columns, original model boundary and coherent erasure."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.integer_square import (
    integer_square_ir,
    sum_of_squares_ir,
    sum_of_squares_requirements,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n: int, k: int):
    result_width, work_width = sum_of_squares_requirements(n, k)
    start = n * k
    return (
        tuple(tuple(range(i * n, (i + 1) * n)) for i in range(k)),
        tuple(range(start, start + result_width)),
        tuple(range(start + result_width, start + result_width + work_width)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "n,k,ands,model",
    [
        (2, 1, 2, 2),
        (1, 2, 1, 1),
        (1, 3, 1, 1),
        (2, 2, 6, 6),
        (2, 3, 8, 9),
        (3, 3, 23, 23),
        (4, 3, 43, 43),
        (3, 4, 33, 33),
        (3, 6, 50, 50),
    ],
)
def test_sum_of_squares_layout_resources_and_every_branch(n, k, ands, model):
    pytest.importorskip("qualtran")
    import qualtran
    from qualtran import Side
    from qualtran.bloqs.arithmetic.multiplication import SumOfSquares
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    inputs, result, work = layout(n, k)
    gates = sum_of_squares_ir(inputs, result, work, measurement_bit=7)
    source = SumOfSquares(n, k)
    in_reg = source.signature.get_left("input")
    out_reg = source.signature.get_right("result")
    assert in_reg.shape == (k,) and in_reg.bitsize == n
    assert out_reg.bitsize == len(result) and out_reg.side == Side.RIGHT
    assert get_cost_value(source, QECGatesCost()).toffoli == model
    with pytest.raises(qualtran.DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(NotImplementedError):
        source.tensor_contract()
    assert source.on_classical_vals(input=(0,) * k) is NotImplemented
    counts = Counter(g.name for g in gates)
    assert len(work) == ands
    assert counts["t"] + counts["tdg"] == 4 * ands
    assert counts["measure"] == counts["reset"] == ands
    assert all(g.metadata["classical_bit"] == 7 for g in gates if g.name == "measure")
    if k == 1:
        assert gates == integer_square_ir(inputs[0], result, work, measurement_bit=7)

    def word(value, width):
        return tuple((value >> i) & 1 for i in reversed(range(width)))

    # Exhaust all input values for smaller layouts. For the wide case,
    # exercise all values of each input and the maximum simultaneous carry.
    values = (
        product(range(1 << n), repeat=k)
        if n * k <= 12
        else ((v,) * k for v in range(1 << n))
    )
    before, after = [], []
    for row in values:
        prefix = tuple(bit for value in row for bit in word(value, n))
        total = sum(value * value for value in row)
        # The circuit is also an XOR on a previously occupied result.
        seed = (1 << len(result)) - 1 if row[0] % 2 else 0
        before.append(prefix + word(seed, len(result)) + (0,) * len(work))
        after.append(prefix + word(seed ^ total, len(result)) + (0,) * len(work))
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), ands)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), ands)


@pytest.mark.parametrize("n,k", [(1, 3), (2, 2), (2, 3)])
def test_sum_of_squares_complex_amplitudes_and_clean_work(n, k):
    inputs, result, work = layout(n, k)
    gates = sum_of_squares_ir(inputs, result, work)
    width = n * k + len(result) + len(work) + 1
    before, after = {}, {}
    scale = sqrt(2 ** (n * k + 1))
    for row in product(range(1 << n), repeat=k):
        prefix = sum(value << (n * (k - 1 - i)) for i, value in enumerate(row))
        total = sum(value * value for value in row)
        for spectator in (0, 1):
            i = (prefix << (width - n * k)) | spectator
            j = i | (total << (len(work) + 1))
            before[i] = after[j] = exp(1j * (prefix - 0.4 * spectator)) / scale

    for incoming, outgoing in ((before, after), (after, before)):
        if n == 1:
            initial = torch.zeros(1 << width, dtype=torch.complex128)
            expected = torch.zeros_like(initial)
            for i, value in incoming.items():
                initial[i] = value
            for i, value in outgoing.items():
                expected[i] = value
            for history in product((0, 1), repeat=len(work)):
                torch.testing.assert_close(
                    forced_branch(initial, gates, history),
                    expected / sqrt(2 ** len(work)),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (4, 19, 73):
                actual = execute_sparse_register(
                    CircuitIR(width, gates), incoming, seed=seed
                )
                assert all(
                    abs(actual.get(i, 0) - outgoing.get(i, 0)) < 2e-12
                    for i in actual.keys() | outgoing.keys()
                )


def test_sum_of_squares_append_trajectory() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    inputs, result, work = layout(1, 3)
    gates = sum_of_squares_ir(inputs, result, work)
    state = torch.zeros(256, dtype=torch.complex128)
    state[0], state[7 << 5] = 0.8, 0.6j
    expected = state.clone()
    expected[7 << 5] = 0
    expected[(7 << 5) | (3 << 1)] = 0.6j
    circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(8, gates))
    actual = run_dynamic(circuit, shots=8, seed=13, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states, expected.expand(8, -1), atol=2e-12, rtol=0
    )


def test_sum_of_squares_rejects_invalid_binding() -> None:
    with pytest.raises(ValueError, match="positive register count"):
        sum_of_squares_requirements(2, 0)
    with pytest.raises(ValueError, match="equal-width"):
        sum_of_squares_ir(((0,), (1, 2)), (), ())
    with pytest.raises(ValueError, match="result and"):
        sum_of_squares_ir(((0,), (1,)), (), ())
