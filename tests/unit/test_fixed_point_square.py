"""Odd and even D6 squares with exact measured cleanup."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.and_compute import and_compute_ir
from flagquantum.compiler.registers.fixed_point_square import (
    fixed_point_square_ir,
    fixed_point_square_workspace_size,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(width):
    size = fixed_point_square_workspace_size(width)
    return (
        tuple(range(width)),
        tuple(range(width, 2 * width)),
        tuple(range(2 * width, 2 * width + size)),
    )


def truncated_square(value, width):
    """Evaluate the ordered D21 sum and the even D28 boundary adjustment."""
    bits = tuple((value >> k) & 1 for k in reversed(range(width)))
    total = sum(
        a * b * (1 << (width - i - j))
        for i, a in enumerate(bits, 1)
        for j, b in enumerate(bits, 1)
        if i + j <= width
    )
    if width % 2 == 0:
        total -= bits[width // 2 - 1]
        total += sum(a * b for a, b in zip(bits, reversed(bits), strict=True)) // 2
    return total


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "d,ands,model",
    [(3, 2, 0), (4, 7, 4), (5, 7, 8), (6, 15, 14), (8, 29, 28), (10, 46, 46)],
)
def test_fixed_square_source_boundary_columns_and_all_histories(d, ands, model):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, QFxp, Side
    from qualtran.bloqs.arithmetic.multiplication import SquareRealNumber
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = SquareRealNumber(d)
    assert tuple(reg.name for reg in source.signature.lefts()) == ("a", "b")
    assert all(reg.dtype == QFxp(d, d) for reg in source.signature)
    assert source.signature.get_right("result").side == Side.RIGHT
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(NotImplementedError):
        source.tensor_contract()
    assert source.on_classical_vals(a=0, b=0) is NotImplemented
    assert get_cost_value(source, QECGatesCost()).toffoli == model == d * d // 2 - 4

    data, result, work = layout(d)
    gates = fixed_point_square_ir(data, result, work, measurement_bit=7)
    pairs = tuple(
        (i, j)
        for i, j in product(range(d), repeat=2)
        if i > j and i + j + 2 <= d + (1 - d % 2)
    )
    prefix = tuple(
        g
        for (i, j), tmp in zip(pairs, work[: len(pairs)], strict=True)
        for g in and_compute_ir(data[i], data[j], tmp)
    )
    assert gates[: len(prefix)] == prefix
    counts = Counter(g.name for g in gates)
    assert len(work) == counts["measure"] == counts["reset"] == ands
    assert counts["t"] + counts["tdg"] == 4 * ands
    assert all(g.metadata["classical_bit"] == 7 for g in gates if g.name == "measure")

    def word(value):
        return tuple(value >> k & 1 for k in reversed(range(d)))

    initial, expected = [], []
    for value in range(1 << d):
        target = truncated_square(value, d)
        assert 0 <= target < 1 << d
        dirty = (1 << d) - 1 if value & 1 else 0
        initial.append(word(value) + word(dirty) + (0,) * len(work))
        expected.append(word(value) + word(dirty ^ target) + (0,) * len(work))
    assert certified_and_channel_map(gates, tuple(initial)) == (tuple(expected), ands)
    assert certified_and_channel_map(gates, tuple(expected)) == (tuple(initial), ands)


@pytest.mark.parametrize("d", [3, 4, 5, 8])
def test_fixed_square_complex_amplitudes_and_clean_work(d):
    data, result, work = layout(d)
    gates = fixed_point_square_ir(data, result, work)
    width = 2 * d + len(work) + 1
    before, after = {}, {}
    assert truncated_square(7, 3) == 4
    assert truncated_square(4, 4) == 0  # The even midpoint diagonal is omitted.
    assert truncated_square(9, 4) == 5  # The even i+j=d+1 term survives.
    for value, spectator in product(range(1 << d), (0, 1)):
        i = (value << (width - d)) | spectator
        j = i | (truncated_square(value, d) << (len(work) + 1))
        before[i] = after[j] = exp(1j * (value - 0.21 * spectator)) / sqrt(2 ** (d + 1))

    for initial, target in ((before, after), (after, before)):
        if d <= 4:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            desired = torch.zeros_like(dense)
            for i, value in initial.items():
                dense[i] = value
            for i, value in target.items():
                desired[i] = value
            for history in product((0, 1), repeat=len(work)):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history),
                    desired / sqrt(2 ** len(work)),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (3, 19, 72):
                actual = execute_sparse_register(
                    CircuitIR(width, gates), initial, seed=seed
                )
                assert all(
                    abs(actual.get(i, 0) - target.get(i, 0)) < 2e-12
                    for i in actual.keys() | target.keys()
                )


def test_fixed_square_append_round_trip():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    gates = fixed_point_square_ir(*layout(3))
    state = torch.zeros(256, dtype=torch.complex128)
    state[0], state[224] = 0.8, 0.6j
    target = state.clone()
    target[224], target[240] = 0, 0.6j
    circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128)
    op = FieldOperation(8, gates)
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


def test_fixed_square_rejects_invalid_binding():
    with pytest.raises(ValueError, match="at least three"):
        fixed_point_square_workspace_size(2)
    with pytest.raises(ValueError, match="result and"):
        fixed_point_square_ir((0, 1, 2), (), ())
    with pytest.raises(ValueError, match="distinct nonnegative"):
        fixed_point_square_ir((0, 1, 2), (2, 3, 4), (5, 6))
