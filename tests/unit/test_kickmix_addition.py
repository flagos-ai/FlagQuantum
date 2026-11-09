"""Check the locked in-place arithmetic, including each measured carry history."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.arithmetic.inplace_addition import (
    inplace_addition_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


def test_two_bit_controlled_add_matches_locked_measured_schedule() -> None:
    gates = inplace_addition_ir(
        (1, 0), (3, 2), clean=(5,), control=4, measurement_bit=0
    )
    assert encode_kmx(lower_reversible_phase(CircuitIR(6, gates))).splitlines()[1:] == [
        "CCX q4 q2 q5",
        "CCX q5 q0 q1",
        "HMR q5 b0",
        "CZ q4 q2 if b0",
        "CCX q4 q2 q0",
        "CCX q4 q3 q1",
    ]


@pytest.mark.parametrize("subtract", [False, True])
@pytest.mark.parametrize(
    "n,m,clean_count,dirty_count,controlled,minimize,measurements",
    [
        (2, 1, 1, 0, True, False, 1),
        (2, 1, 0, 1, True, True, 0),
        (4, 3, 2, 0, True, False, 2),
        (4, 3, 2, 0, False, True, 1),
        (4, 3, 0, 1, True, False, 0),
        (4, 5, 0, 1, True, False, 0),
        (5, 2, 3, 0, True, False, 1),
    ],
)
def test_addition_restores_inputs_and_work_on_every_complex_branch(
    n: int,
    m: int,
    clean_count: int,
    dirty_count: int,
    controlled: bool,
    minimize: bool,
    measurements: int,
    subtract: bool,
) -> None:
    # Logical words, an enable/spectator, arbitrary dirty data, then clean wires.
    inputs = n + m + 1 + dirty_count
    total = inputs + clean_count
    basis = torch.arange(1 << inputs)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[:: 1 << clean_count] = amplitudes
    gates = inplace_addition_ir(
        tuple(range(n)),
        tuple(range(n, n + m)),
        clean=tuple(range(inputs, total)),
        dirty=tuple(range(n + m + 1, inputs)),
        control=n + m if controlled else True,
        minimize_qubits=minimize,
        subtract=subtract,
        measurement_bit=0 if measurements else None,
    )
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128).append(
        FieldOperation(total, gates)
    )
    result = run_dynamic(circuit, shots=48, seed=41, strategy="trajectory")
    counts = Counter(
        op.opcode for op in lower_reversible_phase(CircuitIR(total, gates)).operations
    )
    if (n, m, clean_count, dirty_count) == (4, 3, 2, 0):
        assert counts["ccx"] == (10 if controlled else 4)
    assert counts["hmr"] == measurements
    assert result.statistics["branch_count"] == 1 << measurements
    lower_width = inputs - n
    lhs = basis >> lower_width
    rhs = (basis >> (1 + dirty_count)) & ((1 << m) - 1)
    enabled = ((basis >> dirty_count) & 1) if controlled else 1
    update = lhs + (-rhs if subtract else rhs) * enabled
    expected = torch.zeros_like(state)
    output = ((update % (1 << n)) << lower_width) | (basis % (1 << lower_width))
    expected[output << clean_count] = amplitudes
    torch.testing.assert_close(
        result.final_states.reshape(48, -1), expected.expand(48, -1), atol=1e-12, rtol=0
    )


def test_addition_rejects_missing_padding_carry_or_result_slot() -> None:
    with pytest.raises(ValueError, match="zero-extend"):
        inplace_addition_ir((0, 1, 2), (3,))
    with pytest.raises(ValueError, match="carry wire"):
        inplace_addition_ir((0, 1, 2), (3,), clean=(4,))
    with pytest.raises(ValueError, match="measurement slot"):
        inplace_addition_ir((0, 1, 2), (3, 4), clean=(5,))
    assert not inplace_addition_ir((), (0,), subtract=True)
    assert not inplace_addition_ir((0,), (1,), control=False)
