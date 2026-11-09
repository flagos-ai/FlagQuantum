"""Verify mixed carry XOR against the locked schedule and a classical recurrence."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.arithmetic.carry_parity import kickmix_carry_xor_ir
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives.measurement import measure_x_clear_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


def test_single_carry_prefers_the_offset_as_the_parity_carrier() -> None:
    # Native gen_ixor_carries_from_addition for one bit, with a clean wire.
    gates = kickmix_carry_xor_ir(
        (0,), (2,), (1,), carry_in=4, control=3, clean=(5,), measurement_bit=2
    )
    assert encode_kmx(lower_reversible_phase(CircuitIR(6, gates))).splitlines()[1:] == [
        "CX q4 q0",
        "CX q4 q2",
        "CCX q2 q3 q5",
        "CCX q5 q0 q1",
        "HMR q5 b2",
        "CZ q2 q3 if b2",
        "CX q4 q2",
        "CX q4 q0",
        "CCX q3 q4 q1",
    ]


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("quantum_carry", [False, True])
def test_carry_xor_preserves_every_complex_branch(n: int, quantum_carry: bool) -> None:
    # A high source bit is a spectator. Destination, enable, offset and carry
    # can all start in superposition. Three trailing wires start at zero.
    data = 2 * n + 4
    total = data + 3
    target = tuple(range(n + 1, 2 * n + 1))
    quantum_offset, quantum_flag, enable = range(2 * n + 1, data)
    clean, measured_offset, measured_carry = range(data, total)
    basis = torch.arange(1 << data)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::8] = amplitudes

    offset = (
        (quantum_offset,)
        if n == 1
        else (True, quantum_offset, ClassicalFieldCoefficient(0))
    )
    carry_in = quantum_flag if quantum_carry else ClassicalFieldCoefficient(1)
    prefix = (
        *measure_x_clear_ir(measured_offset, 0),
        *measure_x_clear_ir(measured_carry, 1),
    )
    gates = kickmix_carry_xor_ir(
        tuple(range(n + 1)),
        offset,
        target,
        carry_in=carry_in,
        control=enable,
        clean=(clean,),
        measurement_bit=2,
    )
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128).append(
        FieldOperation(total, (*prefix, *gates))
    )
    result = run_dynamic(circuit, shots=96, seed=41, strategy="trajectory")
    counts = Counter(
        op.opcode
        for op in lower_reversible_phase(CircuitIR(total, (*prefix, *gates))).operations
    )
    measured = int(n == 1 or quantum_carry)
    assert counts["hmr"] == 2 + measured
    assert result.statistics["branch_count"] == 1 << (2 + measured)
    histories = result.classical_bits[:, :2]
    for classical_offset in (0, 1):
        for classical_carry in (0, 1):
            offset_bit = (basis >> 2) & 1
            rhs = offset_bit if n == 1 else 4 | (offset_bit << 1) | classical_offset
            carry = ((basis >> 1) & 1) if quantum_carry else classical_carry
            lhs = (basis >> (n + 3)) & ((1 << n) - 1)
            word = torch.zeros_like(basis)
            for bit in range(n):
                carry = (((lhs >> bit) & 1) + ((rhs >> bit) & 1) + carry) >> 1
                word |= carry << bit
            expected = torch.zeros_like(state)
            outputs = basis ^ ((word * (basis & 1)) << 3)
            expected[outputs * 8] = amplitudes
            actual = result.final_states.reshape(96, -1)[
                (histories[:, 0] == classical_offset)
                & (histories[:, 1] == classical_carry)
            ]
            torch.testing.assert_close(
                actual, expected.expand_as(actual), atol=1e-12, rtol=0
            )


def test_carry_xor_preserves_locked_unconditional_dispatch_boundary() -> None:
    assert kickmix_carry_xor_ir((0,), (True,), (1,))[0].wires == (0, 1)
    assert not kickmix_carry_xor_ir((0,), (), ())
    with pytest.raises(ValueError, match="source must cover"):
        kickmix_carry_xor_ir((0,), (True,), (1, 2), control=3)
    with pytest.raises(ValueError, match="quantum enable"):
        kickmix_carry_xor_ir((0, 1), (True, True), (2, 3))
