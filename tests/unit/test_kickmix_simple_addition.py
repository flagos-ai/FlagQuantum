"""Check mixed ripple addition and complete measured carry erasure."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field import ClassicalFieldCoefficient
from flagquantum.compiler.registers.kickmix_simple_addition import (
    kickmix_simple_addition_ir,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("controlled", [False, True])
def test_simple_addition_restores_each_complex_branch(
    width: int, controlled: bool
) -> None:
    low, carry, control = range(width, width + 3)
    data = width + 3
    total = data + width
    offset = (ClassicalFieldCoefficient(0), low)
    clean = tuple(range(data, total))
    setup = ClassicalXor((), 0, constant=True).instruction()
    gates = kickmix_simple_addition_ir(
        tuple(range(width)),
        offset,
        clean=clean,
        carry_in=carry,
        control=control if controlled else True,
        measurement_bit=1,
    )
    counts = Counter(
        op.opcode
        for op in lower_reversible_phase(CircuitIR(total, (setup, *gates))).operations
    )
    assert counts["hmr"] == (width if width >= 3 else int(width == 2 and controlled))
    basis = torch.arange(1 << data)
    amps = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amps /= amps.norm()
    stride = 1 << width
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[::stride] = amps
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(total, (setup, *gates)))
    shots = 384 if width == 5 else 128
    result = run_dynamic(circuit, shots=shots, seed=41, strategy="trajectory")
    assert result.statistics["branch_count"] == 1 << counts["hmr"]
    assert torch.all(result.classical_bits[:, 0] == 1)
    enabled = (basis & 1) if controlled else 1
    value = (
        (basis >> 3) + (2 + ((basis >> 2) & 1) + ((basis >> 1) & 1)) * enabled
    ) % stride
    output = (value << 3) | (basis & 7)
    expected = torch.zeros_like(state)
    expected[output * stride] = amps
    torch.testing.assert_close(
        result.final_states.reshape(shots, -1),
        expected.expand(shots, -1),
        atol=1e-12,
        rtol=0,
    )


def test_simple_addition_uses_the_locked_carry_order_and_gate_budget() -> None:
    setup = tuple(ClassicalXor((), bit).instruction() for bit in range(3))
    gates = kickmix_simple_addition_ir(
        (4, 3, 2, 1, 0),
        tuple(ClassicalFieldCoefficient(bit) for bit in (2, 1, 0)),
        clean=(10, 11, 12, 13, 14),
        carry_in=9,
        control=8,
        measurement_bit=4,
    )
    program = lower_reversible_phase(CircuitIR(15, (*setup, *gates)))
    counts = Counter(op.opcode for op in program.operations)
    assert (counts["ccx"], counts["cx"], counts["hmr"]) == (5, 17, 5)
    text = encode_kmx(program).splitlines()
    assert [line for line in text if line.startswith("HMR")] == [
        f"HMR q{wire} b4" for wire in (14, 13, 12, 11, 10)
    ]


def test_simple_addition_rejects_the_source_out_of_bounds_offsets() -> None:
    with pytest.raises(ValueError, match="explicit low offset bits"):
        kickmix_simple_addition_ir((0,), (), clean=(1,))
    with pytest.raises(ValueError, match="explicit low offset bits"):
        kickmix_simple_addition_ir((0, 1), (True,), clean=(2, 3))
    with pytest.raises(ValueError, match="clean wire per target"):
        kickmix_simple_addition_ir((0,), (True,), clean=())
