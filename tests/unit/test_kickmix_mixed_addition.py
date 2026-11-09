"""Verify short, vented, recursive and approximate mixed addition schedules."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.arithmetic.mixed_addition import (
    kickmix_mixed_addition_ir,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("subtract", [False, True])
@pytest.mark.parametrize(
    "width,clean_count,dirty_count",
    [(1, 0, 0), (2, 2, 0), (3, 2, 0), (4, 2, 2), (5, 3, 2)],
)
def test_mixed_addition_cleans_workspace_on_every_complex_branch(
    width: int,
    clean_count: int,
    dirty_count: int,
    subtract: bool,
) -> None:
    offset_qubit, carry, control = range(width, width + 3)
    data = control + 1 + dirty_count
    total = data + clean_count
    rhs = (ClassicalFieldCoefficient(0), offset_qubit)
    setup = ClassicalXor((), 0, constant=True).instruction()
    gates = kickmix_mixed_addition_ir(
        tuple(range(width)),
        rhs,
        control=control,
        carry_in=carry,
        clean=tuple(range(data, total)),
        dirty=tuple(range(control + 1, data)),
        classical_bits=tuple(range(1, width + 3)),
        subtract=subtract,
    )
    basis = torch.arange(1 << data)
    amps = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amps /= amps.norm()
    stride = 1 << clean_count
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial[::stride] = amps
    result = run_dynamic(
        DynamicCircuit(total, inputs=initial, dtype=torch.complex128).append(
            FieldOperation(total, (setup, *gates))
        ),
        shots=192,
        seed=41,
        strategy="trajectory",
    )
    counts = Counter(
        op.opcode
        for op in lower_reversible_phase(CircuitIR(total, (setup, *gates))).operations
    )
    assert result.statistics["branch_count"] == 1 << counts["hmr"]
    shift = data - width
    rhs_value = 2 + ((basis >> (dirty_count + 2)) & 1)
    carry_value = (basis >> (dirty_count + 1)) & 1
    enable = (basis >> dirty_count) & 1
    change = (rhs_value + carry_value) * enable
    value = ((basis >> shift) + (-change if subtract else change)) % (1 << width)
    final = (value << shift) | (basis % (1 << shift))
    expected = torch.zeros_like(initial)
    expected[final * stride] = amps
    torch.testing.assert_close(
        result.final_states.reshape(192, -1),
        expected.expand(192, -1),
        atol=1e-12,
        rtol=0,
    )


def test_complemented_tolerance_restores_mutable_classical_inputs() -> None:
    setup = tuple(
        ClassicalXor((), slot, constant=True).instruction() for slot in (0, 1)
    )
    gates = kickmix_mixed_addition_ir(
        (0, 1, 2, 3, 4),
        (True, True, True, True, ClassicalFieldCoefficient(0)),
        carry_in=ClassicalFieldCoefficient(1),
        clean=(5, 6),
        dirty=(7, 8, 9),
        classical_bits=(2,),
        btol=0.5,
    )
    assert all(not set(gate.wires) & {0, 1, 2} for gate in gates)
    text = encode_kmx(
        lower_reversible_phase(CircuitIR(10, (*setup, *gates)))
    ).splitlines()
    assert text.count("BIT_INVERT b0") == text.count("BIT_INVERT b1") == 3
    circuit = (
        DynamicCircuit(10, dtype=torch.complex128)
        .x(0)
        .x(4)
        .append(FieldOperation(10, (*setup, *gates)))
    )
    result = run_dynamic(circuit, shots=12, seed=41, strategy="trajectory")
    assert torch.all(result.classical_bits[:, :2] == 1)
    assert torch.all(
        result.final_states.reshape(12, -1).abs().argmax(dim=1) == 512 + 32
    )


def test_unconditional_vented_cleanup_preserves_locked_failure() -> None:
    with pytest.raises(ValueError, match="quantum enable"):
        kickmix_mixed_addition_ir(
            (0, 1, 2, 3),
            (True,),
            clean=(4, 5),
            dirty=(6, 7),
            classical_bits=(0, 1, 2, 3),
        )


def test_locked_extra_clean_prefix_preserves_the_source_witness() -> None:
    setup = ClassicalXor((), 1, constant=True).instruction()
    gates = kickmix_mixed_addition_ir(
        (4, 3, 2, 1, 0),
        (ClassicalFieldCoefficient(1), 5),
        carry_in=8,
        control=7,
        clean=(9, 10, 11, 12, 13),
        classical_bits=(3,),
    )
    circuit = (
        DynamicCircuit(14, dtype=torch.complex128)
        .x(7)
        .append(FieldOperation(14, (setup, *gates)))
    )
    result = run_dynamic(circuit, shots=96, seed=41, strategy="trajectory")
    states = result.final_states.reshape(96, -1)
    # The unmodified native generator and simulator give 0 + 2 -> 6 and both
    # signs for this witness. All five workspace bits still return to zero.
    output = (1 << 12) | (1 << 11) | (1 << 6)
    assert result.statistics["branch_count"] == 32
    assert torch.all(states.abs().argmax(dim=1) == output)
    assert set(states[:, output].real.sign().tolist()) == {-1, 1}
