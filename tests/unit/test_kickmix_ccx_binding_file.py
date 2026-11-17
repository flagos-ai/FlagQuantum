"""Whole-file source verification for CCX and measured AND lifecycle bindings."""

from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives.broadcast_ccx import (
    broadcast_ccx_ir,
    broadcast_del_and_ir,
    broadcast_init_and_ir,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.fixtures.kickmix_binary_gate_matrix import (
    Port,
    ternary_matrix,
    verify_gate_stream,
)
from tests.unit.measurement_checks import forced_branch


def _emit(
    opcode: str, columns: tuple[object, ...], result: int
) -> tuple[Instruction, ...]:
    first, second, target = columns
    if opcode == "ccx":
        return broadcast_ccx_ir(first, second, target)
    if opcode == "init_and":
        return broadcast_init_and_ir(first, second, target)
    return broadcast_del_and_ir(first, second, target, measurement_bit=result)


@pytest.mark.parametrize(
    "opcode,columns,expected",
    [
        (opcode, columns, expected)
        for opcode in ("ccx", "init_and", "del_and")
        for columns, expected in ternary_matrix(opcode)
    ],
)
def test_ccx_and_source_matrix(
    opcode: str, columns: tuple[Port, ...], expected: tuple[int, str]
) -> None:
    verify_gate_stream(
        lambda ports, result: _emit(opcode, ports, result),
        columns,
        expected,
        include_resets=opcode == "init_and",
    )


def test_broadcast_and_recovers_each_measured_branch_after_dirty_reset() -> None:
    compute = broadcast_init_and_ir((0, Bit(0)), (Bit(1), True), (2, 3))
    erase = broadcast_del_and_ir((0, Bit(0)), (Bit(1), True), (2, 3), measurement_bit=2)
    mask = torch.arange(4)
    basis = mask.double()
    amplitudes = torch.complex(basis + 1, 3 * basis - 2)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[mask << 2 | 3] = amplitudes
    expected[mask << 2] = amplitudes
    for b0, b1 in product((False, True), repeat=2):
        prefix = tuple(
            ClassicalXor((), slot, constant=bit).instruction()
            for slot, bit in enumerate((b0, b1))
        )
        circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(4, (*prefix, *compute, *erase)))
        program = lower_reversible_phase(circuit.to_ir())
        lines = encode_kmx(program).splitlines()
        assert [line for line in lines if line.startswith(("R ", "HMR "))] == [
            "R q2",
            "R q3",
            "HMR q2 b2",
            "HMR q3 b2",
        ]
        assert program.n_bits == 3
        result = run_dynamic(circuit, shots=16, seed=17, strategy="batched")
        actual = result.final_states.reshape(16, -1)
        signs = (actual[:, 0] / expected[0]).real.sign()
        assert set(signs.tolist()) == {-1.0, 1.0}
        torch.testing.assert_close(
            actual, signs[:, None] * expected, rtol=0, atol=1e-12
        )

        # Observe the discarded reset outcomes without changing the reset map.
        instrumented = tuple(
            gate
            for step in (*prefix, *compute, *erase)
            for gate in (
                (
                    Instruction(
                        "measure",
                        step.wires,
                        metadata={
                            "is_dynamic": True,
                            "classical_bit": step.wires[0] + 1,
                        },
                    ),
                    step,
                )
                if step.name == "reset"
                else (step,)
            )
        )
        for history in product((0, 1), repeat=4):
            branch = forced_branch(initial, instrumented, history)
            phase = (-1) ** (history[0] + history[1])
            torch.testing.assert_close(branch, phase * expected / 4, rtol=0, atol=1e-15)


def test_and_feedback_cannot_replace_later_control() -> None:
    with pytest.raises(ValueError, match="broadcast control"):
        broadcast_del_and_ir((0, Bit(2)), True, (1, 3), measurement_bit=2)
    assert broadcast_init_and_ir("|->", (), 2) == broadcast_ccx_ir("|->", (), 2) == ()
    assert broadcast_del_and_ir("|->", (), 2, measurement_bit=2) == ()
