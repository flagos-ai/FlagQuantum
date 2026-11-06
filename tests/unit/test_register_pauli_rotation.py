"""Verify absolute source phase for axis rotations and Pauli powers."""

from math import pi

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import RotationCall
from flagquantum.compiler.registers.pauli_rotation import (
    PauliAxis,
    pauli_power_ir,
    pauli_rotation_ir,
)
from flagquantum.core._composition import controlled_instruction
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.unit


def apply(gates: tuple[Instruction, ...], state: torch.Tensor) -> torch.Tensor:
    return (
        Circuit(3, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(3, gates))
        .state()
        .flatten()
    )


def source_application(
    matrix: object, state: torch.Tensor, *, controlled: bool
) -> torch.Tensor:
    source = torch.as_tensor(matrix, dtype=torch.complex128)
    if controlled:
        ordered = state.reshape(2, 2, 2).permute(2, 0, 1).reshape(4, 2)
        return (source @ ordered).reshape(2, 2, 2).permute(1, 2, 0).flatten()
    return (source @ state.reshape(2, 4)).flatten()


def coherent_state() -> torch.Tensor:
    state = torch.tensor(
        [complex(i + 1, 3 * i - 7) for i in range(8)], dtype=torch.complex128
    )
    return state / state.norm()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("exponent", [0.0, -0.375, 0.25, 1.0, 2.0])
def test_power_preserves_source_phase_with_and_without_control(
    axis: PauliAxis, exponent: float
) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.basic_gates import XPowGate, YPowGate, ZPowGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    state = coherent_state()
    source_cls = {"x": XPowGate, "y": YPowGate, "z": ZPowGate}[axis]
    for shift in (0.0,) if axis == "z" else (0.0, -0.5, 0.125):
        source = source_cls(
            exponent=exponent, **({} if axis == "z" else {"global_shift": shift})
        )
        for control in (None, 2):
            gates = pauli_power_ir(
                axis, 0, exponent, global_shift=shift, control=control
            )
            reference_gate = (
                source.cirq_gate
                if control is None
                else cirq.ControlledGate(source.cirq_gate)
            )
            expected = source_application(
                cirq.unitary(reference_gate), state, controlled=control is not None
            )
            actual = apply(gates, state)
            torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
            inverse = pauli_power_ir(
                axis, 0, -exponent, global_shift=shift, control=control
            )
            torch.testing.assert_close(
                apply(inverse, actual), state, rtol=0, atol=2e-12
            )
            assert all(gate.name not in {"measure", "reset"} for gate in gates)

    opcodes = {"x": "xpow", "y": "ypow", "z": "zpow"}
    budget = query_qec_gate_cost(RotationCall(opcodes[axis], exponent))
    source_budget = get_cost_value(source_cls(exponent=exponent), QECGatesCost())
    assert (budget.t, budget.clifford, budget.rotation) == (
        source_budget.t,
        source_budget.clifford,
        source_budget.rotation,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("angle", [-0.3, pi / 2, 2 * pi])
def test_radian_rotation_and_core_control_match_source(
    axis: PauliAxis, angle: float
) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.basic_gates import Rx, Ry, Rz

    source = {"x": Rx, "y": Ry, "z": Rz}[axis](angle).cirq_gate
    state = coherent_state()
    gates = pauli_rotation_ir(axis, 0, angle)
    assert gates == (Instruction("r" + axis, (0,), params={"theta": angle}),)
    torch.testing.assert_close(
        apply(gates, state),
        source_application(cirq.unitary(source), state, controlled=False),
        rtol=0,
        atol=2e-12,
    )
    controlled = controlled_instruction(gates[0], (2,))
    torch.testing.assert_close(
        apply(controlled, state),
        source_application(
            cirq.unitary(cirq.ControlledGate(source)), state, controlled=True
        ),
        rtol=0,
        atol=2e-12,
    )


def test_power_gate_sequence_and_input_validation() -> None:
    assert [g.name for g in pauli_power_ir("x", 0, 0.3)] == ["h", "phase", "h"]
    assert [g.name for g in pauli_power_ir("y", 0, 0.3)] == [
        "sdg",
        "h",
        "phase",
        "h",
        "s",
    ]
    assert [g.name for g in pauli_power_ir("z", 0, 0.3, control=2)] == ["cphase"]
    assert [
        g.name for g in pauli_power_ir("x", 0, 0.3, global_shift=0.2, control=2)
    ] == ["h", "cphase", "h", "phase"]
    with pytest.raises(ValueError, match="zero global shift"):
        pauli_power_ir("z", 0, 0.3, global_shift=0.2)
    with pytest.raises(ValueError, match="distinct"):
        pauli_power_ir("x", 0, 0.3, control=0)
    with pytest.raises(ValueError, match="finite"):
        pauli_power_ir("x", 0, 0.3, global_shift=float("inf"))
