"""Compare direct controlled register export and all original SDK diagrams."""

from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_control import control_register_operation
from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")
q = pytest.importorskip("qualtran")


def ports(signature: Signature) -> dict[str, Any]:
    return {
        register.name: np.array(
            [
                cirq.NamedQubit(f"{register.name}{i}")
                for i in range(int(register.total_qubits()))
            ],
            dtype=object,
        ).reshape((*register.shape, int(register.bitsize)))
        for register in signature
    }


@pytest.mark.parametrize("activations", [(1,), (0,), (0, 1)])
def test_original_x_power_diagrams_phases_and_child_output_identity(
    activations: tuple[int, ...],
) -> None:
    from qualtran.bloqs.basic_gates import XPowGate

    spec = q.CtrlSpec(
        cvs=activations[0] if len(activations) == 1 else list(activations)
    )
    source = XPowGate(0.25).controlled(spec)
    ctrl = Register("ctrl", quantum_bit_type(), () if len(activations) == 1 else (2,))
    signature = Signature((ctrl, Register("q", quantum_bit_type())))
    qubits = ports(signature)
    target = qubits["q"][0]
    child = cirq.XPowGate(exponent=0.25).on(target)
    op, outputs = control_register_operation(
        child,
        {"q": qubits["q"]},
        (ctrl,),
        (np.asarray(spec.cvs[0]),),
        {"ctrl": qubits["ctrl"]},
    )
    original, original_outputs = source.as_cirq_op(cirq.SimpleQubitManager(), **qubits)
    assert (
        cirq.Circuit(op).to_text_diagram() == cirq.Circuit(original).to_text_diagram()
    )
    np.testing.assert_allclose(
        cirq.unitary(op), cirq.unitary(original), rtol=0, atol=1e-15
    )
    assert outputs["q"] is qubits["q"] and outputs["ctrl"] is qubits["ctrl"]
    assert tuple(outputs) == tuple(original_outputs)
    _assert_appended(op, original)


def _assert_appended(operation: Any, reference: Any | None = None) -> None:
    order = tuple(operation.qubits)
    register = Register("q", quantum_bit_type(), (len(order),))
    placement = {"q": np.array(order, dtype=object).reshape(len(order), 1)}
    program = import_cirq_register_optree(
        operation,
        signature=Signature((register,)),
        in_quregs=placement,
        out_quregs=placement,
    )
    native = bind_register_program(program, tuple((i,) for i in range(len(order))))
    if reference is not None:
        reference_program = import_cirq_register_optree(
            reference,
            signature=Signature((register,)),
            in_quregs=placement,
            out_quregs=placement,
        )
        reference_bound = bind_register_program(
            reference_program, tuple((i,) for i in range(len(order)))
        )
        from tests.unit.test_register_subtraction import assert_source

        assert_source(native.circuit.instructions, reference_bound.circuit.instructions)
    width = len(order)
    assert native.circuit.n_wires == width
    size = 1 << width
    state = torch.complex(
        torch.arange(1, 2 * size + 1).double(), torch.arange(2 * size).double()
    )
    state /= state.norm()
    expected = torch.as_tensor(cirq.unitary(operation).copy()) @ state.reshape(size, 2)
    circuit = Circuit(width + 1, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(width + 1, native.circuit.instructions))
    torch.testing.assert_close(
        circuit.state().reshape(size, 2), expected, rtol=0, atol=3e-14
    )


def test_shaped_swap_native_gate_diagram_and_complex_execution() -> None:
    from qualtran.bloqs.basic_gates import Swap

    ctrl = Register("ctrl", quantum_bit_type(), (2,))
    child_signature = Signature(
        (Register("x", qtype := opaque_register_type(2)), Register("y", qtype))
    )
    qubits = ports(Signature((ctrl, *tuple(child_signature))))
    body = CircuitIR(4, (Instruction("swap", (0, 2)), Instruction("swap", (1, 3))))

    def symbol(
        register: Register, index: tuple[int, ...]
    ) -> tuple[Any, dict[str, str | int | bool]]:
        return "TextBox", {"text": f"×({register.name})"}

    child = RegisterCirqGate(body, child_signature, wire_symbol=symbol).on(
        *qubits["x"], *qubits["y"]
    )
    op, outputs = control_register_operation(
        child,
        {"x": qubits["x"], "y": qubits["y"]},
        (ctrl,),
        (np.array([0, 1]),),
        {"ctrl": qubits["ctrl"]},
    )
    source = Swap(2).controlled(q.CtrlSpec(cvs=[0, 1]))
    original, _ = source.as_cirq_op(cirq.SimpleQubitManager(), **qubits)
    # Source as_cirq_op can specialize its child SDK diagram differently from its
    # GateWithRegisters wrapper. Compare the wrapper's original wire protocol.
    wrapper = source.on_registers(**qubits)
    assert cirq.Circuit(op).to_text_diagram() == cirq.Circuit(wrapper).to_text_diagram()
    np.testing.assert_allclose(
        cirq.unitary(op), cirq.unitary(original), rtol=0, atol=1e-15
    )
    assert outputs["x"] is qubits["x"]
    _assert_appended(op, original)


def test_rejected_aliases_and_shapes_leave_inputs_unchanged() -> None:
    ctrl = Register("ctrl", quantum_bit_type())
    a, b = cirq.LineQubit.range(2)
    quregs = {"ctrl": np.array([a], dtype=object)}
    values = (np.array(1),)
    child = cirq.X(b)
    outputs = {"q": np.array([b], dtype=object)}
    for actual_child, actual_outputs, controls, actual_quregs in (
        (None, outputs, (ctrl,), quregs),
        (child, outputs, (ctrl, ctrl), quregs),
        (child, {"ctrl": outputs["q"]}, (ctrl,), quregs),
        (child, outputs, (ctrl,), {"ctrl": np.array([[a]], dtype=object)}),
        (child, outputs, (ctrl,), {"ctrl": np.array([b], dtype=object)}),
    ):
        with pytest.raises(ValueError):
            control_register_operation(
                actual_child, actual_outputs, controls, values, actual_quregs
            )
    assert tuple(quregs) == ("ctrl",) and tuple(outputs) == ("q",)


@pytest.mark.parametrize("power", [1, 2, -1])
def test_native_child_protocol_exponent_and_direct_control(power: int) -> None:
    child_signature = Signature((Register("q", quantum_bit_type()),))
    child_gate = (
        RegisterCirqGate(CircuitIR(1, (Instruction("h", (0,)),)), child_signature)
        ** power
    )
    a, b = cirq.LineQubit.range(2)
    child = child_gate.on(b)
    ctrl = Register("ctrl", quantum_bit_type())
    operation, _ = control_register_operation(
        child,
        {"q": np.array([b], dtype=object)},
        (ctrl,),
        (np.array(0),),
        {"ctrl": np.array([a], dtype=object)},
    )
    expected = child.controlled_by(a, control_values=[0])
    assert (
        cirq.circuit_diagram_info(operation).exponent
        == cirq.circuit_diagram_info(child).exponent
    )
    assert (
        cirq.Circuit(operation).to_text_diagram()
        == cirq.Circuit(expected).to_text_diagram()
    )
    np.testing.assert_allclose(
        cirq.unitary(operation), cirq.unitary(expected), rtol=0, atol=1e-15
    )
    _assert_appended(operation)
