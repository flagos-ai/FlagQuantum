"""Approved real expressions retain complex amplitudes across SDK boundaries."""

from typing import Any

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter, ParameterExpression
from flagquantum.ecosystem.cirq import from_cirq, to_cirq
from flagquantum.ecosystem.qiskit import from_qiskit, to_qiskit

pytestmark = pytest.mark.unit
np = pytest.importorskip("numpy")
OPERATIONS = ("sin", "cos", "exp", "log", "pow", "div", "acos")
SDK_CASES = [
    (operation, sdk) for operation in OPERATIONS for sdk in ("cirq", "qiskit")
] + [("floor", "cirq")]


def _owned(operation: str) -> ParameterExpression:
    theta = Parameter("theta")
    args = {"pow": (theta, 0.5), "div": (1.0, theta)}.get(operation, (theta,))
    return ParameterExpression(operation, args)


def _sympy(operation: str, theta: Any) -> Any:
    sympy = pytest.importorskip("sympy")
    if operation == "pow":
        return theta ** sympy.Rational(1, 2)
    if operation == "div":
        return 1 / theta
    return getattr(sympy, operation)(theta)


@pytest.mark.parametrize("operation,sdk", SDK_CASES)
def test_nonlinear_sdk_round_trip_preserves_entangled_amplitudes_and_gradient(
    operation: str, sdk: str
) -> None:
    if sdk == "cirq":
        cirq = pytest.importorskip("cirq")
        sympy = pytest.importorskip("sympy")
        q0, q1 = cirq.LineQubit.range(2)
        source = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.rz(_sympy(operation, sympy.Symbol("theta")))(q0),
        )
        imported = from_cirq(source)
        restored = to_cirq(CircuitIR.from_json(imported.to_json()))
        expected = cirq.unitary(cirq.resolve_parameters(source, {"theta": 0.37}))[:, 0]
        actual = cirq.unitary(cirq.resolve_parameters(restored, {"theta": 0.37}))[:, 0]
    else:
        qiskit = pytest.importorskip("qiskit")
        circuit_module = pytest.importorskip("qiskit.circuit")
        quantum_info = pytest.importorskip("qiskit.quantum_info")
        theta = circuit_module.Parameter("theta")
        angle = (
            theta**0.5
            if operation == "pow"
            else (
                1 / theta
                if operation == "div"
                else getattr(theta, "arccos" if operation == "acos" else operation)()
            )
        )
        source = qiskit.QuantumCircuit(2)
        source.h(0)
        source.cx(0, 1)
        source.rz(angle, 0)
        imported = from_qiskit(source)
        restored = to_qiskit(CircuitIR.from_json(imported.to_json()))
        expected = quantum_info.Statevector(
            source.assign_parameters({"theta": 0.37})
        ).data
        actual = quantum_info.Statevector(
            restored.assign_parameters({"theta": 0.37})
        ).data
        expected = expected.reshape(2, 2).T.flatten()
        actual = actual.reshape(2, 2).T.flatten()
    assert [instruction.name for instruction in imported.instructions] == [
        "h",
        "cx",
        "rz",
    ]
    assert imported.n_wires == 2
    native = Circuit.from_ir(imported, dtype=torch.complex128)
    value = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    state = native.bind_parameters({"theta": value}).state().flatten()
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
    torch.testing.assert_close(state, torch.as_tensor(expected), atol=2e-15, rtol=0)
    angle_tensor = _owned(operation).bind({"theta": value})
    expected_real = torch.cos(angle_tensor / 2) / 2**0.5
    gradient = torch.autograd.grad(state[0].real, value, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected_real, value)[0]
    torch.testing.assert_close(gradient, expected_gradient, atol=2e-15, rtol=0)


@pytest.mark.parametrize("operation,sdk", SDK_CASES)
def test_owned_nonlinear_export_uses_existing_rotation_without_added_gates(
    operation: str, sdk: str
) -> None:
    ir = CircuitIR(1, (Instruction("rz", (0,), {"theta": _owned(operation)}),))
    if sdk == "cirq":
        cirq = pytest.importorskip("cirq")
        exported = to_cirq(ir)
        assert len(tuple(exported.all_operations())) == 1
        restored = from_cirq(exported)
        angle = (
            float(
                next(
                    cirq.resolve_parameters(exported, {"theta": 0.37}).all_operations()
                ).gate.exponent
            )
            * np.pi
        )
    else:
        pytest.importorskip("qiskit")
        exported = to_qiskit(ir)
        assert len(exported.data) == 1
        restored = from_qiskit(exported)
        angle = float(
            exported.assign_parameters({"theta": 0.37}).data[0].operation.params[0]
        )
    expected = _owned(operation).bind({"theta": 0.37})
    assert angle == pytest.approx(expected, abs=2e-15)
    assert restored.instructions[0].params["theta"].bind(
        {"theta": 0.37}
    ) == pytest.approx(expected, abs=2e-15)


def test_qiskit_symbolic_floor_keeps_explicit_diagnostic_and_bound_export() -> None:
    pytest.importorskip("qiskit")
    from flagquantum.ecosystem.qiskit import QiskitConversionError

    ir = CircuitIR(1, (Instruction("rz", (0,), {"theta": _owned("floor")}),))
    with pytest.raises(QiskitConversionError) as raised:
        to_qiskit(ir)
    assert [issue.code for issue in raised.value.report.issues] == [
        "unsupported_parameter_expression"
    ]
    assert "floor" in raised.value.report.issues[0].message
    bound = Circuit.from_ir(ir).bind_parameters({"theta": -0.37})
    exported = to_qiskit(bound)
    assert len(exported.data) == 1
    assert float(exported.data[0].operation.params[0]) == -1.0


def test_distinct_assumption_symbols_in_separate_gates_remain_rejected() -> None:
    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    from flagquantum.ecosystem.cirq import CirqConversionError
    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
    from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree

    qubit = cirq.LineQubit(0)
    source = cirq.Circuit(
        cirq.rx(sympy.sin(sympy.Symbol("theta")))(qubit),
        cirq.rz(sympy.cos(sympy.Symbol("theta", positive=True)))(qubit),
    )
    for convert in (from_cirq, import_cirq_register_optree):
        with pytest.raises(CirqConversionError, match="distinct Cirq symbols"):
            convert(source)
    with pytest.raises(CirqConversionError, match="distinct Cirq symbols"):
        import_cirq_feedback(source, qubit_order=(qubit,))
