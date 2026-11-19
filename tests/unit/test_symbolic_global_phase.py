"""Owned scalar phases retain arbitrary-state amplitudes and exact gate order."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.primitives.global_phase import scalar_phase_ir
from flagquantum.core.parameters import Parameter, ParameterExpression

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("operation", ("sin", "cos", "exp", "log", "pow", "div"))
def test_symbolic_scalar_phase_preserves_entangled_state_and_gradient(
    operation: str,
) -> None:
    theta = Parameter("theta")
    args = {"pow": (theta, 0.5), "div": (1.0, theta)}.get(operation, (theta,))
    angle = ParameterExpression(operation, args)
    gates = scalar_phase_ir(1, angle)
    assert [(gate.name, gate.wires) for gate in gates] == [
        ("x", (1,)),
        ("phase", (1,)),
        ("x", (1,)),
        ("phase", (1,)),
    ]
    initial = torch.tensor([1, 0, 0, 1j], dtype=torch.complex128) / 2**0.5
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(2, gates)
    )
    value = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    actual = circuit.bind_parameters({"theta": value}).state().flatten()
    expected = torch.exp(1j * angle.bind({"theta": value})) * initial
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=0)
    gradient = torch.autograd.grad(actual[0].real, value, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(expected[0].real, value)[0]
    torch.testing.assert_close(gradient, reference_gradient, atol=2e-15, rtol=0)


def test_symbolic_phase_structure_is_validated_without_dummy_binding() -> None:
    with pytest.raises(ValueError, match="invalid immutable operands"):
        scalar_phase_ir(0, ParameterExpression("log", ()))
    assert (
        len(scalar_phase_ir(0, ParameterExpression("log", (Parameter("theta"),)))) == 4
    )
    assert scalar_phase_ir(0, 0.0) == ()


@pytest.mark.parametrize("operation", ("sin", "cos", "exp", "log", "pow", "div"))
@pytest.mark.parametrize("sdk", ("cirq", "qiskit"))
def test_sdk_global_phase_round_trip_keeps_full_complex_amplitudes(
    operation: str, sdk: str
) -> None:
    import numpy as np

    from flagquantum.core.ir import CircuitIR
    from flagquantum.ecosystem.cirq import from_cirq, to_cirq
    from flagquantum.ecosystem.qiskit import from_qiskit, to_qiskit
    from tests.unit.test_nonlinear_sdk_parameters import _sympy

    if sdk == "cirq":
        cirq = pytest.importorskip("cirq")
        sympy = pytest.importorskip("sympy")
        q0, q1 = cirq.LineQubit.range(2)
        angle = _sympy(operation, sympy.Symbol("theta"))
        source = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.global_phase_operation(sympy.exp(sympy.I * angle)),
            strategy=cirq.InsertStrategy.NEW,
        )
        imported = from_cirq(source)
        restored = to_cirq(CircuitIR.from_json(imported.to_json()))
        expected = cirq.unitary(cirq.resolve_parameters(source, {"theta": 0.37}))[:, 0]
        external = cirq.unitary(cirq.resolve_parameters(restored, {"theta": 0.37}))[
            :, 0
        ]
    else:
        qiskit = pytest.importorskip("qiskit")
        module = pytest.importorskip("qiskit.circuit")
        quantum_info = pytest.importorskip("qiskit.quantum_info")
        theta = module.Parameter("theta")
        angle = (
            theta**0.5
            if operation == "pow"
            else (1 / theta if operation == "div" else getattr(theta, operation)())
        )
        source = qiskit.QuantumCircuit(2)
        source.h(0)
        source.cx(0, 1)
        source.global_phase = angle
        imported = from_qiskit(source)
        assert imported.metadata["interop"]["global_phase"] is None
        restored = to_qiskit(CircuitIR.from_json(imported.to_json()))
        expected = (
            quantum_info.Statevector(source.assign_parameters({"theta": 0.37}))
            .data.reshape(2, 2)
            .T.flatten()
        )
        external = (
            quantum_info.Statevector(restored.assign_parameters({"theta": 0.37}))
            .data.reshape(2, 2)
            .T.flatten()
        )
    assert [gate.name for gate in imported.instructions] == [
        "h",
        "cx",
        "x",
        "phase",
        "x",
        "phase",
    ]
    actual = (
        Circuit.from_ir(imported, dtype=torch.complex128)
        .bind_parameters({"theta": 0.37})
        .state()
        .flatten()
    )
    np.testing.assert_allclose(external, expected, atol=3e-15, rtol=0)
    torch.testing.assert_close(actual, torch.as_tensor(expected), atol=3e-15, rtol=0)


@pytest.mark.parametrize("operation", ("sin", "cos", "exp", "log", "pow", "div"))
@pytest.mark.parametrize("polarity", (0, 1))
def test_symbolic_phase_measurement_feedback_all_branches(
    operation: str, polarity: int
) -> None:
    import numpy as np

    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_cirq_feedback_import import source_branch
    from tests.unit.test_nonlinear_sdk_parameters import _sympy

    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    order = tuple(cirq.LineQubit.range(3))
    q0, q1, record = order
    angle = _sympy(operation, sympy.Symbol("theta"))
    condition = cirq.SympyCondition(sympy.Eq(sympy.Symbol("m"), polarity))
    source = cirq.Circuit(
        cirq.H(record),
        cirq.measure(record, key="m"),
        cirq.global_phase_operation(sympy.exp(sympy.I * angle)).with_classical_controls(
            condition
        ),
        cirq.reset(record),
        strategy=cirq.InsertStrategy.NEW,
    )
    imported = import_cirq_feedback(source, qubit_order=order).ir
    bound = Circuit.from_ir(imported).bind_parameters({"theta": 0.37}).to_ir()
    resolved = cirq.resolve_parameters(source, {"theta": 0.37})
    initial = np.array([1, 0, 0, 0, 0, 0, 1j, 0], dtype=np.complex128) / 2**0.5
    for outcome in (0, 1):
        expected = source_branch(resolved, order, initial, (outcome,))
        actual = forced_branch(torch.as_tensor(initial), bound.instructions, (outcome,))
        torch.testing.assert_close(
            actual, torch.as_tensor(expected), atol=3e-15, rtol=0
        )
        assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize("values", ((1,), (0,), (1, 1), (1, 0), (0, 1), (0, 0)))
def test_symbolic_controlled_global_phase_through_register_import(
    values: tuple[int, ...],
) -> None:
    import numpy as np

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree

    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    controls = len(values)
    qubits = tuple(cirq.LineQubit.range(controls))
    source = cirq.global_phase_operation(
        sympy.exp(sympy.I * sympy.sin(sympy.Symbol("theta")))
    ).controlled_by(*qubits, control_values=values)
    program = import_cirq_register_optree(source)
    bound = bind_register_program(program, tuple((i,) for i in range(controls)))
    zeros = sum(value == 0 for value in values)
    assert [gate.name for gate in bound.circuit.instructions] == ["x"] * zeros + [
        "phase" if controls == 1 else "cphase"
    ] + ["x"] * zeros
    resolved_program = import_cirq_register_optree(
        cirq.resolve_parameters(source, {"theta": 0.37})
    )
    resolved_bound = bind_register_program(
        resolved_program, tuple((i,) for i in range(controls))
    )
    native_resolved = (
        Circuit.from_ir(bound.circuit).bind_parameters({"theta": 0.37}).to_ir()
    )
    assert [(g.name, g.wires) for g in native_resolved.instructions] == [
        (g.name, g.wires) for g in resolved_bound.circuit.instructions
    ]
    for left, right in zip(
        native_resolved.instructions, resolved_bound.circuit.instructions, strict=True
    ):
        if left.params:
            assert left.params["theta"] == pytest.approx(
                right.params["theta"], abs=2e-15
            )
    initial = np.arange(1, 2 ** (controls + 1) + 1).astype(np.complex128) * (1 + 1j)
    initial /= np.linalg.norm(initial)
    actual = (
        Circuit(controls + 1, inputs=torch.as_tensor(initial), dtype=torch.complex128)
        .append(FieldOperation(controls + 1, bound.circuit.instructions))
        .bind_parameters({"theta": 0.37})
        .state()
        .flatten()
    )
    expected = (
        np.kron(
            cirq.unitary(cirq.resolve_parameters(source, {"theta": 0.37})), np.eye(2)
        )
        @ initial
    )
    torch.testing.assert_close(actual, torch.as_tensor(expected), atol=3e-15, rtol=0)


def test_unknown_modulus_and_nonreal_phase_are_rejected() -> None:
    from flagquantum.ecosystem.cirq.conversion import _global_phase_angle

    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    theta = sympy.Symbol("theta")
    for coefficient in (
        theta,
        2,
        sympy.exp(theta),
        sympy.exp(sympy.I * sympy.tan(theta)),
    ):
        with pytest.raises(ValueError):
            _global_phase_angle(cirq, coefficient)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("assumption", (False, True))
@pytest.mark.parametrize("operation", ("sin", "cos", "exp", "log", "pow", "div"))
def test_frozen_composite_nonlinear_leaf_and_symbolic_scalar_workflow(
    assumption: bool, operation: str
) -> None:
    from hashlib import sha1
    from pathlib import Path

    import numpy as np

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
    from tests.unit.test_nonlinear_sdk_parameters import _sympy

    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    pytest.importorskip("qualtran")
    from qualtran.cirq_interop._cirq_to_bloq import cirq_optree_to_cbloq

    raw = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/_infra/composite_bloq.py"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "b0317f675c96f9cce1581fcbb7030c48ab32a529"
    )
    symbol = (
        sympy.Symbol("theta", positive=True) if assumption else sympy.Symbol("theta")
    )
    expression = _sympy(operation, symbol)
    q0 = cirq.LineQubit(0)
    original = cirq.Circuit(cirq.rz(expression)(q0))
    reference = cirq_optree_to_cbloq(original).to_cirq_circuit()
    # Preserve actual symbol identity for assumption-bearing source parameters.
    assignment = {symbol: 0.37}
    expected = cirq.unitary(cirq.resolve_parameters(reference, assignment))
    program = import_cirq_register_optree(original)
    bound = bind_register_program(program, ((0,),))
    assert [(gate.name, gate.wires) for gate in bound.circuit.instructions] == [
        ("rz", (0,))
    ]
    initial = np.array([1, 2j], dtype=np.complex128) / 5**0.5
    actual = (
        Circuit(1, inputs=torch.as_tensor(initial), dtype=torch.complex128)
        .append(FieldOperation(1, bound.circuit.instructions))
        .bind_parameters({"theta": 0.37})
        .state()
        .flatten()
    )
    torch.testing.assert_close(
        actual, torch.as_tensor(expected @ initial), atol=3e-15, rtol=0
    )
    phase = cirq.Circuit(cirq.global_phase_operation(sympy.exp(sympy.I * expression)))
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    modeled = GlobalPhase(exponent=expression / sympy.pi)
    assert get_cost_value(modeled, QECGatesCost()).asdict() == {}
    assert get_cost_value(modeled, QubitCount()) == 0
    phase_program = import_cirq_register_optree(phase)
    phase_bound = bind_register_program(phase_program, (), clean=(0,))
    assert [(event.kind, event.wires) for event in phase_bound.events] == [
        ("allocate", (0,)),
        ("free", (0,)),
    ]
    phase_state = (
        Circuit(1, dtype=torch.complex128)
        .append(FieldOperation(1, phase_bound.circuit.instructions))
        .bind_parameters({"theta": 0.37})
        .state()
        .flatten()
    )
    coefficient = complex(sympy.exp(sympy.I * expression).subs(assignment))
    torch.testing.assert_close(
        phase_state,
        torch.tensor([coefficient, 0], dtype=torch.complex128),
        atol=3e-15,
        rtol=0,
    )
