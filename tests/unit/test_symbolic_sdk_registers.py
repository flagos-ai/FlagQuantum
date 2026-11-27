"""Frozen parameterized SDK leaves preserve exact phase and measurement histories."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    import cirq
    import numpy as np
    import sympy

    _source_interop = import_module("qualtran.cirq_interop._cirq_to_bloq")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler import estimate_resources
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
from flagquantum.ecosystem.cirq.conversion import _flagquantum_parameter_value
from flagquantum.ecosystem.cirq.models import CirqConversionError
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_feedback_import import source_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _gate(kind: str, angle: Any) -> Any:
    gates: dict[str, Callable[[], Any]] = {
        "rx": lambda: cirq.rx(angle),
        "ry": lambda: cirq.ry(angle),
        "rz": lambda: cirq.rz(angle),
        "rzz": lambda: cirq.ZZPowGate(exponent=angle / sympy.pi, global_shift=-0.5),
        "phase": lambda: cirq.ZPowGate(exponent=angle),
        "cphase": lambda: cirq.CZPowGate(exponent=angle),
        "crx": lambda: cirq.rx(angle).controlled(),
        "cry": lambda: cirq.ry(angle).controlled(),
        "crz": lambda: cirq.rz(angle).controlled(),
    }
    return gates[kind]()


@pytest.mark.parametrize(
    "kind", ["rx", "ry", "rz", "rzz", "phase", "cphase", "crx", "cry", "crz"]
)
def test_parameterized_leaf_matches_frozen_source_before_and_after_resolution(
    kind: str,
) -> None:
    cirq_optree_to_cbloq = _source_interop.cirq_optree_to_cbloq

    theta, other = sympy.symbols("theta other")
    angle = 2 * theta - other + sympy.Rational(1, 8)
    gate = _gate(kind, angle)
    width = cirq.num_qubits(gate)
    qubits = tuple(cirq.LineQubit.range(width + 1))
    operands = tuple(reversed(qubits[:width]))
    source = cirq_optree_to_cbloq(cirq.Circuit(gate.on(*operands)))
    source_circuit = source.to_cirq_circuit(
        cirq_quregs={"qubits": np.array([[q] for q in qubits[:width]], dtype=object)}
    )
    # Frozen CZPow/CRz wrappers omit the Cirq parameter protocol themselves.
    assert cirq.is_parameterized(source_circuit) == (kind not in {"cphase", "crz"})
    source_ops = tuple(source_circuit.all_operations())
    assert len(source_ops) == 1
    assert source_ops[0].qubits == operands
    program = copy_component_program(import_cirq_register_optree(gate.on(*operands)))
    bound = bind_register_program(program, tuple((i,) for i in range(width)))
    assert [(op.name, op.wires, op.metadata) for op in bound.circuit.instructions] == [
        (kind, tuple(q.x for q in operands), {})
    ]
    assert not bound.events
    assert estimate_resources(bound.circuit).operation_counts == {kind: 1}
    registers = program.functions[0].registers
    assert registers is not None
    exported = RegisterCirqGate(bound, Signature(registers))
    restored = import_cirq_register_optree(exported.on(*qubits[:width]))
    rebound = bind_register_program(restored, tuple((i,) for i in range(width)))
    state = np.arange(1, 2 ** (width + 1) + 1, dtype=np.complex128) * np.exp(
        0.31j * np.arange(2 ** (width + 1))
    )
    state /= np.linalg.norm(state)
    for values in ({"theta": 0.37, "other": -0.21}, {"theta": -0.5, "other": 0.75}):
        resolved_gate = cirq.resolve_parameters(gate, values)
        resolved_source = cirq_optree_to_cbloq(
            cirq.Circuit(resolved_gate.on(*operands))
        )
        resolved_circuit = resolved_source.to_cirq_circuit(
            cirq_quregs={
                "qubits": np.array([[q] for q in qubits[:width]], dtype=object)
            }
        )
        if kind == "cphase":
            # Its frozen graph export decomposes into measured And erasure.
            matrix = resolved_source.tensor_contract()
            expected = np.kron(matrix, np.eye(2)) @ state
        else:
            expected = resolved_circuit.unitary(qubit_order=qubits) @ state
        for body in (bound.circuit, rebound.circuit):
            circuit = Circuit(
                width + 1, inputs=torch.from_numpy(state), dtype=torch.complex128
            ).append(FieldOperation(width + 1, body.instructions))
            assert circuit.parameter_names == ("other", "theta")
            actual = circuit.bind_parameters(values).state().flatten()
            torch.testing.assert_close(
                actual, torch.from_numpy(expected), atol=4e-15, rtol=0
            )


@pytest.mark.parametrize("kind", ["rz", "phase", "cphase", "crx"])
def test_symbolic_feedback_every_branch_and_record_reset(kind: str) -> None:
    theta = sympy.Symbol("theta")
    data, control, record, spectator = qubits = tuple(cirq.LineQubit.range(4))
    gate = _gate(kind, theta + sympy.Rational(1, 8))
    operands = (data,) if cirq.num_qubits(gate) == 1 else (control, data)
    source = cirq.Circuit(
        cirq.Moment((op,))
        for op in (
            cirq.H(record),
            cirq.measure(record, key="m"),
            gate.on(*operands).with_classical_controls("m"),
            cirq.Y(data).with_classical_controls("m"),
            cirq.reset(record),
        )
    )
    signature = Signature((Register("q", quantum_bit_type(), (3,)),))
    placement = {"q": np.array([[data], [control], [record]], dtype=object)}
    program = import_cirq_register_optree(
        source, signature=signature, in_quregs=placement, out_quregs=placement
    )
    bound = bind_register_program(program, ((0,), (1,), (2,)))
    registers = program.functions[0].registers
    assert registers is not None
    exported = RegisterCirqGate(
        bound, Signature(registers), feedback_key_prefix="symbolic"
    )
    restored = import_cirq_register_optree(exported.on(data, control, record))
    rebound = bind_register_program(restored, ((0,), (1,), (2,)))
    initial = np.zeros(16, dtype=np.complex128)
    indices = np.array([i for i in range(16) if not i & 2])
    initial[indices] = np.arange(1, 9) * np.exp(0.21j * np.arange(8))
    initial /= np.linalg.norm(initial)
    resolved = cirq.resolve_parameters(source, {"theta": 0.37})
    for body in (bound.circuit, rebound.circuit):
        circuit = (
            Circuit(4, inputs=torch.from_numpy(initial), dtype=torch.complex128)
            .append(FieldOperation(4, body.instructions))
            .bind_parameters({"theta": 0.37})
        )
        physical = circuit.to_ir().instructions
        assert [op.name for op in physical] == ["h", "measure", kind, "y", "reset"]
        assert physical[2].wires == tuple(q.x for q in operands)
        assert physical[2].metadata["conditions"] == ((0, 1),)
        for outcome in (0, 1):
            actual = forced_branch(torch.from_numpy(initial), physical, (outcome,))
            expected = source_branch(resolved, qubits, initial, (outcome,))
            torch.testing.assert_close(
                actual, torch.from_numpy(expected), atol=4e-15, rtol=0
            )
            assert torch.count_nonzero(actual[[i for i in range(16) if i & 2]]) == 0


def test_unsupported_symbolic_forms_keep_explicit_boundary() -> None:
    theta = sympy.Symbol("theta")
    for value in (sympy.gamma(theta), sympy.I * theta):
        with pytest.raises(ValueError):
            _flagquantum_parameter_value(cirq, value)
    unsupported = cirq.Circuit(cirq.rz(sympy.gamma(theta)).on(cirq.LineQubit(0)))
    with pytest.raises(CirqConversionError, match="unsupported_parameter_expression"):
        import_cirq_register_optree(unsupported)


@pytest.mark.parametrize("kind", ("sin", "pow", "reciprocal", "eigen_shift"))
def test_approved_nonlinear_and_eigen_shift_forms_preserve_full_operator(
    kind: str,
) -> None:
    theta = sympy.Symbol("theta")

    def gate(value: Any) -> Any:
        if kind == "eigen_shift":
            return cirq.ZPowGate(exponent=value, global_shift=value)
        expression = {
            "sin": sympy.sin(value),
            "pow": value**2,
            "reciprocal": 1 / value,
        }[kind]
        if not isinstance(value, sympy.Basic):
            expression = float(expression)
        return cirq.rz(expression)

    original = _source_interop.cirq_optree_to_cbloq(
        cirq.Circuit(gate(theta).on(cirq.LineQubit(0)))
    )
    assert cirq.is_parameterized(original.to_cirq_circuit())
    program = import_cirq_register_optree(gate(theta).on(cirq.LineQubit(0)))
    bound = bind_register_program(program, ((0,),))
    for value in (0.37, -0.5):
        circuit = Circuit(1, dtype=torch.complex128).append(
            FieldOperation(1, bound.circuit.instructions)
        )
        resolved = circuit.bind_parameters({"theta": value})
        # The SDK discovers only an eigen-gate exponent, not its symbolic shift.
        # A fresh numeric gate substitutes both fields without editing source.
        torch.testing.assert_close(
            get_unitary(resolved, dtype=torch.complex128),
            torch.as_tensor(cirq.unitary(gate(value))),
            rtol=0,
            atol=2e-12,
        )
