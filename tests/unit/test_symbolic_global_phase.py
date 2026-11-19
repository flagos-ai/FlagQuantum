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
