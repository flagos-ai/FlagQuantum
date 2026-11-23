"""Bound symbolic Pauli recipes preserve phase, operands and numerical execution."""

from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.primitives.pauli_rotation import (
    PauliAxis,
    pauli_power_ir,
    pauli_rotation_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.parameters import Parameter, ParameterExpression

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("axis", ("x", "y", "z"))
@pytest.mark.parametrize("power", (False, True))
def test_bound_recipe_matches_numeric_gates_and_gradient(
    axis: PauliAxis, power: bool
) -> None:
    theta = Parameter("theta")
    expression = ParameterExpression("sin", (theta,)) + theta * 0.5
    shifts = (0.0,) if axis == "z" or not power else (0.0, -0.5, 0.31)
    initial = torch.tensor([1 + 2j, -3j, 4 + 1j, -2], dtype=torch.complex128)
    initial /= initial.norm()
    for shift in shifts:
        gates = (
            pauli_power_ir(axis, 1, expression, global_shift=shift)
            if power
            else pauli_rotation_ir(axis, 1, expression)
        )
        circuit = Circuit(2, inputs=initial, dtype=initial.dtype).append(
            FieldOperation(2, gates)
        )
        assert circuit.parameter_names == ("theta",)
        assert CircuitIR.from_json(circuit.to_ir().to_json()) == circuit.to_ir()
        value = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
        bound = circuit.bind_parameters({"theta": value})
        numeric_angle = float(expression.bind({"theta": value.detach()}))
        numeric = (
            pauli_power_ir(axis, 1, numeric_angle, global_shift=shift)
            if power
            else pauli_rotation_ir(axis, 1, numeric_angle)
        )
        expected = Circuit(2, inputs=initial, dtype=initial.dtype).append(
            FieldOperation(2, numeric)
        )
        torch.testing.assert_close(bound.state(), expected.state(), rtol=0, atol=3e-12)
        gradient = torch.autograd.grad(bound.state().flatten()[0].real, value)[0]
        angle = expression.bind({"theta": value})
        radians = angle * torch.pi if power else angle
        pauli = {
            "x": torch.tensor([[0, 1], [1, 0]], dtype=initial.dtype),
            "y": torch.tensor([[0, -1j], [1j, 0]], dtype=initial.dtype),
            "z": torch.diag(torch.tensor([1, -1], dtype=initial.dtype)),
        }[axis]
        matrix = (
            torch.cos(radians / 2) * torch.eye(2, dtype=initial.dtype)
            - 1j * torch.sin(radians / 2) * pauli
        )
        if power:
            matrix = torch.exp(1j * radians * (shift + 0.5)) * matrix
        independent = (matrix @ initial.reshape(2, 2).T).T.flatten()
        reference_gradient = torch.autograd.grad(independent[0].real, value)[0]
        torch.testing.assert_close(gradient, reference_gradient, rtol=0, atol=3e-12)


@pytest.mark.parametrize(
    "angle",
    (
        ParameterExpression("bad", (Parameter("t"),)),
        ParameterExpression("sin", (complex(1, 1),)),
    ),
)
def test_malformed_owned_nodes_fail_before_emission(angle: Any) -> None:
    with pytest.raises(ValueError):
        pauli_rotation_ir("x", 0, angle)
    with pytest.raises(ValueError):
        pauli_power_ir("y", 0, angle)


def test_invalid_axis_keeps_existing_validation_for_owned_angles() -> None:
    invalid: Any = "other"
    for angle in (0.37, Parameter("theta")):
        with pytest.raises(ValueError, match="axis"):
            pauli_rotation_ir(invalid, 0, angle)
        with pytest.raises(ValueError, match="axis"):
            pauli_power_ir(invalid, 0, angle)
