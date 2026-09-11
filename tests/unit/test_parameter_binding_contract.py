"""Parameter binding preserves key precedence and validates expression arity."""

import pytest
import torch

from flagquantum.core.parameters import (
    Parameter,
    ParameterExpression,
    bind_parameter_value,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "neg"])
@pytest.mark.parametrize("count", [0, 3])
def test_expression_rejects_wrong_operand_count(operation: str, count: int) -> None:
    expression = ParameterExpression(operation, (Parameter("missing"),) * count)
    with pytest.raises(ValueError, match="operand"):
        expression.bind({})


def test_unary_expression_rejects_a_second_operand() -> None:
    with pytest.raises(ValueError, match="operand"):
        ParameterExpression("neg", (1.0, 2.0)).bind({})


@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
def test_binary_expression_rejects_a_single_operand(operation: str) -> None:
    with pytest.raises(ValueError, match="operand"):
        ParameterExpression(operation, (1.0,)).bind({})


def test_unsupported_operation_is_rejected_before_binding() -> None:
    with pytest.raises(ValueError, match="Unsupported parameter expression op"):
        ParameterExpression("unknown", (Parameter("missing"),)).bind({})


def test_subtraction_and_negation_preserve_arithmetic() -> None:
    parameter = Parameter("angle")
    assert (3.0 - parameter).bind({"angle": 0.5}) == 2.5
    assert (parameter - 3.0).bind({"angle": 0.5}) == -2.5
    assert (-parameter).bind({"angle": 0.5}) == -0.5


def test_object_binding_takes_precedence_and_preserves_tensor_gradients() -> None:
    parameter = Parameter("angle")
    value = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    ignored = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    bindings: dict[str | Parameter, torch.Tensor] = {parameter: value, "angle": ignored}
    original = {"expression": (parameter * 2.0 + 1.0,), "direct": [parameter]}
    bound = bind_parameter_value(original, bindings)
    assert bound["direct"][0] is value
    assert original["direct"][0] is parameter
    bound["expression"][0].backward()
    torch.testing.assert_close(value.grad, torch.tensor(2.0, dtype=torch.float64))
    assert ignored.grad is None
    assert bindings[parameter] is value


def test_string_binding_and_missing_parameter_diagnostic() -> None:
    parameter = Parameter("angle")
    assert parameter.bind({"angle": 0.2}) == 0.2
    with pytest.raises(KeyError, match="Missing value for parameter 'angle'"):
        parameter.bind({})
