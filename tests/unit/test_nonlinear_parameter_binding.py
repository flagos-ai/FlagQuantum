"""Real nonlinear parameters bind without losing gradients or old encodings."""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.physical import control_scalar
from flagquantum.core.ir.program.scalar import scalar_literal_error
from flagquantum.core.parameters import Parameter, ParameterExpression

pytestmark = pytest.mark.unit

CASES: tuple[tuple[str, tuple[float, ...], Callable[[float], float]], ...] = (
    ("sin", (), math.sin),
    ("cos", (), math.cos),
    ("exp", (), math.exp),
    ("log", (), math.log),
    ("acos", (), math.acos),
    ("floor", (), math.floor),
    ("pow", (1.5,), lambda x: x**1.5),
    ("div", (3.0,), lambda x: x / 3),
)


@pytest.mark.parametrize("operation,args,evaluate", CASES)
def test_real_scalar_roundtrip_and_structural_admission(
    operation: str, args: tuple[float, ...], evaluate: Callable[[float], float]
) -> None:
    parameter = Parameter("theta")
    expression = ParameterExpression(operation, (parameter, *args))
    assert scalar_literal_error(expression) is None
    assert control_scalar(expression, "amplitude") is expression
    circuit = CircuitIR(1, (Instruction("rz", (0,), {"theta": expression}),))
    restored = CircuitIR.from_json(circuit.to_json())
    assert restored == circuit
    assert restored.content_hash == circuit.content_hash
    for value in ((-0.7, 0.2, 0.9) if operation == "acos" else (0.2, 0.9, 2.3)):
        assert expression.bind({"theta": value}) == pytest.approx(evaluate(value))


@pytest.mark.parametrize("operation,args,evaluate", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensor_binding_preserves_device_dtype_and_two_gradient_orders(
    operation: str,
    args: tuple[float, ...],
    evaluate: Callable[[float], float],
    dtype: torch.dtype,
) -> None:
    del evaluate
    value = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    ignored = torch.tensor(0.4, dtype=dtype, requires_grad=True)
    parameter = Parameter("theta")
    expression = ParameterExpression(operation, (parameter, *args))
    bindings: dict[str | Parameter, torch.Tensor] = {parameter: value, "theta": ignored}
    actual = expression.bind(bindings)
    expected = (
        value ** args[0]
        if operation == "pow"
        else value / args[0] if operation == "div" else getattr(torch, operation)(value)
    )
    assert actual.dtype == value.dtype and actual.device == value.device
    torch.testing.assert_close(actual, expected)
    derivative = torch.autograd.grad(actual, value, create_graph=True)[0]
    expected_derivative = torch.autograd.grad(expected, value, create_graph=True)[0]
    torch.testing.assert_close(derivative, expected_derivative)
    if derivative.requires_grad:
        torch.testing.assert_close(
            torch.autograd.grad(derivative, value)[0],
            torch.autograd.grad(expected_derivative, value)[0],
        )
    assert ignored.grad is None


@pytest.mark.parametrize(
    "operation,args",
    [
        ("log", (0.0,)),
        ("log", (-1.0,)),
        ("div", (1.0, 0.0)),
        ("pow", (-1.0, 0.5)),
        ("pow", (0.0, -1.0)),
        ("exp", (1000.0,)),
        ("sin", (float("nan"),)),
        ("cos", (complex(1, 2),)),
        ("sin", (True,)),
        ("acos", (-1.000000000001,)),
        ("acos", (1.000000000001,)),
        ("floor", (float("inf"),)),
    ],
)
def test_invalid_real_domains_raise_before_execution(
    operation: str, args: tuple[Any, ...]
) -> None:
    expression = ParameterExpression(operation, args)
    with pytest.raises(ValueError):
        expression.bind({})
    symbolic = ParameterExpression(
        operation, tuple(Parameter(str(i)) for i in range(len(args)))
    )
    with pytest.raises(ValueError):
        symbolic.bind({str(i): value for i, value in enumerate(args)})


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([0.2]),
        torch.tensor(True),
        torch.tensor(1 + 2j),
        torch.tensor(float("inf")),
    ],
)
def test_invalid_bound_tensor_is_rejected(value: torch.Tensor) -> None:
    expression = ParameterExpression("sin", (Parameter("theta"),))
    with pytest.raises(ValueError):
        expression.bind({"theta": value})


@pytest.mark.parametrize(
    "operation,count",
    [("sin", 2), ("log", 0), ("pow", 1), ("div", 3), ("acos", 2), ("floor", 0)],
)
def test_wrong_arity_precedes_parameter_binding(operation: str, count: int) -> None:
    expression = ParameterExpression(operation, (Parameter("missing"),) * count)
    with pytest.raises(ValueError, match="operands"):
        expression.bind({})
    assert (
        scalar_literal_error(expression)
        == "scalar expression has invalid immutable operands"
    )


def test_nested_domains_are_validated_after_binding_and_cycles_are_rejected() -> None:
    parameter = Parameter("theta")
    expression = ParameterExpression(
        "log", (ParameterExpression("div", (parameter, 2)),)
    )
    assert scalar_literal_error(expression) is None
    assert expression.bind({"theta": 6}) == pytest.approx(math.log(3))
    with pytest.raises(ValueError):
        expression.bind({"theta": -1})
    recursive = ParameterExpression("exp", (parameter,))
    object.__setattr__(recursive, "args", (recursive,))
    assert scalar_literal_error(recursive) == "scalar expression must be acyclic"
    with pytest.raises(ValueError, match="acyclic"):
        recursive.bind({"theta": 1})


@pytest.mark.parametrize("endpoint", (-1.0, 1.0))
def test_acos_endpoint_value_and_native_singular_derivative(endpoint: float) -> None:
    parameter = Parameter("theta")
    expression = ParameterExpression("acos", (parameter,))
    value = torch.tensor(endpoint, dtype=torch.float64, requires_grad=True)
    actual = expression.bind({"theta": value})
    expected = torch.acos(value)
    torch.testing.assert_close(actual, expected)
    assert expression.bind({"theta": endpoint}) == math.acos(endpoint)
    derivative = torch.autograd.grad(actual, value)[0]
    assert torch.isneginf(derivative)
    assert torch.equal(derivative, torch.autograd.grad(expected, value)[0])


@pytest.mark.parametrize("value", (-2.3, -0.2, 0, 0.9, 2**60 + 1, -(2**60 + 1)))
def test_floor_keeps_python_integer_semantics(value: int | float) -> None:
    result = ParameterExpression("floor", (Parameter("theta"),)).bind({"theta": value})
    assert type(result) is int
    assert result == math.floor(value)


@pytest.mark.parametrize("operation", ("acos", "floor"))
@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_integral_tensor_follows_native_torch_promotion(
    operation: str, dtype: torch.dtype
) -> None:
    value = torch.tensor(0, dtype=dtype)
    result = ParameterExpression(operation, (Parameter("theta"),)).bind(
        {"theta": value}
    )
    expected = getattr(torch, operation)(value)
    assert result.dtype == expected.dtype and result.device == expected.device
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("operation", ("acos", "floor"))
@pytest.mark.parametrize(
    "value",
    (
        torch.tensor([0.2]),
        torch.tensor(True),
        torch.tensor(0.2j),
        torch.tensor(float("nan")),
    ),
)
def test_new_unary_operations_preserve_scalar_input_rejections(
    operation: str, value: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        ParameterExpression(operation, (Parameter("theta"),)).bind({"theta": value})
