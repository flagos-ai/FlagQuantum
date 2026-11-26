"""Named circuit parameters for FlagQuantum."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, TypeVar, cast

import torch

_EXPRESSION_ARITIES = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "neg": 1,
    "sin": 1,
    "cos": 1,
    "exp": 1,
    "log": 1,
    "acos": 1,
    "floor": 1,
    "pow": 2,
    "div": 2,
}

_ParameterKey = TypeVar("_ParameterKey", bound="str | Parameter")


@dataclass(frozen=True)
class Parameter:
    """A named scalar placeholder used in parameterized circuits."""

    name: str

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("Parameter name cannot be empty.")
        object.__setattr__(self, "name", str(self.name))

    def bind(self, values: Mapping[_ParameterKey, Any]) -> Any:
        # Membership establishes that this key can be looked up in the mapping.
        if self in values:
            return values[cast(_ParameterKey, self)]
        if self.name in values:
            return values[cast(_ParameterKey, self.name)]
        raise KeyError(f"Missing value for parameter {self.name!r}.")

    def __add__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("add", (self, other))

    def __radd__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("add", (other, self))

    def __sub__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("sub", (self, other))

    def __rsub__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("sub", (other, self))

    def __mul__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("mul", (self, other))

    def __rmul__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("mul", (other, self))

    def __neg__(self) -> "ParameterExpression":
        return ParameterExpression("neg", (self,))


@dataclass(frozen=True)
class ParameterExpression:
    """An immutable scalar expression over named circuit parameters.

    Arithmetic nodes retain their existing binding behavior. Real nonlinear
    nodes support sin, cos, exp, log, acos, floor, pow and div, with finite
    scalar domains.
    """

    op: str
    args: tuple[Any, ...]

    def bind(self, values: Mapping[_ParameterKey, Any]) -> Any:
        """Resolve a valid expression, rejecting malformed operands with ValueError."""
        if not isinstance(self.op, str) or self.op not in _EXPRESSION_ARITIES:
            raise ValueError(f"Unsupported parameter expression op {self.op!r}.")
        expected_operands = _EXPRESSION_ARITIES[self.op]
        if len(self.args) != expected_operands:
            raise ValueError(
                f"Parameter expression {self.op!r} requires {expected_operands} "
                f"operands; got {len(self.args)}."
            )
        nonlinear = self.op in {
            "sin",
            "cos",
            "exp",
            "log",
            "acos",
            "floor",
            "pow",
            "div",
        }
        if nonlinear:
            error = _parameter_literal_error(self)
            if error:
                raise ValueError(error)
        bound = tuple(bind_parameter_value(arg, values) for arg in self.args)
        if nonlinear:
            return _bind_real_expression(self.op, bound)
        if self.op == "add":
            return bound[0] + bound[1]
        if self.op == "sub":
            return bound[0] - bound[1]
        if self.op == "mul":
            return bound[0] * bound[1]
        return -bound[0]

    def __add__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("add", (self, other))

    def __radd__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("add", (other, self))

    def __sub__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("sub", (self, other))

    def __rsub__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("sub", (other, self))

    def __mul__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("mul", (self, other))

    def __rmul__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("mul", (other, self))

    def __neg__(self) -> "ParameterExpression":
        return ParameterExpression("neg", (self,))


def _require_real_scalar(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool:
            raise ValueError("Nonlinear expression requires real scalar operands")
        finite = bool(torch.isfinite(value))
    elif isinstance(value, Real) and not isinstance(value, bool):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
    else:
        raise ValueError("Nonlinear expression requires real scalar operands")
    if not finite:
        raise ValueError("Nonlinear expression requires finite real values")


def _bind_real_expression(operation: str, operands: tuple[Any, ...]) -> Any:
    for operand in operands:
        _require_real_scalar(operand)
    left = operands[0]
    if operation == "acos" and (left < -1 or left > 1):
        raise ValueError("Inverse cosine requires a real operand in [-1, 1]")
    if operation == "log" and left <= 0:
        raise ValueError("Logarithm requires a positive real operand")
    if operation == "div" and operands[1] == 0:
        raise ValueError("Division requires a nonzero denominator")
    try:
        if operation == "div":
            result = left / operands[1]
        elif operation == "pow":
            right = operands[1]
            result = (
                torch.pow(left, right)
                if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor)
                else left**right
            )
        else:
            owner = torch if isinstance(left, torch.Tensor) else math
            result = getattr(owner, operation)(left)
    except (OverflowError, ZeroDivisionError, RuntimeError) as error:
        raise ValueError(
            "Nonlinear expression is outside its finite real domain"
        ) from error
    _require_real_scalar(result)
    return result


def _parameter_literal_error(value: object) -> str | None:
    """Validate a scalar expression structurally, without assigning dummy values."""
    pending = [(value, frozenset[int]())]
    while pending:
        item, ancestors = pending.pop()
        if isinstance(item, Parameter):
            if not isinstance(item.name, str) or not item.name:
                return "scalar parameter requires a nonempty name"
        elif isinstance(item, ParameterExpression):
            if id(item) in ancestors:
                return "scalar expression must be acyclic"
            if not isinstance(item.op, str) or item.op not in _EXPRESSION_ARITIES:
                return "scalar expression uses an unsupported operation"
            if (
                not isinstance(item.args, tuple)
                or len(item.args) != _EXPRESSION_ARITIES[item.op]
            ):
                return "scalar expression has invalid immutable operands"
            pending.extend((arg, ancestors | {id(item)}) for arg in item.args)
        elif isinstance(item, (int, float)) and type(item) in (int, float):
            try:
                if not math.isfinite(item):
                    return "scalar numeric leaves must be finite"
            except OverflowError:
                return "scalar numeric leaves must be finite"
        else:
            return "scalar literal requires a number or an owned parameter expression"
    return None


def is_parameterized_value(value: Any) -> bool:
    """Return whether *value* contains an unbound named parameter."""

    if isinstance(value, (Parameter, ParameterExpression)):
        return True
    if isinstance(value, Mapping):
        return any(is_parameterized_value(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(is_parameterized_value(item) for item in value)
    return False


def bind_parameter_value(value: Any, values: Mapping[_ParameterKey, Any]) -> Any:
    """Bind parameters recursively inside a scalar/list/mapping value."""

    if isinstance(value, Parameter):
        return value.bind(values)
    if isinstance(value, ParameterExpression):
        return value.bind(values)
    if isinstance(value, Mapping):
        return {key: bind_parameter_value(item, values) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(bind_parameter_value(item, values) for item in value)
    if isinstance(value, list):
        return [bind_parameter_value(item, values) for item in value]
    return value


def parameter_names_in_value(value: Any) -> tuple[str, ...]:
    """Collect named parameters contained in *value*."""

    names: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Parameter):
            names.add(item.name)
        elif isinstance(item, ParameterExpression):
            for arg in item.args:
                visit(arg)
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(sorted(names))


def value_to_tensor(
    value: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert a bound scalar value to a tensor with a helpful symbolic error."""

    if is_parameterized_value(value):
        names = ", ".join(parameter_names_in_value(value))
        raise ValueError(
            f"Unbound circuit parameter(s): {names}. Call bind_parameters first."
        )
    if isinstance(value, torch.Tensor):
        if device is None and dtype is None:
            return value
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


__all__ = [
    "Parameter",
    "ParameterExpression",
    "bind_parameter_value",
    "is_parameterized_value",
    "parameter_names_in_value",
    "value_to_tensor",
]
