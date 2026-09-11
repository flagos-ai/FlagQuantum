"""Named circuit parameters for FlagQuantum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeVar, cast

import torch

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
    """A small symbolic expression over named circuit parameters."""

    op: str
    args: tuple[Any, ...]

    def bind(self, values: Mapping[_ParameterKey, Any]) -> Any:
        """Resolve a valid expression, rejecting malformed operands with ValueError."""
        if self.op not in ("add", "sub", "mul", "neg"):
            raise ValueError(f"Unsupported parameter expression op {self.op!r}.")
        expected_operands = 1 if self.op == "neg" else 2
        if len(self.args) != expected_operands:
            raise ValueError(
                f"Parameter expression {self.op!r} requires {expected_operands} "
                f"operands; got {len(self.args)}."
            )
        bound = tuple(bind_parameter_value(arg, values) for arg in self.args)
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
