"""Immutable types for the private Phase 1 QuantumIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _freeze_parameter(value: Any) -> object:
    if isinstance(value, tuple):
        return tuple(_freeze_parameter(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return value
    raise TypeError(f"unsupported IR type parameter: {type(value).__name__}")


@dataclass(frozen=True, order=True)
class IRType:
    """A deterministic structural type.

    ``linear`` is part of the semantic type. A linear value must be consumed
    according to its operation schema; the verifier enforces that in Batch B.
    """

    name: str
    parameters: tuple[object, ...] = ()
    linear: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise ValueError("IR type name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "parameters",
            tuple(_freeze_parameter(parameter) for parameter in self.parameters),
        )

    def canonical(self) -> tuple[object, ...]:
        return (self.name, self.parameters, self.linear)


QUBIT = IRType("qubit", linear=True)
BIT = IRType("bit")
BOOL = IRType("bool")
INDEX = IRType("index")
SCALAR = IRType("scalar")


def tensor_type(dtype: str, shape: tuple[int, ...]) -> IRType:
    """Create a ranked, non-linear tensor type with validated dimensions."""

    normalized_dtype = str(dtype).removeprefix("torch.").strip().lower()
    normalized_shape = tuple(int(size) for size in shape)
    if not normalized_dtype:
        raise ValueError("tensor dtype cannot be empty")
    if any(size < 0 for size in normalized_shape):
        raise ValueError("tensor shape dimensions cannot be negative")
    return IRType("tensor", (normalized_dtype, normalized_shape))


__all__ = ["BIT", "BOOL", "INDEX", "IRType", "QUBIT", "SCALAR", "tensor_type"]
