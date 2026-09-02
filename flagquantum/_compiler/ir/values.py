"""Deterministic value identities and typed references."""

from __future__ import annotations

from dataclasses import dataclass

from .types import IRType


@dataclass(frozen=True, order=True)
class ValueId:
    """A deterministic identifier allocated within one module scope."""

    index: int
    scope: str = "entry"

    def __post_init__(self) -> None:
        index = int(self.index)
        scope = str(self.scope).strip()
        if index < 0:
            raise ValueError("value index cannot be negative")
        if not scope:
            raise ValueError("value scope cannot be empty")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "scope", scope)

    def __str__(self) -> str:
        return f"%{self.scope}.{self.index}"


@dataclass(frozen=True)
class ValueRef:
    """A typed reference to an operation or block value."""

    id: ValueId
    type: IRType

    @property
    def linear(self) -> bool:
        return self.type.linear

    def canonical(self) -> tuple[object, ...]:
        return (self.id.scope, self.id.index, self.type.canonical())


__all__ = ["ValueId", "ValueRef"]
