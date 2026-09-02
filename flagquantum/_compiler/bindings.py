"""Immutable parameter descriptors and external runtime binding storage."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolicParameter:
    name: str
    attribute_kind: str = "symbolic"

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ValueError("symbolic parameter name cannot be empty")
        object.__setattr__(self, "name", name)

    def canonical(self) -> tuple[str, str]:
        return ("parameter", self.name)


@dataclass(frozen=True)
class SymbolicExpression:
    op: str
    args: tuple[object, ...]
    attribute_kind: str = "symbolic"

    def __post_init__(self) -> None:
        op = str(self.op).strip().lower()
        if op not in {"add", "sub", "mul", "neg"}:
            raise ValueError(f"unsupported symbolic expression op {op!r}")
        expected = 1 if op == "neg" else 2
        args = tuple(self.args)
        if len(args) != expected:
            raise ValueError(f"symbolic expression {op!r} expects {expected} args")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "args", args)

    def canonical(self) -> tuple[object, ...]:
        return ("expression", self.op, self.args)


@dataclass(frozen=True)
class RuntimeBindingRef:
    slot: str
    shape: tuple[int, ...]
    dtype: str
    device_policy: str = "preserve"
    attribute_kind: str = "binding"

    def __post_init__(self) -> None:
        slot = str(self.slot).strip()
        shape = tuple(int(size) for size in self.shape)
        dtype = str(self.dtype).removeprefix("torch.")
        if not slot or not dtype or any(size < 0 for size in shape):
            raise ValueError("invalid runtime binding descriptor")
        object.__setattr__(self, "slot", slot)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)

    def canonical(self) -> tuple[object, ...]:
        return ("binding", self.slot, self.shape, self.dtype, self.device_policy)


class BindingTable(Mapping[str, Any]):
    """Immutable slot structure retaining original runtime objects by identity."""

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[tuple[RuntimeBindingRef, Any], ...] = ()) -> None:
        normalized = tuple(entries)
        slots = tuple(reference.slot for reference, _ in normalized)
        if len(slots) != len(set(slots)):
            raise ValueError("runtime binding slots must be unique")
        self._entries = normalized

    def __getitem__(self, slot: str) -> Any:
        for reference, value in self._entries:
            if reference.slot == slot:
                return value
        raise KeyError(slot)

    def __iter__(self) -> Iterator[str]:
        return (reference.slot for reference, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def references(self) -> tuple[RuntimeBindingRef, ...]:
        return tuple(reference for reference, _ in self._entries)


__all__ = [
    "BindingTable",
    "RuntimeBindingRef",
    "SymbolicExpression",
    "SymbolicParameter",
]
