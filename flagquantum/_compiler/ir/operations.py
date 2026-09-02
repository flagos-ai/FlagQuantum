"""Immutable operations and semantic attribute encoding."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .values import ValueRef

if TYPE_CHECKING:
    from .modules import Region


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenAttributes):
        return value
    if isinstance(value, Mapping):
        return FrozenAttributes(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise TypeError(
            "unordered set values are not valid deterministic IR attributes"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("IR floating-point attributes must be finite")
    if isinstance(value, complex) and not (
        math.isfinite(value.real) and math.isfinite(value.imag)
    ):
        raise ValueError("IR complex attributes must be finite")
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return value
    canonical = getattr(value, "canonical", None)
    if callable(canonical):
        return value
    raise TypeError(f"unsupported immutable IR attribute type: {type(value).__name__}")


def canonical_value(value: Any) -> Any:
    """Return a JSON-compatible semantic encoding with explicit scalar kinds."""

    if isinstance(value, FrozenAttributes):
        return {key: canonical_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [canonical_value(item) for item in value]
    if isinstance(value, complex):
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", "value": value}
    if value is None or isinstance(value, str):
        return value
    canonical = getattr(value, "canonical", None)
    if callable(canonical):
        return canonical_value(canonical())
    raise TypeError(
        f"IR attribute is not canonically encodable: {type(value).__name__}"
    )


class FrozenAttributes(Mapping[str, Any]):
    """A deeply immutable mapping with deterministic key order."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        source = values or {}
        self._items = tuple(
            (str(key), _freeze(value))
            for key, value in sorted(source.items(), key=lambda item: str(item[0]))
        )
        keys = tuple(key for key, _ in self._items)
        if len(keys) != len(set(keys)):
            raise ValueError(
                "IR attribute keys must be unique after string normalization"
            )

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return f"FrozenAttributes({dict(self._items)!r})"


@dataclass(frozen=True, order=True)
class SourceLocation:
    """Debug provenance excluded from operation and module identity."""

    source: str
    line: int
    column: int = 0

    def __post_init__(self) -> None:
        source = str(self.source)
        line = int(self.line)
        column = int(self.column)
        if not source:
            raise ValueError("source location source cannot be empty")
        if line < 1 or column < 0:
            raise ValueError("source location requires line >= 1 and column >= 0")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "line", line)
        object.__setattr__(self, "column", column)


@dataclass(frozen=True)
class Operation:
    """A generic immutable operation node.

    Structural verification against an :class:`OperationSchema` belongs to
    Batch B. Batch A guarantees deterministic representation only.
    """

    name: str
    operands: tuple[ValueRef, ...] = ()
    results: tuple[ValueRef, ...] = ()
    attributes: FrozenAttributes | Mapping[str, Any] = field(
        default_factory=FrozenAttributes
    )
    regions: tuple[Region, ...] = ()
    location: SourceLocation | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name or "." not in name:
            raise ValueError("operation name must be a qualified name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "attributes", FrozenAttributes(self.attributes))
        object.__setattr__(self, "regions", tuple(self.regions))

    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operands": [item.canonical() for item in self.operands],
            "results": [item.canonical() for item in self.results],
            "attributes": canonical_value(self.attributes),
            "regions": [region.canonical() for region in self.regions],
        }


__all__ = [
    "FrozenAttributes",
    "Operation",
    "SourceLocation",
    "canonical_value",
]
