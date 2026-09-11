"""Immutable private model for structured hybrid programs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    raise TypeError(f"unsupported hybrid IR attribute value: {type(value).__name__}")


def _encode(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, IRType):
        return value.to_data()
    if isinstance(value, Value):
        return value.to_data()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported hybrid IR canonical value: {type(value).__name__}")


@dataclass(frozen=True)
class IRType:
    """A structural value type used by the private program IR."""

    kind: str
    parameters: tuple[Any, ...] = ()
    linear: bool = False

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("IR type kind cannot be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parameters", tuple(_freeze(self.parameters)))

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameters": _encode(self.parameters),
            "linear": self.linear,
        }


BOOL = IRType("bool")
INDEX = IRType("index")
QUANTUM_EFFECT = IRType("quantum.effect", linear=True)


def scalar_type(dtype: str) -> IRType:
    normalized = str(dtype).strip().removeprefix("torch.")
    if not normalized:
        raise ValueError("scalar dtype cannot be empty")
    return IRType("scalar", (normalized,))


def tensor_type(dtype: str, shape: Sequence[int | None]) -> IRType:
    normalized = str(dtype).strip().removeprefix("torch.")
    if not normalized:
        raise ValueError("tensor dtype cannot be empty")
    dimensions = tuple(None if size is None else int(size) for size in shape)
    if any(size is not None and size < 0 for size in dimensions):
        raise ValueError(f"tensor dimensions must be non-negative: {dimensions}")
    return IRType("tensor", (normalized, dimensions))


@dataclass(frozen=True, order=True)
class ValueId:
    """Stable identity of one SSA definition."""

    scope: str
    index: int

    def __post_init__(self) -> None:
        scope = str(self.scope).strip()
        if not scope:
            raise ValueError("value scope cannot be empty")
        if int(self.index) < 0:
            raise ValueError("value index must be non-negative")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "index", int(self.index))

    def to_data(self) -> dict[str, Any]:
        return {"scope": self.scope, "index": self.index}


@dataclass(frozen=True)
class Value:
    """A typed reference to an SSA definition."""

    id: ValueId
    type: IRType

    def to_data(self) -> dict[str, Any]:
        return {"id": self.id.to_data(), "type": self.type.to_data()}


@dataclass(frozen=True)
class SourceLocation:
    """Diagnostic-only source position excluded from semantic identity."""

    filename: str
    line: int
    column: int = 0

    def __post_init__(self) -> None:
        filename = str(self.filename).strip()
        line = int(self.line)
        column = int(self.column)
        if not filename:
            raise ValueError("source filename cannot be empty")
        if line < 1 or column < 0:
            raise ValueError("source location must use a positive line and column >= 0")
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "line", line)
        object.__setattr__(self, "column", column)


@dataclass(frozen=True)
class Operation:
    """One qualified operation with operands, results, and nested regions."""

    name: str
    operands: tuple[Value, ...] = ()
    results: tuple[Value, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    regions: tuple[Region, ...] = ()
    location: SourceLocation | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name or "." not in name:
            raise ValueError(
                "operation name must be qualified, for example 'arith.add'"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "attributes", _freeze(self.attributes))
        object.__setattr__(self, "regions", tuple(self.regions))

    def to_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operands": [_encode(value) for value in self.operands],
            "results": [_encode(value) for value in self.results],
            "attributes": _encode(self.attributes),
            "regions": [region.to_data() for region in self.regions],
        }


@dataclass(frozen=True)
class Block:
    """A single-entry ordered sequence with explicit block arguments."""

    arguments: tuple[Value, ...] = ()
    operations: tuple[Operation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "operations", tuple(self.operations))

    def to_data(self) -> dict[str, Any]:
        return {
            "arguments": [_encode(value) for value in self.arguments],
            "operations": [operation.to_data() for operation in self.operations],
        }


@dataclass(frozen=True)
class Region:
    """A structured-control-flow region."""

    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if not self.blocks:
            raise ValueError("region requires at least one block")

    def to_data(self) -> dict[str, Any]:
        return {"blocks": [block.to_data() for block in self.blocks]}


@dataclass(frozen=True)
class HybridProgram:
    """Private single-function hybrid program container."""

    name: str
    body: Region
    location: SourceLocation | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("program name cannot be empty")
        object.__setattr__(self, "name", name)

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": "flagquantum.hybrid_program",
            "name": self.name,
            "body": self.body.to_data(),
        }

    @property
    def semantic_identity(self) -> str:
        payload = json.dumps(
            self.to_data(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
