"""Immutable operation schemas and fail-closed registry."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

from .types import QUBIT, IRType

SCHEMA_VERSION = "1.0"


class UnknownOperationError(LookupError):
    """Raised when an operation has no registered schema."""


class SchemaVersionError(ValueError):
    """Raised when a caller requests an incompatible schema version."""


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    kinds: tuple[str, ...]
    required: bool = True

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        kinds = tuple(str(kind).strip().lower() for kind in self.kinds)
        if not name or not kinds or any(not kind for kind in kinds):
            raise ValueError("attribute spec requires a name and at least one kind")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kinds", kinds)


@dataclass(frozen=True)
class OperationSchema:
    name: str
    operands: tuple[IRType, ...]
    results: tuple[IRType, ...]
    attributes: tuple[AttributeSpec, ...] = ()
    region_count: int = 0
    effects: tuple[str, ...] = ()
    variadic_qubit_arity: bool = False
    reversible_to_circuit_ir_v1: bool = False
    version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name or "." not in name:
            raise ValueError("schema name must be a qualified operation name")
        if self.version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unsupported schema version {self.version!r}; expected {SCHEMA_VERSION!r}"
            )
        if int(self.region_count) < 0:
            raise ValueError("schema region count cannot be negative")
        attributes = tuple(self.attributes)
        names = tuple(item.name for item in attributes)
        if len(names) != len(set(names)):
            raise ValueError("schema attribute names must be unique")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "region_count", int(self.region_count))
        object.__setattr__(
            self, "effects", tuple(sorted(str(effect) for effect in self.effects))
        )


class OperationSchemaRegistry(Mapping[str, OperationSchema]):
    """An immutable, versioned collection of operation schemas."""

    __slots__ = ("_schemas", "version")

    def __init__(
        self,
        schemas: tuple[OperationSchema, ...],
        *,
        version: str = SCHEMA_VERSION,
    ) -> None:
        if version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unsupported registry version {version!r}; expected {SCHEMA_VERSION!r}"
            )
        ordered = tuple(sorted(schemas, key=lambda schema: schema.name))
        names = tuple(schema.name for schema in ordered)
        if len(names) != len(set(names)):
            raise ValueError("operation schema names must be unique")
        self._schemas = ordered
        self.version = version

    def __getitem__(self, name: str) -> OperationSchema:
        normalized = str(name).strip().lower()
        for schema in self._schemas:
            if schema.name == normalized:
                return schema
        raise UnknownOperationError(f"unknown internal operation {normalized!r}")

    def __iter__(self) -> Iterator[str]:
        return (schema.name for schema in self._schemas)

    def __len__(self) -> int:
        return len(self._schemas)


def circuit_ir_v1_schema_registry() -> OperationSchemaRegistry:
    """Build the immutable registry for the approved 35-opcode static profile."""

    schemas = []
    for source in OPERATOR_SCHEMAS.values():
        parameter_attributes = tuple(
            AttributeSpec(name, ("static", "symbolic", "binding"))
            for name in source.parameters
        )
        schemas.append(
            OperationSchema(
                name=f"quantum.{source.opcode}",
                operands=(QUBIT,) * source.arity,
                results=(QUBIT,) * source.arity,
                attributes=parameter_attributes,
                effects=("consume_linear", "produce_linear", source.semantic_kind),
                reversible_to_circuit_ir_v1=True,
            )
        )
    schemas.append(
        OperationSchema(
            name="quantum.custom_unitary",
            operands=(),
            results=(),
            attributes=(
                AttributeSpec("symbolic_name", ("string",)),
                AttributeSpec("matrix", ("static",)),
                AttributeSpec("arity", ("int",)),
            ),
            effects=("consume_linear", "produce_linear", "unitary"),
            variadic_qubit_arity=True,
            reversible_to_circuit_ir_v1=True,
        )
    )
    return OperationSchemaRegistry(tuple(schemas))


__all__ = [
    "AttributeSpec",
    "OperationSchema",
    "OperationSchemaRegistry",
    "SCHEMA_VERSION",
    "SchemaVersionError",
    "UnknownOperationError",
    "circuit_ir_v1_schema_registry",
]
