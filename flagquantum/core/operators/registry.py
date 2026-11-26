"""Immutable operator tables; frontend method installation belongs to Compiler."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import get_type_hints as _class_type_hints

from .schema import (
    OPERATOR_ALIASES,
    OPERATOR_SCHEMAS,
    OperatorSchema,
)


@dataclass(frozen=True)
class OperatorRegistry:
    """An immutable operator table: the built-ins plus whatever was registered.

    Attributes:
        schemas: Every opcode the table knows, keyed by opcode. Read-only.
    """

    schemas: Mapping[str, OperatorSchema] = field(
        default_factory=lambda: OPERATOR_SCHEMAS
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "schemas", MappingProxyType(dict(self.schemas)))

    @property
    def aliases(self) -> Mapping[str, str]:
        """Alias to opcode, over the built-in table and this one together."""

        merged = dict(OPERATOR_ALIASES)
        merged.update(
            {
                alias: schema.opcode
                for schema in self.schemas.values()
                for alias in schema.aliases
            }
        )
        return MappingProxyType(merged)

    def canonical(self, name: str) -> str:
        """Resolve ``name`` through this table's aliases, or return it unchanged.

        This is the registry's own resolution by string. ``Circuit`` resolves a
        gate name through the built-in table, but a registration installs a method
        for the opcode and for each alias, and those methods emit the canonical
        opcode, so the declared spelling reaches a program even though the string
        lookup here is what knows the alias.
        """

        normalized = str(name).strip().lower()
        return self.aliases.get(normalized, normalized)

    def with_operator(self, schema: OperatorSchema) -> "OperatorRegistry":
        """Return a new registry that also holds ``schema``.

        Raises:
            ValueError: If the opcode or one of its aliases is already taken by
                this table.
        """

        opcode = str(schema.opcode).strip().lower()
        if opcode in self.schemas or opcode in self.aliases:
            raise ValueError(f"operator schema {opcode!r} is already registered")
        for alias in schema.aliases:
            if alias in self.schemas or alias in self.aliases or alias == opcode:
                raise ValueError(
                    f"operator schema {opcode!r} declares alias {alias!r}, which "
                    "this table already resolves"
                )
        updated = dict(self.schemas)
        updated[opcode] = schema
        return replace(self, schemas=updated)

    def registered(self) -> tuple[str, ...]:
        """The opcodes this table holds that the built-in table does not."""

        return tuple(sorted(set(self.schemas) - set(OPERATOR_SCHEMAS)))


__all__ = ("OperatorRegistry",)


# Preserve historical import identities for pickles and public introspection.
# Resolve field types before preserving historical public class module names.
# Dataclass constructor/field contracts stay unchanged; introspection needs no facade import.
for _public_hint_class in (OperatorRegistry,):
    _public_field_hints = _class_type_hints(_public_hint_class, include_extras=True)
    _public_hint_class.__annotations__ = {
        name: _public_field_hints[name] for name in _public_hint_class.__annotations__
    }
del _public_hint_class, _public_field_hints

for _identity_export in (OperatorRegistry,):
    _identity_export.__module__ = "flagquantum.operators.registry"
del _identity_export
