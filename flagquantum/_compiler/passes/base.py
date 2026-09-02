"""Immutable contracts for private Phase 1 compiler passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..diagnostics import Diagnostic
from ..ir.modules import QuantumModule
from ..ir.operations import FrozenAttributes, canonical_value


@dataclass(frozen=True)
class PassDescriptor:
    name: str
    version: str
    options: FrozenAttributes = field(default_factory=FrozenAttributes)
    seed: int | None = None
    preserves_semantics: bool = True

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        version = str(self.version).strip()
        if not name or not version:
            raise ValueError("pass name and version cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "options", FrozenAttributes(self.options))
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))

    def canonical(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "options": canonical_value(self.options),
            "seed": self.seed,
            "preserves_semantics": self.preserves_semantics,
        }


@dataclass(frozen=True)
class PassResult:
    module: QuantumModule
    changed: bool = False
    preserved_analyses: frozenset[str] = frozenset()
    diagnostics: tuple[Diagnostic, ...] = ()
    statistics: FrozenAttributes = field(default_factory=FrozenAttributes)

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed", bool(self.changed))
        object.__setattr__(
            self,
            "preserved_analyses",
            frozenset(str(item) for item in self.preserved_analyses),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "statistics", FrozenAttributes(self.statistics))


class CompilerPass(Protocol):
    descriptor: PassDescriptor

    def run(self, module: QuantumModule) -> PassResult: ...


__all__ = ["CompilerPass", "PassDescriptor", "PassResult"]
