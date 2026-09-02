"""Semantics-preserving canonicalization seed pass for Phase 1."""

from __future__ import annotations

from ..analyses.base import walk_operations
from ..ir.modules import QuantumModule
from .base import PassDescriptor, PassResult


class CanonicalizeAttributesPass:
    """Confirm the already-canonical immutable attribute representation."""

    descriptor = PassDescriptor("canonicalize_attributes", "1.0")

    def run(self, module: QuantumModule) -> PassResult:
        operation_count = sum(1 for _ in walk_operations(module))
        return PassResult(
            module,
            changed=False,
            preserved_analyses=frozenset({"def_use", "qubit_lifetime"}),
            statistics={"operations_scanned": operation_count},
        )


__all__ = ["CanonicalizeAttributesPass"]
