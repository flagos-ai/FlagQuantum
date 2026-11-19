# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Default declaration adjoints adapted from Qualtran v0.7.0 _infra/adjoint.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/adjoint.py

"""Retain default adjoint identities in the existing declared resource graph."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass

from .call_graph import CallDeclaration, SymbolAllocator, component_callee_counts
from .symbolic_math import SymbolicInt


@dataclass(frozen=True)
class AdjointCall:
    """A lazy inverse declaration with the original node as its identity.

    This is a resource node, independent of executable Program construction.
    Specialized inverses can be supplied when resolving its children. Physical
    and measured inverses still require the existing exact Program definitions.
    """

    child: Hashable

    def __str__(self) -> str:
        return f"{self.child}†"

    def adjoint(self) -> Hashable:
        return self.child

    def callee_counts(
        self,
        declaration: CallDeclaration[Hashable],
        *,
        inverse: Callable[[Hashable], Hashable] | None = None,
        allocator: SymbolAllocator | None = None,
    ) -> dict[Hashable, SymbolicInt]:
        """Resolve the forward declaration and combine its inverse children.

        Mapping and iterable declarations use the shared count accumulator.
        Missing declarations and their exceptions propagate. The same allocator
        reaches the forward declaration, preserving symbolic variable identity.
        """
        return component_callee_counts(
            self.child,
            declaration,
            generalizer=adjoint_call if inverse is None else inverse,
            allocator=allocator,
            ignore_unavailable=False,
        )


def adjoint_call(node: Hashable) -> Hashable:
    """Take the default declaration inverse, unwrapping an existing inverse."""
    return node.child if isinstance(node, AdjointCall) else AdjointCall(node)
