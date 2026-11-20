# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Resource delegation adapted from Qualtran v0.7.0 Always.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/always.py

"""Delegate an unconditional wrapper's resource graph to its native child."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass

from .adjoint import adjoint_call
from .call_graph import CallDeclaration, SymbolAllocator, component_callee_counts
from .symbolic_math import SymbolicInt


@dataclass(frozen=True)
class AlwaysCall:
    """Retain a generic child identity independently of physical gate emission.

    The child declaration supplies its own direct graph. Atomic declaration
    errors propagate, even when a separate child cost metric knows its cost.
    Existing typed Program and emitter wrappers own execution and controls.
    """

    child: Hashable

    def __str__(self) -> str:
        return str(self.child)

    def adjoint(
        self, inverse: Callable[[Hashable], Hashable] = adjoint_call
    ) -> AlwaysCall:
        """Wrap the child's existing inverse, retaining specialized identities."""
        return AlwaysCall(inverse(self.child))

    def callee_counts(
        self,
        declaration: CallDeclaration[Hashable],
        *,
        allocator: SymbolAllocator | None = None,
    ) -> dict[Hashable, SymbolicInt]:
        """Use the child's direct declaration and the caller's symbol allocator.

        The shared accumulator handles mappings, iterables and repeated keys.
        No wrapper-to-child edge is added to the declared resource graph.
        """
        return component_callee_counts(
            self.child,
            declaration,
            allocator=allocator,
            ignore_unavailable=False,
        )
