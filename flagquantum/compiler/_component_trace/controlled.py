# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled.build_call_graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Propagate default controls through existing declared resource graphs."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass

from .call_graph import CallDeclaration, SymbolAllocator, component_callee_counts
from .resource_generalization import ControlPattern
from .symbolic_math import SymbolicInt


@dataclass(frozen=True)
class ControlledCall:
    """A declared controlled child identified by its encoded activation bits.

    A symbolic integer stands for the number of unknown control bits. Physical
    register metadata and execution remain with the typed Program. Specialized
    control factories can replace default child identities during resolution.
    """

    child: Hashable
    controls: ControlPattern = (1,)

    def __str__(self) -> str:
        count = (
            len(self.controls) if isinstance(self.controls, tuple) else self.controls
        )
        prefix = "C" if count == 1 else f"C[{count}]"
        return f"{prefix}[{self.child}]"

    def callee_counts(
        self,
        declaration: CallDeclaration[Hashable],
        *,
        control: Callable[[Hashable, ControlPattern], Hashable] | None = None,
        allocator: SymbolAllocator | None = None,
    ) -> dict[Hashable, SymbolicInt]:
        """Control each declared child, merging equal specialized identities.

        The same allocator reaches the forward declaration. Original missing
        decomposition failures propagate. Counts remain analytical declarations
        and do not infer a physical decomposition or primitive cost.
        """

        def transform(child: Hashable) -> Hashable:
            return (
                ControlledCall(child, self.controls)
                if control is None
                else control(child, self.controls)
            )

        return component_callee_counts(
            self.child,
            declaration,
            generalizer=transform,
            allocator=allocator,
            ignore_unavailable=False,
        )
