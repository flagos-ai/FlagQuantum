# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/trigonometric/arctan.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/trigonometric/arctan.py

"""Own arctangent word layout and the original fresh-symbol product estimate."""

from dataclasses import dataclass

from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ....core.value_parameters import is_symbolic_value
from ..._component_trace.call_graph import SymbolAllocator


@dataclass(frozen=True)
class ArctangentDeclaration:
    """Retain select/sign/occupied-target ports and a self-inverse XOR image."""

    selection_bitsize: SymbolicInt
    target_bitsize: SymbolicInt

    @property
    def signature(self) -> Signature:
        return Signature.build(
            select=self.selection_bitsize, sign=1, target=self.target_bitsize
        )

    def register_dimensions(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return the original binary digit layouts, refusing symbolic dimensions."""
        if is_symbolic_value(self.selection_bitsize, self.target_bitsize):
            raise TypeError("Cannot build arctangent registers for symbolic widths")
        assert isinstance(self.selection_bitsize, int) and isinstance(
            self.target_bitsize, int
        )
        return (2,) * self.selection_bitsize, (2,), (2,) * self.target_bitsize

    def direct_calls(
        self, allocator: SymbolAllocator | None = None
    ) -> dict[tuple[str, SymbolicInt, SymbolicInt, SymbolicInt], SymbolicInt]:
        """Allocate the original c multiplicity independently for each graph query."""
        symbols = SymbolAllocator() if allocator is None else allocator
        width = self.target_bitsize
        return {
            ("plus_equal_product", width, width, 2 * width): symbols.new_symbol("c")
        }

    def adjoint(self) -> "ArctangentDeclaration":
        return self
