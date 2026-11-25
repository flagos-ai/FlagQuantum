# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/trotter/grid_ham/qvr.py.

"""Retain the original opaque word and freshly allocated Rz model child."""

from dataclasses import dataclass
from typing import cast

from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt, smax
from ..._component_trace.call_graph import SymbolAllocator
from ..._component_trace.resource_generalization import RotationCall
from ..resources.symbolic_resource_model import SymbolicResourceFormula


@dataclass(frozen=True)
class GridVariableRotationDeclaration:
    """Preserve source metadata without assigning its symbolic angle a value."""

    phi_bitsize: SymbolicInt

    @property
    def signature(self) -> Signature:
        return Signature((Register("phi", opaque_register_type(self.phi_bitsize)),))

    def direct_calls(
        self, allocator: SymbolAllocator | None = None
    ) -> dict[RotationCall, SymbolicInt]:
        """Allocate one theta per query, retaining the source Rz precision."""
        symbols = SymbolAllocator() if allocator is None else allocator
        return {RotationCall("rz", symbols.new_symbol("theta")): self.phi_bitsize}

    def resource_formula(self) -> SymbolicResourceFormula:
        """Count source rotations and its bodyless maximum-child width fallback."""
        return SymbolicResourceFormula(
            0,
            0,
            0,
            cast(SymbolicInt, smax(1, self.phi_bitsize)),
            rotation=self.phi_bitsize,
        )
