# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/trigonometric.

"""Original trigonometric signatures and independent declared child costs."""

from collections.abc import Hashable

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.fixed_point.arcsine_declaration import (
    ArcsineDeclaration,
)
from flagquantum.compiler.registers.lookup.arctangent_declaration import (
    ArctangentDeclaration,
)
from flagquantum.core.ir.program.signatures import Signature

TrigExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]


def arcsine_example() -> TrigExample:
    """Return the original width-ten, seven-fractional-bit ArcSin factory."""
    declaration = ArcsineDeclaration(10, 7)
    calls: dict[Hashable, SymbolicInt] = {
        key: value for key, value in declaration.direct_calls().items()
    }
    return "arcsine", declaration.signature, calls


def arctangent_example(source_width: SymbolicInt, width: SymbolicInt) -> TrigExample:
    """Retain both occupied target ports and the original product declaration."""
    declaration = ArctangentDeclaration(source_width, width)
    calls: dict[Hashable, SymbolicInt] = {
        key: value for key, value in declaration.direct_calls().items()
    }
    return "arctangent", declaration.signature, calls


def trigonometric_example_graph(example: TrigExample) -> SymbolicCallGraph[Hashable]:
    """Render the source one-level estimate without choosing an executable path."""
    name, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
