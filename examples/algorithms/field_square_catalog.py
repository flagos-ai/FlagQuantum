# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 gf2_square notebook.

"""Native field-square signatures and declared linear-map resource graphs."""

from __future__ import annotations

from collections.abc import Hashable

from sympy import Symbol

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.finite_field.field_linear import (
    source_gf2_square_resource_bound,
)
from flagquantum.compiler.registers.finite_field.linear_resource_model import (
    source_binary_linear_resource_formula,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_frobenius import field_frobenius_matrix

SquareExample = tuple[str, Signature, SymbolicResourceFormula]


def field_square_examples() -> tuple[SquareExample, ...]:
    """Retain the original concrete degree four and unrestricted symbolic m."""
    examples = []
    for width, suffix in [(4, "concrete"), (Symbol("m"), "symbolic")]:
        encoding = binary_field_encoding(width)
        signature = Signature((Register("x", encoding.register_type()),))
        if type(width) is int:
            matrix = field_frobenius_matrix(encoding)
            model = source_binary_linear_resource_formula(
                tuple(tuple(int(bit) for bit in row) for row in matrix)
            )
        else:
            model = source_gf2_square_resource_bound(width)
        examples.append((f"field_square_{suffix}", signature, model))
    return tuple(examples)


def field_square_example_graph(
    example: SquareExample, *, max_depth: int | None = 1
) -> SymbolicCallGraph[Hashable]:
    """Expose the source's single linear-map child and optional primitive leaves."""
    root, _, model = example
    child = "Binary linear map"
    declarations: dict[Hashable, dict[Hashable, SymbolicInt]] = {
        root: {child: 1},
        child: {CliffordCall("cx", 2): model.clifford},
    }
    return symbolic_call_graph(root, declarations, max_depth=max_depth)
