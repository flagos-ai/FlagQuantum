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
from flagquantum.compiler.registers.fixed_point.trigonometric_resource_model import (
    ARCTANGENT_PRODUCT_CALLS,
    source_arcsine_toffoli_count,
)
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

TrigExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]


def arcsine_example() -> TrigExample:
    """Return the original width-ten, seven-fractional-bit ArcSin factory."""
    word = fixed_point_register_type(10, 7)
    return (
        "arcsine",
        Signature((Register("x", word), Register("result", word))),
        {"toffoli": source_arcsine_toffoli_count(10, 7)},
    )


def arctangent_example(source_width: SymbolicInt, width: SymbolicInt) -> TrigExample:
    """Retain both occupied target ports and the original product declaration."""
    return (
        "arctangent",
        Signature.build(select=source_width, sign=1, target=width),
        {("plus_equal_product", width, width, 2 * width): ARCTANGENT_PRODUCT_CALLS},
    )


def trigonometric_example_graph(example: TrigExample) -> SymbolicCallGraph[Hashable]:
    """Render the source one-level estimate without choosing an executable path."""
    name, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
