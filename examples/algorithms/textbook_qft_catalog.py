# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 textbook QFT notebook.

"""Textbook Fourier examples with source declarations and native execution."""

from sympy import Symbol

from flagquantum.algorithms.primitives.qft_resource_model import (
    source_textbook_qft_calls,
)
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt


def textbook_qft_examples() -> tuple[SymbolicInt, SymbolicInt]:
    """Return the original concrete factory and unrestricted symbolic example."""
    return 3, Symbol("n")


def textbook_qft_graph(
    width: SymbolicInt, *, with_reverse: bool = True
) -> SymbolicCallGraph[str]:
    """Render the source notebook's one-level graph."""
    root = f"QFTTextBook({width})"
    children = {
        f"{name}({bits})" if name == "PhaseGradientUnitary" else name: count
        for (name, bits), count in source_textbook_qft_calls(
            width, with_reverse=with_reverse
        ).items()
    }
    return symbolic_call_graph(root, {root: children}, max_depth=1)
