# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 approximate QFT notebook.

"""Native notebook instances and direct call-graph labels."""

from flagquantum.algorithms.primitives.approximate_qft import approximate_qft_precision
from flagquantum.algorithms.primitives.approximate_qft_resource_model import (
    source_approximate_qft_calls,
)
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)


def approximate_qft_examples() -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the explicit and epsilon-derived locked examples."""
    return (6, 5), (50, approximate_qft_precision(50, 1e-5))


def approximate_qft_graph(width: int, precision: int) -> SymbolicCallGraph[str]:
    """Give the concrete source's shifted add windows distinct graph nodes."""
    root = f"Approximate QFT ({width}, {precision})"
    children = {
        f"{name}({sizes[0]}, {sizes[1]}; shift 1)" if sizes else name: count
        for (name, sizes), count in source_approximate_qft_calls(
            width, precision
        ).items()
    }
    return symbolic_call_graph(root, {root: children}, max_depth=1)
