# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 drawing_call_graph notebook and test declarations.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/drawing_call_graph.ipynb

"""Draw a symbolic declared resource graph with native Compiler and Drawer owners.

Run python -m examples.algorithms.resource_call_graphs with Graphviz installed.
The atomic declaration costs 100T. The root costs 100*n + 600T. These are
synthetic model counts, and this example emits no quantum circuit.
"""

from sympy import Symbol

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    DeclaredResourceCalls,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.drawer.call_graph import export_call_graph_dot, render_call_graph_svg


def declared_example() -> tuple[ComponentCost, DeclaredResourceCalls]:
    """Declare the notebook DAG independently of the reference implementation."""
    root = ComponentCost("Declared program", 0)
    parallel = ComponentCost("Parallel", 3)
    serial = ComponentCost("Serial", 1)
    atom = ComponentCost("Atom", 1, GateCost(t=100))
    split = ComponentCost("Split", 3)
    join = ComponentCost("Join", 3)
    serial_atoms = tuple(
        ComponentCost(f"Atom {index}", 1, GateCost(t=100)) for index in range(3)
    )
    repetitions = Symbol("_n0")
    return root, {
        root: {parallel: 1, serial: 1, atom: repetitions},
        parallel: {split: 1, atom: 3, join: 1},
        serial: dict.fromkeys(serial_atoms, 1),
    }


def annotated_example() -> (
    tuple[SymbolicCallGraph[ResourceCall], dict[ResourceCall, dict[str, str]]]
):
    """Use the existing symbolic graph and recursive QEC cost-query cache."""
    root, declarations = declared_example()
    graph = symbolic_call_graph(root, declarations)
    costs: dict[ResourceCall, QECGateCost] = {}
    query_qec_gate_cost(root, declared_calls=declarations, cache=costs)
    annotations = {
        node: {
            "T count": str(cost.t),
            "clifford": str(cost.clifford),
            "rot": str(cost.rotation),
        }
        for node, cost in costs.items()
    }
    return graph, annotations


def main() -> None:
    graph, annotations = annotated_example()
    dot = export_call_graph_dot(
        graph.root,
        graph.direct_calls,
        title=lambda node: node.name if isinstance(node, ComponentCost) else str(node),
        annotations=annotations,
    )
    svg = render_call_graph_svg(dot)
    print("Declared root model", annotations[graph.root])
    print("Rendered SVG bytes", len(svg))


if __name__ == "__main__":
    main()
