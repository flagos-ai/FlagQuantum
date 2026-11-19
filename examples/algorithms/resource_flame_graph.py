"""Render a native declared-resource graph with the packaged FlameGraph asset."""

import argparse
from importlib.resources import as_file, files
from pathlib import Path

from flagquantum.compiler._component_trace.resource_generalization import (
    DeclaredResourceCalls,
    PhaseCall,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
)
from flagquantum.drawer.flame_graph import (
    iter_resource_flame_graph_data,
    render_flame_graph_svg,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination SVG path")
    arguments = parser.parse_args()

    leaf = ComponentCost("Select", 3)
    root = ComponentCost("Algorithm", 3, calls=(CostCall(leaf, 1),))
    calls: DeclaredResourceCalls = {leaf: {PhaseCall("t"): 4}}
    rows = iter_resource_flame_graph_data(root, declared_calls=calls)
    asset = files("flagquantum.drawer").joinpath("_assets/resource_flame_graph.pl")
    with as_file(asset) as executable:
        svg = render_flame_graph_svg(rows, executable=str(executable))
    arguments.output.write_bytes(svg)
    print(f"Wrote {arguments.output} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
