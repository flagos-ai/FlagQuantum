"""Exercise the typed factory against unchanged source graph and cost owners."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args
from xml.etree import ElementTree

import pydot
import pytest
from sympy import Symbol, pretty

from examples.algorithms.resource_call_graphs import declared_example
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
    ControlledSwapCall,
    PhaseCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
    render_call_graph_svg,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    with (REFERENCE.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for path in (
        "drawing/bloq_counts_graph.py",
        "resource_counting/_qubit_counts.py",
        "bloqs/mcmt/and_bloq.py",
        "bloqs/swap_network/cswap_approx.py",
    ):
        raw = (REFERENCE / "qualtran" / path).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == blobs["qualtran/" + path]
        )


def parsed_dot(dot: str) -> tuple[dict[str, dict[str, str]], set[tuple[str, str, str]]]:
    graphs = pydot.graph_from_dot_data(dot)
    assert graphs is not None and len(graphs) == 1
    graph: Any = graphs[0]
    labels = {}
    rows = {}
    for node in graph.get_nodes():
        label = node.get_label()
        if label is None:
            continue
        cells = [list(row) for row in ElementTree.fromstring(label[1:-1])]
        title = "".join(cells[0][0].itertext())
        labels[node.get_name()] = title
        rows[title] = {
            "".join(key.itertext()): "".join(value.itertext())
            for key, value in cells[1:]
        }
    edges = {
        (
            labels[edge.get_source()],
            labels[edge.get_destination()],
            edge.get_label().strip('"'),
        )
        for edge in graph.get_edges()
    }
    return rows, edges


def native_width(node: ResourceCall) -> SymbolicInt:
    if isinstance(node, ControlledSwapCall):
        return 2 * node.width + 1
    if isinstance(node, ConjunctionCall):
        assert isinstance(node.controls, tuple)
        return 2 * len(node.controls) - 1
    if isinstance(node, CliffordCall):
        return node.width
    assert isinstance(node, PhaseCall)
    return 1


def native_title(node: ResourceCall) -> str:
    if isinstance(node, ControlledSwapCall):
        return "CSwapApprox"
    if isinstance(node, ConjunctionCall):
        assert isinstance(node.controls, tuple)
        return "And" if len(node.controls) == 2 else f"MultiAnd(n={len(node.controls)})"
    if isinstance(node, CliffordCall):
        return "X" if node.opcode == "x" else f"ArbitraryClifford(n={node.width})"
    assert isinstance(node, PhaseCall)
    return "T"


@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize(
    "family", ["and-positive", "and-negative", "swap", "symbolic-swap"]
)
def test_factory_matches_source_full_rows_edges_and_svg(
    family: str, depth: int | None, aggregation: GateAggregation
) -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.drawing import GraphvizCallGraph

    root: ResourceCall
    source: Any
    if family.startswith("and"):
        controls = (1, 1, 1, 1) if family == "and-positive" else (0, 1, 0, 1)
        root = ConjunctionCall(controls)
        source = MultiAnd(cvs=controls)
    else:
        width = (
            Symbol("n", positive=True, integer=True) if family == "symbolic-swap" else 3
        )
        root = ControlledSwapCall(width, approximate=True)
        source = CSwapApprox(width)
    visited: list[ResourceCall] = []

    def width_for(node: ResourceCall) -> SymbolicInt | None:
        visited.append(node)
        # Source connection-width traversal visits different nodes than QEC recipes.
        if family == "and-negative" and isinstance(node, CliffordCall):
            return None
        if family == "swap" and isinstance(node, CliffordCall):
            return None
        return native_width(node)

    dot = export_resource_call_graph_dot(
        root,
        logical_width=width_for,
        max_depth=depth,
        aggregation=aggregation,
        title=native_title,
    )
    rows, edges = parsed_dot(dot)
    original = GraphvizCallGraph.from_bloq(
        source, max_depth=depth, agg_gate_counts=aggregation
    )
    assert rows == {str(node): original.bloq_data[node] for node in original.g}
    assert edges == {
        (str(a), str(b), str(pretty(data["n"])))
        for a, b, data in original.g.edges(data=True)
    }
    assert len(visited) == len(set(visited))
    assert len(visited) == len(source.call_graph()[0])
    ElementTree.fromstring(render_call_graph_svg(dot))


def test_declared_symbolic_dag_uses_connection_width_with_bystanders() -> None:
    root, declarations = declared_example()
    # Split/atoms/join preserve three live wires while each atomic call uses one.
    steps = [
        WidthStep("input"),
        WidthStep("component", 1),
        WidthStep("component", 1),
        WidthStep("component", 1),
        WidthStep("output"),
    ]
    connections = [WidthConnection(0, i, 1) for i in (1, 2, 3)]
    connections += [WidthConnection(i, 4, 1) for i in (1, 2, 3)]
    peak = estimate_connection_width(steps, connections)
    assert peak == 3
    widths: dict[ResourceCall, SymbolicInt] = {}
    for node in {
        root,
        *declarations,
        *(child for row in declarations.values() for child in row),
    }:
        assert isinstance(node, ComponentCost)
        widths[node] = node.qubits
    widths[root] = 3
    for node in declarations:
        assert isinstance(node, ComponentCost)
        if node.name == "Parallel":
            widths[node] = peak
    before = {node: dict(row) for node, row in declarations.items()}
    for depth in (0, 1, None):
        rows, _ = parsed_dot(
            export_resource_call_graph_dot(
                root,
                declared_calls=declarations,
                logical_width=widths.__getitem__,
                max_depth=depth,
                title=lambda node: (
                    node.name if isinstance(node, ComponentCost) else str(node)
                ),
            )
        )
        assert rows[root.name] == {"Qubits": "3", "Ts": "100*_n0 + 600"}
    assert declarations == before


@pytest.mark.parametrize("depth", [0, 1, None])
def test_hidden_width_failure_is_visible_and_retry_is_fresh(depth: int | None) -> None:
    root, declarations = declared_example()
    table: dict[ResourceCall, SymbolicInt] = {root: 3}
    with pytest.raises(KeyError):
        export_resource_call_graph_dot(
            root,
            declared_calls=declarations,
            logical_width=table.__getitem__,
            max_depth=depth,
        )
    for node in {
        root,
        *declarations,
        *(child for row in declarations.values() for child in row),
    }:
        assert isinstance(node, ComponentCost)
        table[node] = max(node.qubits, 3) if node == root else node.qubits
    dot = export_resource_call_graph_dot(
        root,
        declared_calls=declarations,
        logical_width=table.__getitem__,
        max_depth=depth,
    )
    assert "<TD>Ts</TD><TD>100*_n0 + 600</TD>" in dot


@pytest.mark.parametrize("depth", [-1, True, 1.5])
def test_invalid_depth_fails_before_width_provider(depth: Any) -> None:
    def width(node: ResourceCall) -> SymbolicInt:
        pytest.fail("Width provider must not run for invalid depth")

    with pytest.raises(ValueError, match="depth"):
        export_resource_call_graph_dot(
            PhaseCall("t"), logical_width=width, max_depth=depth
        )


def test_cycle_failure_and_recovery_do_not_mutate_declarations() -> None:
    root = ComponentCost("root", 1)
    declarations: dict[ResourceCall, Mapping[ResourceCall, SymbolicInt]] = {
        root: {root: 1}
    }
    with pytest.raises(ValueError, match="cycle"):
        export_resource_call_graph_dot(
            root, declared_calls=declarations, logical_width=lambda node: 1, max_depth=0
        )
    assert declarations == {root: {root: 1}}
    declarations[root] = {PhaseCall("t"): 2}
    rows, _ = parsed_dot(
        export_resource_call_graph_dot(
            root, declared_calls=declarations, logical_width=lambda node: 1
        )
    )
    assert rows[str(root)] == {"Qubits": "1", "Ts": "2"}
