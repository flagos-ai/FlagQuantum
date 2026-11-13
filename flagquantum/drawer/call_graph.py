# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 call-graph drawing and Markdown formatting.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/bloq_counts_graph.py

"""Render declared resource graphs without constructing a circuit."""

from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from collections.abc import Callable, Mapping
from dataclasses import asdict
from graphlib import TopologicalSorter
from html import escape
from typing import Literal, TypeVar

from ..compiler._component_trace.call_graph import (
    Node,
    _build_call_graph,
    modeled_call_graph,
)
from ..compiler._component_trace.cost_query import (
    GATE_COST,
    PEAK_LOGICAL_QUBITS,
    query_component_costs,
)
from ..compiler._component_trace.gate_budget import (
    beverland_gate_budget,
    legacy_gate_budget,
    t_and_ccz_budget,
)
from ..compiler._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from ..compiler._component_trace.resource_generalization import (
    DeclaredResourceCalls,
    ResourceCall,
)
from ..compiler._component_trace.resource_model import ComponentCost, GateCost
from ..compiler._component_trace.resource_width import query_resource_logical_width
from ..compiler._component_trace.symbolic_math import SymbolicInt
from ..compiler._component_trace.terminal_t_cost import _terminal_resource_callees

Metric = TypeVar("Metric")

GateAggregation = Literal["factored", "total_t", "t_and_ccz", "beverland", "legacy"]


def _call_order(
    root: Node, direct_calls: Mapping[tuple[Node, Node], SymbolicInt]
) -> tuple[list[Node], dict[Node, list[Node]]]:
    sorter: TopologicalSorter[Node] = TopologicalSorter()
    sorter.add(root)
    successors: dict[Node, list[Node]] = {}
    for caller, callee in direct_calls:
        sorter.add(callee, caller)
        successors.setdefault(caller, []).append(callee)
    return list(sorter.static_order()), successors


def _code_span(label: str) -> str:
    delimiter = "`" * (1 + max(map(len, re.findall(r"`+", label)), default=0))
    if label.startswith(("`", " ")) or label.endswith(("`", " ")):
        label = f" {label} "
    return f"{delimiter}{label}{delimiter}"


def _markdown_count(count: SymbolicInt, *, edge: bool = False) -> str:
    if isinstance(count, int):
        return rf"$\displaystyle {count}$" if edge else str(count)
    from sympy import latex

    return rf"$\displaystyle {latex(count)}$"


def format_call_graph_markdown(
    root: Node,
    direct_calls: Mapping[tuple[Node, Node], SymbolicInt],
    leaf_counts: Mapping[Node, SymbolicInt],
    *,
    title: Callable[[Node], str] = str,
) -> str:
    """Format direct calls in topological order and append weighted leaf totals."""
    order, successors = _call_order(root, direct_calls)
    lines = []
    for caller in order:
        if caller not in successors:
            continue
        lines.append(f" - {_code_span(title(caller))}")
        lines.extend(
            sorted(
                f"   - {_code_span(title(callee))}: "
                f"{_markdown_count(direct_calls[caller, callee], edge=True)}"
                for callee in successors[caller]
            )
        )
    lines.append("#### Counts totals:")
    lines.extend(
        sorted(
            f" - {_code_span(title(node))}: {_markdown_count(count)}"
            for node, count in leaf_counts.items()
        )
    )
    return "\n".join(lines)


def _dot_count(count: SymbolicInt) -> str:
    if isinstance(count, int):
        return str(count)
    from sympy import pretty

    return str(pretty(count))


def _dot_string(label: str) -> str:
    escaped = label.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _dot_attributes(attributes: Mapping[str, str]) -> str:
    fields = []
    for key, value in attributes.items():
        rendered = (
            value
            if value.startswith("<") and value.endswith(">")
            else _dot_string(value)
        )
        fields.append(f"{_dot_string(key)}={rendered}")
    return ", ".join(fields)


def export_call_graph_dot(
    root: Node,
    direct_calls: Mapping[tuple[Node, Node], SymbolicInt],
    *,
    title: Callable[[Node], str] = str,
    annotations: Mapping[Node, Mapping[str, str]] | None = None,
    details: Callable[[Node], str] | None = None,
    node_properties: Callable[[Node], Mapping[str, str]] | None = None,
) -> str:
    """Export DOT with stable node identities, edge weights and optional tables.

    Details select the source rectangular title/detail label. A complete
    node_properties callback replaces default properties and bypasses title,
    details and annotation formatting. Callbacks run once per visible node
    in topological order, and failures propagate. Returned property mappings
    remain caller-owned. Property values use strings, with angle-delimited
    HTML labels passed through as trusted Graphviz markup.

    Examples:
        >>> dot = export_call_graph_dot("root", {}, details=lambda node: "T count = 4")
        >>> '"shape"="rect"' in dot and "T count = 4" in dot
        True
    """
    order, successors = _call_order(root, direct_calls)
    ids = {node: f"n{index}" for index, node in enumerate(order)}
    lines = ["digraph counts {", "  rankdir=TB;"]
    for node in order:
        if node_properties is not None:
            properties = _dot_attributes(node_properties(node))
        elif details is not None:
            label = "<" + escape(title(node))
            detail = escape(details(node))
            if detail:
                label += (
                    f'<br /><font face="monospace" point-size="10">{detail}</font><br/>'
                )
            properties = _dot_attributes({"label": label + ">", "shape": "rect"})
        else:
            table = {} if annotations is None else annotations.get(node, {})
            rows = [f'<TR><TD COLSPAN="2">{escape(title(node))}</TD></TR>']
            rows.extend(
                f"<TR><TD>{escape(key)}</TD><TD>{escape(value)}</TD></TR>"
                for key, value in table.items()
            )
            label = '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
            label += "".join(rows) + "</TABLE>"
            properties = f"shape=plaintext, label=<{label}>"
        lines.append(f"  {ids[node]} [{properties}];")
    for caller in order:
        for callee in successors.get(caller, ()):
            weight = _dot_string(_dot_count(direct_calls[caller, callee]))
            lines.append(f"  {ids[caller]} -> {ids[callee]} [label={weight}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _gate_annotations(
    gates: GateCost | QECGateCost, aggregation: GateAggregation
) -> dict[str, str]:
    labels = {
        "t": "Ts",
        "n_t": "Ts",
        "toffoli": "Toffolis",
        "n_ccz": "CCZs",
        "cswap": "CSwaps",
        "and_bloq": "Ands",
        "clifford": "Cliffords",
        "rotation": "Rotations",
        "rotations": "Rotations",
        "measurement": "Measurements",
    }
    counts: Mapping[str, SymbolicInt]
    if aggregation == "factored":
        counts = {key: value for key, value in asdict(gates).items() if value}
    elif aggregation == "total_t":
        budget = t_and_ccz_budget(gates)
        counts = {"t": budget["n_t"] + 4 * budget["n_ccz"]}
    elif aggregation == "t_and_ccz":
        counts = t_and_ccz_budget(gates)
    elif aggregation == "beverland":
        counts = beverland_gate_budget(gates)
    elif aggregation == "legacy":
        counts = legacy_gate_budget(gates)
    else:
        raise ValueError(f"Unknown gate aggregation {aggregation!r}")
    return {labels.get(key, key): str(value) for key, value in counts.items()}


def format_call_graph_cost_data(
    cost_data: Mapping[Node, Mapping[Metric, object]],
    *,
    aggregation: GateAggregation = "factored",
) -> dict[Node, dict[str, str]]:
    """Format existing Compiler metric rows for native graph annotations.

    GATE_COST accepts concrete GateCost or symbolic QECGateCost values.
    PEAK_LOGICAL_QUBITS values use their declared string representation.
    Unknown metrics emit UserWarning and retain their string key and value.
    Metric and annotation insertion order is preserved. Invalid gate values
    raise TypeError. The input table remains caller-owned.

    Examples:
        >>> root = ComponentCost("Root", 3, GateCost(t=2))
        >>> costs = query_component_costs(root, (PEAK_LOGICAL_QUBITS, GATE_COST))
        >>> format_call_graph_cost_data(costs)[root]
        {'Qubits': '3', 'Ts': '2'}
    """
    result = {}
    for node, metrics in cost_data.items():
        record = {}
        for metric, value in metrics.items():
            if metric is PEAK_LOGICAL_QUBITS:
                record["Qubits"] = str(value)
            elif metric is GATE_COST:
                if not isinstance(value, (GateCost, QECGateCost)):
                    raise TypeError(
                        "Gate cost annotations require GateCost or QECGateCost"
                    )
                record.update(_gate_annotations(value, aggregation))
            else:
                warnings.warn(
                    f"Unknown cost metric {metric}", UserWarning, stacklevel=2
                )
                record[str(metric)] = str(value)
        result[node] = record
    return result


def export_modeled_call_graph_dot(
    component: ComponentCost,
    *,
    max_depth: int | None = None,
    aggregation: GateAggregation = "factored",
    details: Callable[[ComponentCost], str] | None = None,
    node_properties: Callable[[ComponentCost], Mapping[str, str]] | None = None,
) -> str:
    """Annotate a depth-limited graph using costs from the full component model."""
    graph = modeled_call_graph(component, max_depth=max_depth)
    costs = query_component_costs(component, (PEAK_LOGICAL_QUBITS, GATE_COST))
    annotations = format_call_graph_cost_data(costs, aggregation=aggregation)
    return export_call_graph_dot(
        graph.root,
        graph.direct_calls,
        title=lambda node: node.name,
        annotations=annotations,
        details=details,
        node_properties=node_properties,
    )


def export_resource_call_graph_dot(
    root: ResourceCall,
    *,
    logical_width: Callable[[ResourceCall], SymbolicInt | None] | None = None,
    declared_calls: DeclaredResourceCalls | None = None,
    max_depth: int | None = None,
    aggregation: GateAggregation = "factored",
    title: Callable[[ResourceCall], str] = str,
    details: Callable[[ResourceCall], str] | None = None,
    node_properties: Callable[[ResourceCall], Mapping[str, str]] | None = None,
) -> str:
    """Draw typed resource calls with full costs and owned logical widths.

    Shared Compiler recipes supply automatic children and QEC costs. Explicit
    declaration rows replace automatic children. The width provider owns peak
    estimation and receives every node of the full graph exactly once, even
    when display depth hides it. Use an existing connection-width calculator
    or a previously evaluated width table. Return None for an unvisited width
    metric so sparse query rows retain their original meaning. Provider failures
    propagate.
    Without a provider, Compiler queries supported native width bindings once
    from the root and preserves sparse visitation. Unsupported bindings raise
    NotImplementedError. Input declarations and width tables remain caller-owned.

    Examples:
        >>> from ..compiler._component_trace.resource_generalization import PhaseCall
        >>> root = PhaseCall("t")
        >>> dot = export_resource_call_graph_dot(root, logical_width={root: 1}.__getitem__)
        >>> '<TD>Qubits</TD><TD>1</TD>' in dot and '<TD>Ts</TD><TD>1</TD>' in dot
        True
    """
    graph = _build_call_graph(
        root,
        lambda node: _terminal_resource_callees(node, declared_calls),
        None,
        None,
        max_depth,
    )
    complete = (
        graph
        if max_depth is None
        else _build_call_graph(
            root,
            lambda node: _terminal_resource_callees(node, declared_calls),
            None,
            None,
            None,
        )
    )
    nodes, _ = _call_order(complete.root, complete.direct_calls)
    costs: dict[ResourceCall, QECGateCost] = {}
    query_qec_gate_cost(root, declared_calls=declared_calls, cache=costs)
    if logical_width is None:
        widths: dict[ResourceCall, SymbolicInt] = {}
        query_resource_logical_width(root, declared_calls=declared_calls, cache=widths)
        logical_width = widths.get
    table: dict[ResourceCall, dict[object, object]] = {}
    for node in nodes:
        width = logical_width(node)
        table[node] = {} if width is None else {PEAK_LOGICAL_QUBITS: width}
    for node, gates in costs.items():
        table[node][GATE_COST] = gates
    return export_call_graph_dot(
        graph.root,
        graph.direct_calls,
        title=title,
        annotations=format_call_graph_cost_data(table, aggregation=aggregation),
        details=details,
        node_properties=node_properties,
    )


def render_call_graph_svg(
    dot_source: str, *, executable: str = "dot", timeout: float = 30.0
) -> bytes:
    """Render exported DOT to SVG using a local Graphviz executable.

    Pass source from the call-graph exporters or other trusted DOT. The
    executable can be a name on PATH or an explicit path.
    """
    dot = shutil.which(executable)
    if dot is None:
        raise ImportError(
            f"Call-graph SVG rendering requires Graphviz ({executable!r}). "
            "Install Graphviz or use the DOT source directly."
        )
    try:
        return subprocess.run(
            [dot, "-Tsvg"],
            input=dot_source.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=timeout,
        ).stdout
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Graphviz rendering timed out after {timeout:g}s") from exc
    except subprocess.CalledProcessError as exc:
        diagnostic = exc.stderr[-4000:].decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Graphviz SVG rendering failed: {diagnostic}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot run Graphviz executable {dot}: {exc}") from exc
