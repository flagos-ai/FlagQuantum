# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 call-graph drawing and Markdown formatting.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/bloq_counts_graph.py

"""Render declared resource graphs without constructing a circuit."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict
from graphlib import TopologicalSorter
from html import escape
from typing import Literal

from ..compiler._component_trace.call_graph import Node, modeled_call_graph
from ..compiler._component_trace.cost_query import (
    GATE_COST,
    PEAK_LOGICAL_QUBITS,
    query_component_cost,
)
from ..compiler._component_trace.gate_budget import (
    beverland_gate_budget,
    legacy_gate_budget,
    t_and_ccz_budget,
)
from ..compiler._component_trace.resource_model import ComponentCost, GateCost
from ..compiler._component_trace.symbolic_math import SymbolicInt

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


def export_call_graph_dot(
    root: Node,
    direct_calls: Mapping[tuple[Node, Node], SymbolicInt],
    *,
    title: Callable[[Node], str] = str,
    annotations: Mapping[Node, Mapping[str, str]] | None = None,
) -> str:
    """Export DOT with stable node identities, edge weights and optional tables."""
    order, successors = _call_order(root, direct_calls)
    ids = {node: f"n{index}" for index, node in enumerate(order)}
    lines = ["digraph counts {", "  rankdir=TB;"]
    for node in order:
        table = {} if annotations is None else annotations.get(node, {})
        rows = [f'<TR><TD COLSPAN="2">{escape(title(node))}</TD></TR>']
        rows.extend(
            f"<TR><TD>{escape(key)}</TD><TD>{escape(value)}</TD></TR>"
            for key, value in table.items()
        )
        label = '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        label += "".join(rows) + "</TABLE>"
        lines.append(f"  {ids[node]} [shape=plaintext, label=<{label}>];")
    for caller in order:
        for callee in successors.get(caller, ()):
            weight = _dot_string(_dot_count(direct_calls[caller, callee]))
            lines.append(f"  {ids[caller]} -> {ids[callee]} [label={weight}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _gate_annotations(gates: GateCost, aggregation: GateAggregation) -> dict[str, str]:
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
        counts = {"t": gates.total_t_count()}
    elif aggregation == "t_and_ccz":
        counts = t_and_ccz_budget(gates)
    elif aggregation == "beverland":
        counts = beverland_gate_budget(gates)
    elif aggregation == "legacy":
        counts = legacy_gate_budget(gates)
    else:
        raise ValueError(f"Unknown gate aggregation {aggregation!r}")
    return {labels.get(key, key): str(value) for key, value in counts.items()}


def export_modeled_call_graph_dot(
    component: ComponentCost,
    *,
    max_depth: int | None = None,
    aggregation: GateAggregation = "factored",
) -> str:
    """Annotate a depth-limited graph using costs from the full component model."""
    graph = modeled_call_graph(component, max_depth=max_depth)
    gates: dict[ComponentCost, GateCost] = {}
    widths: dict[ComponentCost, int] = {}
    query_component_cost(component, GATE_COST, cache=gates)
    query_component_cost(component, PEAK_LOGICAL_QUBITS, cache=widths)
    annotations = {
        node: {"Qubits": str(widths[node]), **_gate_annotations(cost, aggregation)}
        for node, cost in gates.items()
    }
    return export_call_graph_dot(
        graph.root,
        graph.direct_calls,
        title=lambda node: node.name,
        annotations=annotations,
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
