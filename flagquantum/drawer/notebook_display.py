# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 notebook display helpers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/_show_funcs.py

"""Compose native renderers with optional IPython display."""

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from pathlib import Path

from ..compiler._component_trace.call_graph import Node, SymbolicCallGraph
from ..compiler._component_trace.resource_generalization import (
    DeclaredResourceCalls,
    ResourceCall,
)
from ..compiler._component_trace.symbolic_math import SymbolicInt
from ..core.ir.program import ProgramRecord
from .call_graph import (
    GateAggregation,
    export_call_graph_dot,
    export_resource_call_graph_dot,
    format_counts_markdown,
    render_call_graph_svg,
)
from .component_graph import export_component_graph_dot
from .component_score import component_register_score
from .flame_graph import (
    _keep_small_resource,
    iter_resource_flame_graph_data,
    render_flame_graph_svg,
)
from .latex_figure import LatexFigure
from .register_score_data import register_score_data


def show_register_program(
    program: ProgramRecord | LatexFigure, mode: str = "graph"
) -> None:
    """Display an owned register program, or an explicitly prepared LaTeX figure."""
    mode = mode.lower()
    if mode == "latex":
        if not isinstance(program, LatexFigure):
            raise TypeError("LaTeX display requires an explicit LatexFigure")
        show_latex_diagram(program)
        return
    if mode not in ("graph", "dtype", "musical_score"):
        raise ValueError("Display mode must be graph, dtype, musical_score or latex")
    if not isinstance(program, ProgramRecord):
        raise TypeError("Register graph and score display require a ProgramRecord")
    if mode == "musical_score":
        from .register_score_mpl import draw_register_score

        figure, _ = draw_register_score(
            register_score_data(component_register_score(program))
        )
        figure.show()
    else:
        display = import_module("IPython.display")
        dot = export_component_graph_dot(program, typed=mode == "dtype")
        display.display(display.SVG(render_call_graph_svg(dot)))


def show_register_programs(
    programs: Sequence[ProgramRecord], labels: Sequence[str | None] | None = None
) -> None:
    """Display width-labelled register graphs side by side with Markdown captions."""
    if labels is not None and len(labels) != len(programs):
        raise ValueError("Provide exactly one label per program")
    display = import_module("IPython.display")
    widgets = import_module("ipywidgets")
    outputs = []
    for index, program in enumerate(programs):
        output = widgets.Output()
        if labels is not None and labels[index]:
            output.append_display_data(display.Markdown(labels[index]))
        output.append_display_data(
            display.SVG(render_call_graph_svg(export_component_graph_dot(program)))
        )
        outputs.append(output)
    display.display(widgets.HBox(outputs))


def show_call_graph(
    item: ResourceCall | SymbolicCallGraph[Node],
    *,
    max_depth: int | None = None,
    aggregation: GateAggregation = "factored",
) -> None:
    """Display a prebuilt graph unchanged or query the full declared resource costs."""
    if isinstance(item, SymbolicCallGraph):
        dot = export_call_graph_dot(item.root, item.direct_calls)
    else:
        dot = export_resource_call_graph_dot(
            item, max_depth=max_depth, aggregation=aggregation
        )
    display = import_module("IPython.display")
    display.display(display.SVG(render_call_graph_svg(dot)))


def show_counts(
    counts: Mapping[Node, SymbolicInt], *, title: Callable[[Node], str] = str
) -> None:
    """Display symbolic or concrete leaf totals as Markdown."""
    display = import_module("IPython.display")
    display.display(display.Markdown(format_counts_markdown(counts, title=title)))


def show_flame_graph(
    *roots: ResourceCall,
    executable: str,
    declared_calls: DeclaredResourceCalls | None = None,
    keep: Callable[[ResourceCall], bool] | None = _keep_small_resource,
    max_depth: int | None = None,
    file_path: str | Path | None = None,
) -> None:
    """Display folded T costs, optionally saving the same rendered SVG."""
    rows = iter_resource_flame_graph_data(
        *roots, declared_calls=declared_calls, keep=keep, max_depth=max_depth
    )
    svg = render_flame_graph_svg(rows, executable=executable)
    if file_path is not None:
        Path(file_path).write_bytes(svg)
    display = import_module("IPython.display")
    display.display(display.SVG(svg))


def show_latex_diagram(
    figure: LatexFigure, *, width: int = 1000, height: int = 400
) -> None:
    """Display compiled PNG bytes with explicit notebook dimensions."""
    display = import_module("IPython.display")
    display.display(
        display.Image(
            data=figure.render("png"), format="png", width=width, height=height
        )
    )
