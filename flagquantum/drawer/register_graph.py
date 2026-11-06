# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GraphDrawer port and connection conventions.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/graphviz.py

"""Export instance-level register dependencies with explicit Graphviz ports."""

from dataclasses import dataclass
from html import escape
from itertools import zip_longest

from .register_score import RegisterScore, RegisterStep


@dataclass(frozen=True)
class RegisterConnection:
    """One register edge between operation ordinals or a graph boundary."""

    register: str
    width: int
    producer: int | None
    consumer: int | None


def register_connections(score: RegisterScore) -> tuple[RegisterConnection, ...]:
    """Follow each register through successive operations in display order."""
    widths = {line.register: line.width for line in score.lines}
    producers: dict[str, int | None] = {
        line.register: None for line in score.lines if line.start == -1
    }
    connections = []
    for column in score.columns:
        for register in column.step.inputs:
            connections.append(
                RegisterConnection(
                    register, widths[register], producers.pop(register), column.ordinal
                )
            )
        producers.update(dict.fromkeys(column.step.outputs, column.ordinal))
    connections.extend(
        RegisterConnection(line.register, line.width, producers[line.register], None)
        for line in score.lines
        if line.end == len(score.columns)
    )
    return tuple(connections)


def _instance_table(step: RegisterStep, ports: dict[str, str]) -> str:
    through = tuple(register for register in step.inputs if register in step.outputs)
    left = (register for register in step.inputs if register not in step.outputs)
    right = (register for register in step.outputs if register not in step.inputs)
    rows = [f'<TR><TD COLSPAN="2">{escape(step.name)}</TD></TR>']
    rows.extend(
        f'<TR><TD COLSPAN="2" PORT="{ports[register]}">{escape(register)}</TD></TR>'
        for register in through
    )
    for input_name, output_name in zip_longest(left, right):
        cells = [
            (
                "<TD></TD>"
                if register is None
                else f'<TD PORT="{ports[register]}">{escape(register)}</TD>'
            )
            for register in (input_name, output_name)
        ]
        rows.append("<TR>" + "".join(cells) + "</TR>")
    return (
        '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">' + "".join(rows) + "</TABLE>"
    )


def export_register_graph_dot(score: RegisterScore) -> str:
    """Export a left-to-right graph with one node for each operation instance."""
    ports = {line.register: f"p{i}" for i, line in enumerate(score.lines)}
    lines = ["digraph registers {", "  rankdir=LR;"]
    for prefix, boundary in (("in", -1), ("out", len(score.columns))):
        lines.append("  { rank=same;")
        for line in score.lines:
            position = line.start if prefix == "in" else line.end
            if position == boundary:
                label = escape(line.register)
                lines.append(
                    f"    {prefix}_{ports[line.register]} [shape=plain, label=<{label}>];"
                )
        lines.append("  }")
    for column in score.columns:
        table = _instance_table(column.step, ports)
        lines.append(f"  n{column.ordinal} [shape=plain, label=<{table}>];")
    for edge in register_connections(score):
        port = ports[edge.register]
        source = (
            f"in_{port}:e" if edge.producer is None else f"n{edge.producer}:{port}:e"
        )
        target = (
            f"out_{port}:w" if edge.consumer is None else f"n{edge.consumer}:{port}:w"
        )
        lines.append(f'  {source} -> {target} [label="{edge.width}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"
