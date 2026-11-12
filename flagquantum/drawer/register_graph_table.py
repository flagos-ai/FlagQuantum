# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GraphDrawer.add_binst and PrettyGraphDrawer.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/graphviz.py

"""Present explicit signature groups on the shared register graph geometry."""

from html import escape
from itertools import zip_longest

from ..core.ir.program.registers import RegisterSide
from ..core.ir.program.signatures import Signature
from .register_graph import RegisterConnection
from .register_score import RegisterStep


def signature_instance_table(
    signature: Signature,
    step: RegisterStep,
    ports: dict[str, str],
    *,
    pretty: bool = False,
    reshaping: bool = False,
) -> str:
    """Keep group order, unequal-side row spans and local port display labels.

    Connectivity comes from the validated score. Signature metadata supplies
    display spelling independently of lane lifetimes. Callers identify actual
    split/join operations explicitly instead of inferring them from names.
    """
    left = iter(step.inputs)
    right = iter(step.outputs)
    bindings: dict[tuple[str, RegisterSide, tuple[int, ...]], str] = {}
    for register in signature:
        for index in register.all_idxs():
            lane = ""
            if register.side != RegisterSide.RIGHT:
                lane = next(left)
            if register.side != RegisterSide.LEFT:
                lane = next(right)
            bindings[register.name, register.side, index] = lane
    hidden = pretty and reshaping

    def cell(port: tuple[str, str] | None, *, span: int = 1, empty: bool = True) -> str:
        if port is None:
            return "<TD></TD>" if empty else ""
        lane, label = port
        rowspan = f' ROWSPAN="{span}"' if span > 1 else ""
        return f'<TD{rowspan} PORT="{ports[lane]}">{label}</TD>'

    header = escape(step.name)
    if pretty:
        header = "" if hidden else f'<FONT POINT-SIZE="10">{header}</FONT>'
    rows = [f'<TR><TD COLSPAN="2">{header}</TD></TR>']
    for _, registers in signature.groups():
        sides: dict[RegisterSide, list[tuple[str, str]]] = {
            RegisterSide.LEFT: [],
            RegisterSide.RIGHT: [],
            RegisterSide.THRU: [],
        }
        for register in registers:
            for index in register.all_idxs():
                label = (
                    ""
                    if hidden
                    else escape(
                        register.name
                        + ("[" + ", ".join(map(str, index)) + "]" if index else "")
                    )
                )
                sides[register.side].append(
                    (bindings[register.name, register.side, index], label)
                )
        through = sides[RegisterSide.THRU]
        lefts, rights = sides[RegisterSide.LEFT], sides[RegisterSide.RIGHT]
        if through:
            for lane, label in through:
                rows.append(
                    f'<TR><TD COLSPAN="2" PORT="{ports[lane]}">{label}</TD></TR>'
                )
            continue
        common = min(len(lefts), len(rights))
        for i, (input_port, output_port) in enumerate(zip_longest(lefts, rights)):
            last_common = i == common - 1
            left_span = max(0, len(rights) - len(lefts)) + 1 if last_common else 1
            right_span = max(0, len(lefts) - len(rights)) + 1 if last_common else 1
            rows.append(
                "<TR>"
                + cell(input_port, span=left_span, empty=common == 0)
                + cell(output_port, span=right_span, empty=common == 0)
                + "</TR>"
            )
    attributes = ' BORDER="0" CELLBORDER="1" CELLSPACING="0"' if pretty else ""
    return f"<TABLE{attributes}>" + "".join(rows) + "</TABLE>"


def pretty_connection_attributes(edge: RegisterConnection) -> str:
    """Use the source compact-view edge attributes with existing connectivity."""
    return (
        f'label="{edge.width}", labelfloat=True, fontsize=10, '
        'arrowhead="dot", arrowsize=0.25'
    )
