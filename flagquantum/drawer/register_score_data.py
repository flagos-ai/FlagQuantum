# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 musical-score drawing records and wire symbols.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score.py

"""Translate register layout into musical-score presentation records and JSON."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from typing import Literal, TypedDict

from .register_score import RegisterScore, RegisterStep

SymbolClass = Literal[
    "Text", "TextBox", "RarrowTextBox", "LarrowTextBox", "Circle", "ModPlus"
]
PortSide = Literal["LEFT", "RIGHT", "THRU"]
WireSymbolSupplier = Callable[
    [RegisterStep, str | None, PortSide | None],
    tuple[SymbolClass, dict[str, str | int | bool]],
]


class ScorePort(TypedDict):
    symb_cls: SymbolClass
    symb_attributes: dict[str, str | int | bool]
    ident: str
    y: int
    seq_x: int
    topo_gen: int


class ScoreHorizontalLine(TypedDict):
    y: int
    seq_x_start: int
    seq_x_end: int
    flavor: str


class ScoreLabel(TypedDict):
    text: str
    fontsize: int


class ScoreVerticalLine(TypedDict):
    x: int
    top_y: int
    bottom_y: int
    label: ScoreLabel


class MusicalScoreData(TypedDict):
    max_x: int
    max_y: int
    soqs: list[ScorePort]
    hlines: list[ScoreHorizontalLine]
    vlines: list[ScoreVerticalLine]


def adjoint_wire_symbol(
    symbol: SymbolClass, attributes: dict[str, str | int | bool]
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Return the locked-source adjoint glyph with caller-owned attributes.

    Text and round boxes toggle a trailing dagger. The source removes all
    leading/trailing daggers when the text ends in a dagger. Directional boxes
    exchange arrow direction while retaining their text. Other glyphs retain
    their presentation. Invalid text attributes raise TypeError.

    Examples:
        >>> adjoint_wire_symbol("TextBox", {"text": "S"})
        ('TextBox', {'text': 'S†'})
    """
    result = dict(attributes)
    if symbol in {"Text", "TextBox"}:
        text = result["text"]
        if not isinstance(text, str):
            raise TypeError("text wire symbols require string text")
        result["text"] = text.strip("†") if text.endswith("†") else text + "†"
    if symbol == "RarrowTextBox":
        return "LarrowTextBox", result
    if symbol == "LarrowTextBox":
        return "RarrowTextBox", result
    return symbol, result


def _port_symbol(
    step: RegisterStep, register: str, side: PortSide
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    if step.gate == "and" and step.control_values is not None:
        if side == "THRU":
            return "Circle", {
                "filled": bool(step.control_values[step.inputs.index(register)])
            }
        return ("LarrowTextBox" if side == "RIGHT" else "RarrowTextBox"), {"text": "∧"}
    if step.gate in {"cx", "ccx"} and side == "THRU" and len(step.inputs) in {2, 3}:
        if register != step.inputs[-1]:
            return "Circle", {"filled": True}
        return "ModPlus", {}
    if step.gate == "x" and side == "THRU":
        return "ModPlus", {}
    if step.name == "qreg.control_h" and (register,) == step.inputs[:1]:
        return "Circle", {"filled": True}
    if side == "LEFT":
        text = "free" if step.name == "qreg.release" else register
        return "RarrowTextBox", {"text": text}
    if side == "RIGHT":
        text = "alloc" if step.name == "qreg.local" else register
        return "LarrowTextBox", {"text": text}
    return "TextBox", {"text": register}


def register_score_data(
    score: RegisterScore,
    *,
    classical: Collection[str] = (),
    wire_symbol: WireSymbolSupplier | None = None,
) -> MusicalScoreData:
    """Create source-compatible glyph records with one symbol per through port.

    Native register constructions are quantum. The optional classical set
    marks named lines when projecting a mixed register graph into the score.
    An explicit wire_symbol callback supplies port glyphs and the connector label
    (register=None, side=None), analogous to the source per-component protocol.
    It receives display identities only and cannot alter geometry or execution.
    Connector symbols must be Text with text and fontsize attributes.
    """
    classical_names = set(classical)
    unknown = classical_names - {line.register for line in score.lines}
    if unknown:
        raise ValueError(f"Unknown classical register identities: {sorted(unknown)!r}")
    output_column = len(score.columns)
    ports: list[ScorePort] = []
    hlines: list[ScoreHorizontalLine] = []
    connectors: list[ScoreVerticalLine] = []
    for line in score.lines:
        is_classical = line.register in classical_names
        kind = "CLASSICAL" if is_classical else "QUANTUM"
        hlines.append(
            {
                "y": line.lane,
                "seq_x_start": line.start,
                "seq_x_end": line.end,
                "flavor": f"HLineFlavor.{kind}",
            }
        )
        if is_classical:
            dtype = "CBit()" if line.width == 1 else f"CUInt({line.width})"
        else:
            dtype = "QBit()" if line.width == 1 else f"QAny({line.width})"
        for visible, side, x, generation in (
            (line.start == -1, "l", -1, 0),
            (line.end == output_column, "r", output_column, score.output_generation),
        ):
            if visible:
                ports.append(
                    {
                        "symb_cls": "Text",
                        "symb_attributes": {
                            "text": f"{line.register}/{dtype}",
                            "fontsize": 8,
                        },
                        "ident": f"dang,{line.register},{side},()",
                        "y": line.lane,
                        "seq_x": x,
                        "topo_gen": generation,
                    }
                )

    for column in score.columns:
        step = column.step
        registers = dict.fromkeys((*step.inputs, *step.outputs))
        for register, lane in zip(registers, column.lanes, strict=True):
            side_of_port: PortSide
            if register in step.inputs:
                side_of_port = "THRU" if register in step.outputs else "LEFT"
            else:
                side_of_port = "RIGHT"
            glyph, attributes = (
                _port_symbol(step, register, side_of_port)
                if wire_symbol is None
                else wire_symbol(step, register, side_of_port)
            )
            ports.append(
                {
                    "symb_cls": glyph,
                    "symb_attributes": attributes,
                    "ident": f"{column.ordinal},{register},Side.{side_of_port},()",
                    "y": lane,
                    "seq_x": column.sequence,
                    "topo_gen": column.generation,
                }
            )
        if column.lanes:
            title = (
                ""
                if step.gate in {"cx", "ccx", "and"}
                or step.name in {"qreg.local", "qreg.release"}
                else (
                    "X"
                    if step.gate == "x"
                    else step.gate or step.name.removeprefix("qreg.")
                )
            )
            label: ScoreLabel = {"text": title, "fontsize": 10}
            if wire_symbol is not None:
                symbol, attributes = wire_symbol(step, None, None)
                if (
                    symbol != "Text"
                    or not isinstance(attributes.get("text"), str)
                    or type(attributes.get("fontsize")) is not int
                ):
                    raise ValueError(
                        "connector wire symbols require Text with text and fontsize"
                    )
                label = {
                    "text": str(attributes["text"]),
                    "fontsize": int(attributes["fontsize"]),
                }
            connectors.append(
                {
                    "x": column.sequence,
                    "top_y": max(column.lanes),
                    "bottom_y": min(column.lanes),
                    "label": label,
                }
            )

    return {
        "max_x": max(
            output_column - 1, max((line.end for line in score.lines), default=0)
        ),
        "max_y": max((line.lane for line in score.lines), default=0),
        "soqs": ports,
        "hlines": hlines,
        "vlines": connectors,
    }


def export_register_score_json(data: MusicalScoreData) -> str:
    """Serialize musical-score records using the upstream JSON field names."""
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
