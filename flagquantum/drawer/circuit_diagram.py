# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the pinned Qualtran v0.7.0 logical qpic exporter.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/qpic_diagram.py

"""Export logical register diagrams in qpic format and render through native TeX."""

import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from ..core.ir.program.registers import Register, RegisterSide, RegisterType
from ..core.ir.program.signatures import Signature
from .component_score import _label
from .latex_drawer import _escape_text
from .latex_figure import LatexFigure, _run
from .register_graph_table import _port_label
from .register_score import RegisterScore, _register_groups
from .register_score_data import (
    PortSide,
    WireSymbolSupplier,
    _port_symbol,
    _symbol_port_order,
)

_TEX_SYMBOLS = {
    "<=": r"$\le$",
    ">=": r"$\ge$",
    "<": r"$<$",
    ">": r"$>$",
    "⨁": r"$\oplus$",
    "⊕": r"$\oplus$",
    "⇋": r"$\rightleftharpoons$",
    "⨂": r"$\otimes$",
    "∧": r"$\land$",
    "†": r"$^\dagger$",
}
_SYMBOL_PATTERN = re.compile("|".join(map(re.escape, _TEX_SYMBOLS)))


def _label_text(label: str, scale: float = 0.8) -> str:
    escaped: list[str] = []
    start = 0
    for match in _SYMBOL_PATTERN.finditer(label):
        escaped.extend(
            (_escape_text(label[start : match.start()]), _TEX_SYMBOLS[match[0]])
        )
        start = match.end()
    escaped.append(_escape_text(label[start:]))
    return r"\textrm{\scalebox{" + str(scale) + "}{" + "".join(escaped) + "}}"


def _dtype_label(wire: str, dtype: RegisterType) -> list[str]:
    width = dtype.quantum_bits
    if dtype.encoding == "bit":
        return []
    if dtype.encoding == "uint":
        bound = dict(dtype.parameters).get("bound")
        name = f"QUInt({width})" if bound is None else f"BQUInt({width}, {bound})"
    elif dtype.encoding == "opaque":
        name = f"QAny({width})"
    elif dtype.encoding == "signed":
        name = f"QInt({width})"
    else:
        name = str(dtype)
    return [f"{wire} / {_label_text(name, scale=0.5)}"]


class _DiagramWires:
    """Assign a stable name to each live identity and reuse released suffixes."""

    def __init__(self) -> None:
        self.live: dict[str, str] = {}
        self.defined: dict[str, str] = {}
        self.rows = ["DEFINE off color=white", "DEFINE on color=black"]

    def allocate(self, identity: str, label: str, *, initial: bool = False) -> str:
        if identity in self.live:
            raise ValueError(f"Logical wire {identity!r} is already live")
        prefix = re.sub(r"[^a-zA-Z0-9_]", "_", label)
        if not prefix or prefix[0].isdigit():
            prefix = "q_" + prefix
        active = set(self.live.values())
        suffix = 0
        wire = prefix
        while (
            wire == "_empty_wire"
            or wire in active
            or self.defined.get(wire, label) != label
        ):
            suffix += 1
            wire = f"{prefix}_{suffix}"
        self.live[identity] = wire
        if wire not in self.defined:
            self.defined[wire] = label
            self.rows.append(f"{wire} W " + (_label_text(label) if initial else "off"))
        return wire


def _port_metadata(
    keys: Sequence[str], registers: Sequence[Register], widths: dict[str, int]
) -> dict[str, tuple[Register, str]]:
    return {
        key: (register, _port_label(register.name, index))
        for register, group in _register_groups(keys, registers, widths)
        for key, index in zip(group, register.all_idxs(), strict=True)
    }


def export_register_diagram_qpic(
    score: RegisterScore,
    signature: Signature,
    *,
    wire_symbol: WireSymbolSupplier | None = None,
) -> str:
    """Export one qpic wire per logical register element with source call ordering."""
    widths = {line.register: line.width for line in score.lines}
    wires = _DiagramWires()
    input_keys = tuple(
        _label(register.name, index)
        for register in signature.lefts()
        for index in register.all_idxs()
    )
    for identity, (register, label) in _port_metadata(
        input_keys, tuple(signature.lefts()), widths
    ).items():
        wire = wires.allocate(identity, label, initial=True)
        wires.rows.extend(_dtype_label(wire, register.dtype))
    gates = []
    columns = {column.ordinal: column for column in score.columns}
    for ordinal, (incoming, outgoing) in _symbol_port_order(score).items():
        if ordinal < 0:
            continue
        step = columns[ordinal].step
        if not (incoming or outgoing):
            if "_empty_wire" not in wires.defined:
                wires.defined["_empty_wire"] = ""
                wires.rows.insert(2, "_empty_wire W off")
            width = 5 + 4 * len(step.name)
            gates.append(
                f"_empty_wire G:width={width}:shape=8 {_label_text(step.name)}"
            )
            continue
        inputs = _port_metadata(step.inputs, step.input_registers, widths)
        outputs = _port_metadata(step.outputs, step.output_registers, widths)
        controls, targets, dtype_labels = [], [], []
        for identity in (*incoming, *(key for key in outgoing if key not in inputs)):
            side: PortSide
            if identity not in inputs:
                side = "RIGHT"
                register, label = outputs[identity]
                name = wires.allocate(identity, label)
                dtype_labels.extend(_dtype_label(name, register.dtype))
                name += ":on"
            elif identity not in outputs:
                side = "LEFT"
                _, label = inputs[identity]
                name = wires.live.pop(identity) + ":off"
            else:
                side = "THRU"
                _, label = inputs[identity]
                name = wires.live[identity]
            glyph, attributes = (
                _port_symbol(step, identity, side, text=label)
                if wire_symbol is None
                else wire_symbol(step, identity, side)
            )
            if glyph == "Circle":
                controls.append(name if attributes["filled"] else "-" + name)
            elif glyph == "ModPlus":
                controls.append("+" + name)
            else:
                text = str(attributes["text"]).strip()
                width = 5 + 4 * len(text)
                shape = (
                    ">"
                    if glyph == "RarrowTextBox" and not text
                    else "<" if glyph == "LarrowTextBox" and not text else "box"
                )
                targets.append(
                    f"{name} G:width={width}:shape={shape} {_label_text(text)}"
                )
        gates.extend((" ".join((*targets, *controls)), *dtype_labels))

    final = _port_metadata(score.output_order, tuple(signature.rights()), widths)
    labels = [
        f"{wires.live[key]} / {_label_text(label, scale=0.5)} "
        for key, (register, label) in final.items()
        if register.side == RegisterSide.RIGHT
    ]
    return "\n".join(
        (
            *wires.rows,
            "LABEL length=10",
            *gates,
            *labels,
            *(["LABEL length=10"] if labels else []),
        )
    )


def qpic_latex_figure(
    source: str, *, executable: str = "qpic", timeout: float = 30.0
) -> LatexFigure:
    """Translate qpic into TikZ, then use the native TeX figure for rendering."""
    figure = LatexFigure("", standalone=False, timeout=timeout)
    converter = shutil.which(executable)
    if converter is None:
        raise ImportError(f"Logical qpic conversion requires {executable!r}")
    with TemporaryDirectory(prefix="flagquantum-qpic-") as temp:
        root = Path(temp)
        (root / "diagram.qpic").write_text(source, encoding="utf-8")
        _run(
            [
                str(Path(converter).absolute()),
                "diagram.qpic",
                "-f",
                "tikz",
                "-o",
                "diagram.tikz",
            ],
            root,
            timeout,
        )
        target = root / "diagram.tikz"
        if not target.is_file():
            raise RuntimeError("qpic completed without producing diagram.tikz")
        tex = target.read_text(encoding="utf-8")
        if "\\begin{tikzpicture}" not in tex:
            raise RuntimeError("qpic did not produce a TikZ diagram")
    return LatexFigure(tex, standalone=False, timeout=figure.timeout)
