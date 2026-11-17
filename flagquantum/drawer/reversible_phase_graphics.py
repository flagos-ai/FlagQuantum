# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix Circuit::write_svg_or_html_diagram_to.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/circuit_diagram_html.cc

"""Lay out a physical reversible-phase circuit for SVG and HTML."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from xml.etree.ElementTree import Element, tostring

from ..compiler.artifacts.reaction_depth import reaction_depth_for_operations
from ..compiler.artifacts.reversible_phase import ReversiblePhaseOperation
from ..compiler.formats.kmb import KmxRecord
from ..compiler.formats.kmx_circuit import KmxCircuit, read_kmx_circuit
from ._reversible_phase_viewer import VIEWER_SCRIPT

_SVG_NS = "http://www.w3.org/2000/svg"
_CLASSICAL = {
    "NEG",
    "BIT_INVERT",
    "BIT_STORE0",
    "BIT_STORE1",
    "PUSH_CONDITION",
    "POP_CONDITION",
}
_THREE_WIRE = {"CCX", "CCZ", "CSWAP"}
_STABILIZER = {"CX", "CZ", "R", "HMR"}
_SWAPS = {"SWAP", "CSWAP"}
_Number = int | float


def _y(qubit: int) -> int:
    return 16 * (qubit + 1)


def _text(
    x: _Number,
    y: _Number,
    label: str,
    *,
    size: _Number = 6,
    anchor: str = "middle",
    fill: str = "black",
    baseline: str = "middle",
) -> Element:
    node = Element(
        "text",
        {
            "x": str(x) if isinstance(x, int) else f"{x:.17g}",
            "y": str(y) if isinstance(y, int) else f"{y:.17g}",
            "text-anchor": anchor,
            "dominant-baseline": baseline,
            "font-size": f"{size:.6g}",
            "font-family": "monospace",
            "fill": fill,
        },
    )
    node.text = label
    return node


def _rect(
    x: int, y: int, width: int, height: int, fill: str, stroke: str = "black"
) -> Element:
    return Element(
        "rect",
        x=str(x),
        y=str(y),
        width=str(width),
        height=str(height),
        stroke=stroke,
        fill=fill,
    )


def _circle(x: int, y: int, fill: str) -> Element:
    return Element("circle", cx=str(x), cy=str(y), r="4", stroke="black", fill=fill)


def _path(data: str, *, far: bool = False) -> Element:
    node = Element("path", d=data, stroke="black", fill="none")
    if far:
        node.set("vector-effect", "non-scaling-stroke")
    return node


def _classical_label(record: KmxRecord) -> str:
    condition = record.condition_bit
    target = f"b{record.result_bit}"
    if record.opcode == "NEG":
        return "neg" if condition is None else f"neg if b{condition}"
    if record.opcode == "PUSH_CONDITION":
        return f"push_cond b{condition}"
    if record.opcode == "POP_CONDITION":
        return "pop_cond"
    if condition is None:
        return (
            target
            + {"BIT_INVERT": "^=1", "BIT_STORE0": "=0", "BIT_STORE1": "=1"}[
                record.opcode
            ]
        )
    return (
        target
        + {"BIT_INVERT": "^=b", "BIT_STORE0": "&=!b", "BIT_STORE1": "|=b"}[
            record.opcode
        ]
        + str(condition)
    )


@dataclass
class _Graphic:
    x: int
    near: list[Element] = field(default_factory=list)
    far: list[Element] = field(default_factory=list)

    def both(self, node: Element) -> None:
        self.near.append(node)
        self.far.append(node)

    def classical_label(self, y: int, label: str) -> None:
        self.near.append(_text(self.x, y, label, fill="red"))
        self.far.append(_rect(self.x - 8, y - 4, 16, 4, "none", "red"))


@dataclass(frozen=True)
class _TagSpan:
    name: str
    start: int
    end: int


def _chunks(records: tuple[KmxRecord, ...]) -> Iterator[tuple[KmxRecord, ...]]:
    index = 0
    while index < len(records):
        record = records[index]
        stop = index + 1
        if record.opcode == "CX" and record.condition_bit is None:
            control, target = record.qubits
            targets = {target}
            while stop < len(records):
                other = records[stop]
                if (
                    other.opcode != "CX"
                    or other.condition_bit is not None
                    or other.qubits[0] != control
                    or other.qubits[-1] in targets
                ):
                    break
                targets.add(other.qubits[-1])
                stop += 1
        yield records[index:stop]
        index = stop


class _Scene:
    """Retain the source's column occupancy, wires and tag spans."""

    def __init__(self, circuit: KmxCircuit) -> None:
        self.qubits = circuit.qubits
        self.x = 0
        self.column_width = 32
        self.text_position: int | None = None
        self.has_ops = False
        self.saw_hmr = False
        self.last_used = [0] * self.qubits
        self.active = [True] * self.qubits
        for op in reversed(circuit.records):
            if op.opcode in {"R", "HMR"}:
                self.active[op.qubits[-1]] = op.opcode == "HMR"
        self.transitions = [[0] if active else [] for active in self.active]
        self.open_tags: list[tuple[str, int]] = []
        self.tag_rows: list[list[_TagSpan]] = []
        self.graphics: list[_Graphic] = []
        for index, register in enumerate(circuit.registers):
            for position, (kind, wire) in enumerate(register.members):
                name = register.name or f"r{index}"
                label = f"{name}[{position}]"
                if kind == "q":
                    graphic = self.place((wire,), classical=False)
                    y = _y(wire)
                    graphic.both(_rect(self.x - 12, y - 4, 24, 8, "lightgray"))
                    graphic.near.append(_text(self.x, y, label))
                else:
                    graphic = self.place((), classical=True)
                    assert self.text_position is not None
                    graphic.classical_label(
                        _y(self.text_position) - 8, f"{label}=b{wire}"
                    )
                    self.column_width = max(self.column_width, 48)
        for group in _chunks(circuit.records):
            op = group[0]
            if op.tag is not None:
                self.mark(op.tag)
            else:
                self.draw_group(group)
        self.x += self.column_width
        self.close_tags(0)
        for transitions in self.transitions:
            if len(transitions) % 2:
                transitions.append(self.x)

    def advance(self) -> None:
        self.x += self.column_width
        self.column_width = 16
        self.has_ops = False

    def place(
        self, wires: tuple[int, ...], *, classical: bool, guarded: bool = False
    ) -> _Graphic:
        new_column = bool(wires) and self.text_position is not None
        new_column |= self.saw_hmr and guarded
        if classical:
            if self.text_position is None or self.text_position >= self.qubits:
                self.text_position = 0
                new_column = True
            else:
                self.text_position += 1
        if new_column:
            if not classical:
                self.text_position = None
            self.advance()
            self.saw_hmr = False
        if wires:
            occupied = range(min(wires), max(wires) + 1)
            if any(self.last_used[q] >= self.x for q in occupied):
                self.advance()
            for q in occupied:
                self.last_used[q] = self.x
        self.has_ops = True
        graphic = _Graphic(self.x)
        self.graphics.append(graphic)
        return graphic

    def mark(self, tag: str) -> None:
        if tag.startswith("@enter:"):
            self.end_mark_column()
            if len(self.tag_rows) <= len(self.open_tags):
                self.tag_rows.append([])
            self.open_tags.append((tag[7:], self.x - 6))
        elif tag.startswith("@exit:"):
            for index in range(len(self.open_tags) - 1, -1, -1):
                if self.open_tags[index][0] == tag[6:]:
                    self.end_mark_column()
                    self.close_tags(index)
                    break

    def end_mark_column(self) -> None:
        if self.has_ops:
            self.advance()
            self.text_position = None
            self.saw_hmr = False

    def close_tags(self, index: int) -> None:
        while len(self.open_tags) > index:
            name, start = self.open_tags.pop()
            self.tag_rows[len(self.open_tags)].append(
                _TagSpan(name, start, max(start + 4, self.x - 10))
            )

    def draw_group(self, group: tuple[KmxRecord, ...]) -> None:
        op = group[0]
        classical = op.opcode in _CLASSICAL
        wires = tuple(q for record in group for q in record.qubits)
        graphic = self.place(
            wires, classical=classical, guarded=op.condition_bit is not None
        )
        x = self.x
        if wires:
            low, high = _y(min(wires)), _y(max(wires))
            target = op.qubits[-1]
            if op.opcode == "R" and not self.active[target]:
                self.transitions[target].append(x)
                self.active[target] = True
            elif op.opcode == "HMR" and self.active[target]:
                self.transitions[target].append(x)
                self.active[target] = False
            if low != high:
                stem = f"M{x},{low} L{x},{high}"
                graphic.near.append(_path(stem))
                graphic.far.append(_path(stem, far=True))
            if op.condition_bit is not None:
                graphic.near.extend(
                    (
                        _path(f"M{x - 1},{low - 1} l0,-5.5 m2,0 l0,5.5"),
                        _text(x, low - 8.5, f"b{op.condition_bit}", size=5),
                    )
                )
            if op.opcode in _SWAPS:
                y = _y(op.qubits[-2])
                graphic.near.append(_path(f"M{x - 4},{y - 4} l8,8 m-8,0 l8,-8"))
                if op.opcode == "CSWAP":
                    graphic.both(_circle(x, _y(op.qubits[0]), "black"))
            else:
                for control in reversed(op.qubits[:-1]):
                    graphic.both(_circle(x, _y(control), "black"))
        if classical:
            assert self.text_position is not None
            graphic.classical_label(_y(self.text_position) - 8, _classical_label(op))
            self.column_width = max(self.column_width, 48)
        for record in group:
            if record.qubits:
                self.draw_target(record, graphic)

    def draw_target(self, record: KmxRecord, graphic: _Graphic) -> None:
        x, y = self.x, _y(record.qubits[-1])
        opcode = record.opcode
        if opcode in {"X", "CX", "CCX"}:
            graphic.both(_circle(x, y, "white"))
            graphic.near.append(_path(f"M{x},{y - 4} l0,8 m-4,-4 l8,0"))
        elif opcode in {"Z", "CZ", "CCZ"}:
            graphic.both(_circle(x, y, "black"))
        elif opcode in _SWAPS:
            graphic.near.append(_path(f"M{x - 4},{y - 4} l8,8 m-8,0 l8,-8"))
        elif opcode == "Z_POW":
            assert record.angle is not None
            graphic.both(_rect(x - 4, y - 4, 120, 8, "white"))
            graphic.near.append(
                _text(x + 56, y, f"Z^{record.angle.to_decimal_half_turns()}")
            )
            self.saw_hmr = True
            self.column_width = max(self.column_width, 128)
        elif opcode == "HMR":
            graphic.both(_rect(x - 4, y - 4, 8, 8, "black"))
            graphic.near.extend(
                (
                    _text(x, y, "MX", fill="white"),
                    _path(f"M{x + 4},{y - 1} l3,0 m0,2 l-3,0"),
                    _text(x + 7, y, f"b{record.result_bit}", anchor="left"),
                )
            )
            self.saw_hmr = True
            self.column_width = max(self.column_width, 32)
        elif opcode == "R":
            graphic.near.append(_text(x, y, "|0⟩", anchor="end"))

    @property
    def height(self) -> int:
        return _y(self.qubits) + 12 * len(self.tag_rows)

    def wire_rows(self) -> Iterator[tuple[list[int], list[list[Element]]]]:
        for q, transitions in enumerate(self.transitions):
            y = _y(q)
            yield transitions, [
                [_path(f"M{a},{y} L{b},{y}")]
                for a, b in zip(transitions[::2], transitions[1::2], strict=True)
            ]
        for depth, row in enumerate(self.tag_rows):
            y = _y(self.qubits) + depth * 12
            endpoints: list[int] = []
            groups = []
            for tag in row:
                a, b = tag.start, tag.end
                endpoints.extend((a, b))
                size = min(6, max(1, (b - a) / max(1, len(tag.name.encode())) * 1.5))
                groups.append(
                    [
                        _path(f"M{a},{y - 4} L{a},{y} L{b},{y} L{b},{y - 4}"),
                        _text((a + b) / 2, y - 1, tag.name, size=size, baseline="auto"),
                    ]
                )
            yield endpoints, groups


def render_reversible_phase_svg(kmx: str) -> str:
    """Render a standalone physical circuit diagram."""
    return render_circuit_svg(read_kmx_circuit(kmx))


def render_circuit_svg(circuit: KmxCircuit) -> str:
    """Render an already parsed circuit as standalone SVG."""
    scene = _Scene(circuit)
    root = Element("svg", viewBox=f"0 0 {scene.x} {scene.height}", xmlns=_SVG_NS)
    for _, groups in scene.wire_rows():
        for group in groups:
            root.extend(group)
    for graphic in scene.graphics:
        root.extend(graphic.near)
    return tostring(root, encoding="unicode")


def _statistics(records: Sequence[KmxRecord]) -> str:
    classical = stabilizer = three_wire = t = rotation = 0
    touched: set[int] = set()
    operations = []
    for op in records:
        touched.update(op.qubits)
        if op.tag is not None or op.opcode.startswith("DEBUG_PRINT"):
            continue
        operations.append(
            ReversiblePhaseOperation(
                op.opcode.lower(), op.qubits, op.result_bit, op.condition_bit, op.angle
            )
        )
        if op.opcode in _CLASSICAL - {"NEG"} or op.opcode in {"X", "Z", "SWAP"}:
            classical += 1
        elif op.opcode in _STABILIZER:
            stabilizer += 1
        elif op.opcode in _THREE_WIRE:
            three_wire += 1
        elif op.angle is not None:
            if not op.angle.is_multiple_of_45_degrees():
                rotation += 1
            elif not op.angle.is_multiple_of_90_degrees():
                t += 1
            elif not op.angle.is_multiple_of_180_degrees():
                stabilizer += 1
            else:
                classical += 1
    stats = f"bit+x+z+swap={classical}, cx+cz+r+hmr={stabilizer}, ccx+ccz=<strong>{three_wire}</strong>"
    if t:
        stats += f", t=<strong>{t}</strong>"
    if rotation:
        stats += f", rz_theta=<strong>{rotation}</strong>"
    return (
        f"{stats}, reaction depth=<strong>{reaction_depth_for_operations(operations)}</strong>, "
        f"touched qubits=<strong>{len(touched)}</strong>"
    )


def _markup(nodes: Sequence[Element]) -> str:
    return "".join(tostring(node, encoding="unicode") for node in nodes)


def render_reversible_phase_html(kmx: str) -> str:
    """Render an interactive physical diagram with zoom, pan and detail levels."""
    return render_circuit_html(read_kmx_circuit(kmx))


def render_circuit_html(circuit: KmxCircuit) -> str:
    """Render an already parsed circuit as an interactive HTML document."""
    scene = _Scene(circuit)
    labels = [_text(0, _y(q), f"q{q}:", anchor="end") for q in range(scene.qubits)]
    wires = tuple(scene.wire_rows())
    payload = {
        "objects": [[0, _markup(labels), ""]]
        + [
            [graphic.x, _markup(graphic.near), _markup(graphic.far)]
            for graphic in scene.graphics
        ],
        "transitions": [transitions for transitions, _ in wires],
        "wires": [[_markup(group) for group in groups] for _, groups in wires],
        "width": scene.x,
        "height": scene.height,
    }
    # Prevent a KMX label from ending the script element or evaluating JavaScript.
    encoded = json.dumps(payload, separators=(",", ":")).replace("<", r"\u003c")
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        "<title>Circuit Viewer</title></head><body>"
        f"<div>Controls: mouse wheel to zoom, click and drag to pan. (Circuit stats: {_statistics(circuit.records)})</div>"
        '<svg id="viewportSvg" style="width: 100%; height: 95vh; border: 1px solid black" '
        'viewBox="0 0 2000 800" preserveAspectRatio="none" aria-label="Physical circuit diagram"></svg>'
        f'<script type="application/json" id="circuit-data">{encoded}</script>'
        f"<script>{VIEWER_SCRIPT}</script></body></html>"
    )
