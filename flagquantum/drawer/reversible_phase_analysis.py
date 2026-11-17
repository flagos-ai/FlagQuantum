# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix Circuit::write_analysis_svg_to.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/circuit.cc

"""Render physical reversible-phase circuit analysis as a resource SVG."""

from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement, tostring

from ..compiler.formats.kmb import KmxRecord
from ..compiler.formats.kmx_circuit import read_kmx_circuit


@dataclass
class _Block:
    name: str
    path: str
    starting_weight: int
    touched: set[int]


@dataclass
class _Group:
    depth: int
    weight: int = 0
    qubits: int = 0
    hits: int = 0


def _half_ccz_states(record: KmxRecord) -> int:
    if record.opcode in {"CCX", "CCZ", "CSWAP"}:
        return 2
    if record.angle is None:
        return 0
    if not record.angle.is_multiple_of_45_degrees():
        return 8
    return int(not record.angle.is_multiple_of_90_degrees())


def _group_operations(
    records: tuple[KmxRecord, ...],
) -> tuple[int, dict[str, _Group]]:
    active: set[int] = set()
    stack = [_Block("<entire circuit>", ":<entire circuit>", 0, set())]
    groups: dict[str, _Group] = {}
    weight = 0

    def close_block() -> None:
        depth = len(stack)
        block = stack.pop()
        group = groups.get(block.path)
        if group is None:
            group = groups[block.path] = _Group(depth)
        group.weight += weight - block.starting_weight
        group.qubits = max(group.qubits, len(block.touched))
        group.hits += 1

    for record in records:
        if record.tag is not None:
            if record.tag.startswith("@enter:"):
                name = record.tag[len("@enter:") :]
                stack.append(
                    _Block(name, f"{stack[-1].path}:{name}", weight, active.copy())
                )
            elif record.tag.startswith("@exit:"):
                name = record.tag[len("@exit:") :]
                for index in range(len(stack) - 1, 0, -1):
                    if stack[index].name == name:
                        while len(stack) > index:
                            close_block()
                        break
            continue

        stack[-1].touched.update(record.qubits)
        if record.opcode == "HMR":
            active.discard(record.qubits[0])
        else:
            active.update(record.qubits)
        weight += _half_ccz_states(record)

    while stack:
        close_block()
    return weight, groups


def _rect(
    svg: Element, x: float, y: float, width: float, height: float, fill: str
) -> None:
    node = SubElement(
        svg,
        "rect",
        x=format(x, ".6g"),
        y=format(y, ".6g"),
        width=format(width, ".6g"),
        height=format(height, ".6g"),
        stroke="black",
        fill=fill,
    )
    if fill == "red":
        node.set("opacity", "0.8")


def _text(svg: Element, x: float, y: float, size: float, title: str) -> None:
    SubElement(
        svg,
        "text",
        {
            "x": format(x, ".6g"),
            "y": format(y, ".6g"),
            "font-size": format(size, ".6g"),
            "dominant-baseline": "bottom",
            "text-anchor": "middle",
            "fill": "black",
        },
    ).text = title


def render_reversible_phase_analysis_svg(
    kmx: str, *, reference_qubit_count: int = 0
) -> str:
    """Render source block summaries with vertical markers at reference intervals."""
    if reference_qubit_count < 0:
        raise ValueError("Reference qubit count must be nonnegative")
    circuit = read_kmx_circuit(kmx)
    qubits = circuit.qubits
    total, groups = _group_operations(circuit.records)
    svg = Element(
        "svg",
        viewBox="-100 -140 1200.0 340.0",
        style="display: block; margin: auto;",
        xmlns="http://www.w3.org/2000/svg",
    )
    x_scale = 1000 / max(total, 2)
    y_scale = 100 / max(qubits, 1)
    layer_ends = [0] * (1 + max(group.depth for group in groups.values()))

    for path, group in groups.items():
        depth = group.depth
        start = max(layer_ends[: depth + 1])
        layer_ends[depth] = start + group.weight
        x, width = start * x_scale, group.weight * x_scale
        height = y_scale * group.qubits
        _rect(svg, x, -height, width, height, "blue")
        if depth > 1:
            label = path.rsplit(":", 1)[-1]
            if group.hits > 1:
                label += f" (x{group.hits})"
            label_bytes = len(label.encode())
            font_size = (
                max(0.5, min(5, width * 2 / label_bytes)) if label_bytes else 0.5
            )
            _rect(svg, x, depth * 10 - 20, width, 10, "red")
            _text(svg, x + width / 2, depth * 10 - 13, font_size, label)

    _text(
        svg,
        500,
        -100,
        10,
        f"{qubits} qubits, {total / 2:.6g} CCZ states "
        "(=CCX + CCZ + CSWAP + Z_POW[level=3]/2 + 4 Z_POW[level>=4])",
    )
    if reference_qubit_count:
        path = "".join(
            f" M500,{-y_scale * index:.6g} l50,0"
            for index in range(reference_qubit_count, qubits, reference_qubit_count)
        )
        SubElement(svg, "path", d=path, fill="none", stroke="green")
    return tostring(svg, encoding="unicode")
