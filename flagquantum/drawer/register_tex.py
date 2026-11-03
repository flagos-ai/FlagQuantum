"""TeX layout for named operations and register rows in circuit IR."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..experimental.controlled_pauli import decode_controlled_pauli
from .latex_drawer import _as_ir
from .paper_tex import source_link


@dataclass(frozen=True)
class GateSymbol:
    """Display a named operation with leading controls and trailing targets."""

    label: str
    controls: tuple[int, ...] = ()
    rotate: bool = False


@dataclass(frozen=True)
class RegisterLayout:
    """Geometry and symbols for one circuit panel, in millimetres."""

    columns: Sequence[float]
    symbols: Mapping[str, GateSymbol]
    labels: Mapping[int, str]
    buses: tuple[int, ...] = ()
    workspace_pairs: tuple[str, str] | None = None
    origin: float = 0
    end: float | None = None


def register_circuit_tex(
    program: object,
    layout: RegisterLayout,
    comparison: tuple[object, RegisterLayout] | None = None,
    *,
    row_spacing: float = 4.5,
) -> str:
    """Render one circuit or an equality between two circuit panels."""
    source = r"""
\begin{tikzpicture}[x=1mm,y=1mm,line width=.36pt,
  box/.style={draw,fill=white,inner sep=.65mm},
  every node/.style={font=\fontsize{7}{8}\selectfont}]
"""
    panels = [(program, layout)]
    if comparison is not None:
        panels.append(comparison)
    for panel_index, (panel, spec) in enumerate(panels):
        ir = _as_ir(panel)
        gates = list(zip(spec.columns, ir.instructions, strict=True))
        end = spec.end if spec.end is not None else max(spec.columns) + 2
        if spec is not layout:
            left_end = layout.end if layout.end is not None else max(layout.columns) + 2
            separator = left_end + 0.36 * (spec.origin - left_end)
            source += rf"\node at ({separator},{-row_spacing}) {{\(=\)}};" + "\n"
        for wire, label in spec.labels.items():
            row = -row_spacing * wire
            source += rf"""
\draw ({spec.origin},{row}) -- ({end},{row});
\node[left] at ({spec.origin},{row}) {{{label}}};
"""
            if label:
                source += (
                    source_link(
                        f"input-{panel_index}-{wire}", spec.origin - 3, row, 5, 3
                    )
                    + "\n"
                )
            if wire in spec.buses:
                source += (
                    rf"\draw ({spec.origin + 1.6},{row}) ++(-.4mm,-.9mm) -- ++(.8mm,1.8mm);"
                    + "\n"
                )
        if spec.workspace_pairs is not None:
            start_name, stop_name = spec.workspace_pairs
            starts: dict[int, float] = {}
            for column, gate in gates:
                wire = gate.wires[-1]
                name = gate.name
                if gate.metadata.get("quantum_controls"):
                    bits = "".join(
                        str(bit) for _, bit in gate.metadata["quantum_controls"]
                    )
                    name = f'c{bits}{gate.metadata["target_gate"]}'
                if name == start_name:
                    if wire in starts:
                        raise ValueError("Overlapping workspace lifetimes.")
                    starts[wire] = column
                elif name == stop_name:
                    start = starts.pop(wire)
                    row = -row_spacing * wire
                    source += rf"\draw ({start},{row}) -- ({column},{row});" + "\n"
            if starts:
                raise ValueError("Workspace lifetime has no closing instruction.")

        for step, (column, gate) in enumerate(gates):
            symbol = spec.symbols.get(gate.name)
            if symbol is None:
                controls = gate.metadata.get("quantum_controls")
                pattern = (
                    (
                        "".join(str(bit) for _, bit in controls),
                        gate.metadata["target_gate"].upper(),
                    )
                    if controls
                    else decode_controlled_pauli(gate.name)
                )
                if pattern is None or gate.matrix is None:
                    raise ValueError(f"No drawing symbol for {gate.name}.")
                bits, pauli = pattern
                symbol = GateSymbol(
                    r"$\oplus$" if pauli == "X" else "$Z$", tuple(map(int, bits))
                )
            rows = [-row_spacing * wire for wire in gate.wires]
            controls = rows[: len(symbol.controls)]
            targets = rows[len(controls) :]
            if not targets:
                raise ValueError(f"No drawing target for {gate.name}.")
            middle = (min(targets) + max(targets)) / 2
            span = max(targets) - min(targets)
            key = f"gate-{panel_index}-{step}"
            width = 3.4 if symbol.rotate else max(3.4, len(symbol.label) * 0.65)
            source += source_link(key, column, middle, width, span + 2.8) + "\n"
            for row in controls:
                source += source_link(key, column, row, 2.3, 2.3) + "\n"
            if controls:
                source += (
                    rf"\draw ({column},{min(rows)}) -- ({column},{max(rows)});" + "\n"
                )
            for row, bit in zip(controls, symbol.controls, strict=True):
                fill = "black" if bit else "white"
                source += (
                    rf"\filldraw[fill={fill}] ({column},{row}) circle[radius=.39mm];"
                    + "\n"
                )
            if symbol.label == r"$\oplus$":
                source += rf"""
\filldraw[fill=white] ({column},{middle}) circle[radius=.83mm];
\draw ({column},{middle}) ++(-.83mm,0) -- ++(1.66mm,0);
\draw ({column},{middle}) ++(0,-.83mm) -- ++(0,1.66mm);
"""
            else:
                dimension = "width" if symbol.rotate else "height"
                rotation = (
                    "rotate=90,font=\\fontsize{6.5}{8}\\selectfont,"
                    if symbol.rotate
                    else ""
                )
                source += rf"""
\node[box,{rotation}minimum {dimension}={span + 2.9}mm] at ({column},{middle}) {{{symbol.label}}};
"""
    return source + "\\end{tikzpicture}\n"
