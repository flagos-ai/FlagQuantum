"""Compact journal figure composition; numeric labels, not glyph size, carry data."""

from __future__ import annotations

import re
from dataclasses import replace

from ..core.ir import CircuitIR
from .hybrid_layers import ClassicalLayer
from .latex_drawer import _circuit_body, _escape_text

_FONT = r"\fontencoding{T1}\fontfamily{phv}\fontsize{7}{8.5}\selectfont"
_CANVAS_MM = 181.5


def _shape_tex(shape: tuple[int, ...]) -> str:
    return r"\,$\times$\,\allowbreak{}".join(str(n) for n in shape)


def _label_tex(label: str) -> str:
    return r"\allowbreak{}".join(
        _escape_text(part) for part in re.split(r"(?<=[a-z])(?=[A-Z])", label)
    )


def _tensor_tex(shape: tuple[int, ...], x: float, name: str) -> str:
    lines = [rf"\coordinate ({name}) at ({x:g},0);"]
    half_width = 1.0
    if len(shape) == 1:
        count = min(shape[0], 4)
        for i in range(count):
            y = (count - 1) * 1.1 - i * 2.2
            if shape[0] > 4 and i == 3:
                lines.append(rf"\node[inner sep=0] at ({x:g},{y:g}) {{$\vdots$}};")
            else:
                lines.append(
                    rf"\draw[draw=fhClassical!80!black,fill=fhClassical!22,line width=.4pt]"
                    rf" ({x-1:g},{y-0.8:g}) rectangle ({x+1:g},{y+0.8:g});"
                )
    else:
        count = min(shape[0], 3) if len(shape) == 3 else 1
        width, height = min(8, max(3, shape[-1])), min(8, max(3, shape[-2]))
        half_width = width / 2 + ((count - 1) * 0.7 + 1) / 2
        center_x = ((count - 1) * 0.7 + 1) / 2
        center_y = (count - 1) * 0.35
        for i in reversed(range(count)):
            shift = i * 0.7
            shift_x, shift_y = shift - center_x, shift - center_y
            lines.append(
                rf"\draw[draw=fhClassical!80!black,fill=fhClassical!18,line width=.4pt]"
                rf" ({x-width/2+shift_x:g},{-height/2+shift_y:g}) --"
                rf" ({x+width/2+shift_x:g},{-height/2+shift_y:g}) --"
                rf" ({x+width/2+shift_x+1:g},{height/2+shift_y:g}) --"
                rf" ({x-width/2+shift_x+1:g},{height/2+shift_y:g}) -- cycle;"
            )
    lines += [
        rf"\coordinate ({name}-west) at ({x-half_width:g},0);",
        rf"\coordinate ({name}-east) at ({x+half_width:g},0);",
    ]
    return "\n".join(lines)


def _stage_tex(
    label: str,
    layers: tuple[ClassicalLayer, ...],
    trainable: bool,
    count: int | None,
    unit: str,
    *,
    include_input: bool,
) -> str:
    if not layers:
        description = "unspecified " + unit if count is None else f"{count} {unit}"
        return (
            r"\begin{tikzpicture}[anchor=center,baseline=0mm,font={" + _FONT + r"}]"
            r"\node[draw=black!65,line width=.5pt,fill=white,text width=22mm,"
            r"minimum height=8mm,align=center,inner sep=3pt] {"
            + _label_tex(label)
            + r"};\node[anchor=north,align=center,text width=24mm] at (0,-.6) {"
            + description
            + r"};\end{tikzpicture}"
        )
    tensors = [
        (layer.label, layer.output_shape, trainable and layer.trainable)
        for layer in layers
    ]
    if include_input:
        tensors.insert(0, ("Input", layers[0].input_shape, False))
    lines = [
        r"\begin{tikzpicture}[anchor=center,baseline=0mm,x=1mm,y=1mm,font={"
        + _FONT
        + r"}]"
    ]
    for i, (title, shape, learned) in enumerate(tensors):
        x = i * 15.5
        lines += [
            _tensor_tex(shape, x, f"tensor{i}"),
            rf"\node[anchor=south,text width=18mm,align=center,inner sep=0] at ({x:g},7) {{"
            + _label_tex(title)
            + (r"\textsuperscript{*}" if learned else "")
            + "};",
            rf"\node[anchor=north,inner sep=0,text width=14mm,align=center] at ({x:g},-6) {{"
            + _shape_tex(shape)
            + "};",
        ]
        if i:
            lines.append(
                rf"\draw[fh/data,line width=.5pt] (tensor{i-1}-east) -- (tensor{i}-west);"
            )
    return "\n".join([*lines, r"\end{tikzpicture}"])


def _quantum_tex(ir: CircuitIR, inline: bool) -> str:
    if inline:
        content = r"\begin{yquant}" + _circuit_body(ir) + r"\end{yquant}"
    else:
        content = r"\path[use as bounding box] (-1.4,-.4) rectangle (1.4,.4);\draw[draw=fhInk,line width=.5pt] (-1.4,0)--(-.75,0) (.75,0)--(1.4,0);\node[draw=fhQuantum,fill=fhQuantum!9,line width=.6pt,minimum width=15mm,minimum height=8mm,anchor=center] at (0,0) {$U$};"
    return (
        r"\begin{tikzpicture}[anchor=center,baseline=(current bounding box.center),font={"
        + _FONT
        + r"}]"
        + content
        + r"\end{tikzpicture}"
    )


def _readout_tex(label: str | None, count: int | None) -> str:
    text = "Readout unspecified" if label is None else _escape_text(label)
    lines = [
        r"\begin{tikzpicture}[anchor=center,baseline=0mm,x=1mm,y=1mm,font={"
        + _FONT
        + "}]"
    ]
    if count is not None:
        lines.append(_tensor_tex((count,), 0, "readout"))
    else:
        lines.append(r"\node[inner sep=0] at (0,0) {$z$};")
    lines.append(
        r"\node[anchor=south,text width=20mm,align=center,inner sep=0] at (0,7) {"
        + text
        + "};"
    )
    if count is not None:
        lines.append(
            r"\node[anchor=north,inner sep=0] at (0,-6) {" + f"{count} values" + "};"
        )
    return "\n".join([*lines, r"\end{tikzpicture}"])


def _architecture(
    encoder: str,
    head: str,
    quantum: str,
    readout: str,
    quantum_size: str,
    training: bool,
    trainable: tuple[bool, bool, bool],
    declared_readout: bool,
) -> str:
    lines = [
        r"\begin{fhdiagram}[font={"
        + _FONT
        + r"},fh/label/.append style={font={"
        + _FONT
        + r"}}]",
        r"\node[anchor=base,inner sep=0,outer sep=0] (encoder) {" + encoder + "};",
        r"\node[anchor=base west,inner sep=0,outer sep=0,minimum width=28mm] (quantum) at ([xshift=5mm]encoder.base east) {"
        + quantum
        + "};",
        r"\node[anchor=base west,inner sep=0,outer sep=0] (readout) at ([xshift=5mm]quantum.base east) {"
        + readout
        + "};",
        r"\node[anchor=base west,inner sep=0,outer sep=0] (head) at ([xshift=5mm]readout.base east) {"
        + head
        + "};",
        r"\node[anchor=north,inner sep=0] (quantum-size) at ([yshift=-3mm]quantum.south) {"
        + quantum_size
        + "};",
        r"\fhConnect{encoder.base east}{quantum.base west}{$x$}",
        r"\fhConnect{quantum.base east}{readout.base west}{}",
        r"\fhConnect"
        + ("[fh/measurement]" if declared_readout else "")
        + r"{readout.base east}{head.base west}{$z$}",
        r"\coordinate (titles) at ([yshift=4mm]current bounding box.north);",
    ]
    for name, title, color in (
        ("encoder", "Classical encoder", "fhClassical"),
        ("quantum", "Quantum circuit", "fhQuantum"),
        ("readout", "Readout", "fhInk"),
        ("head", "Classical head", "fhClassical"),
    ):
        marker = r"\textsuperscript{*}" if name == "quantum" and trainable[1] else ""
        lines += [
            r"\node[anchor=south west,inner sep=0,font={"
            + _FONT
            + r"\bfseries}] at ("
            + name
            + r".west |- titles) {"
            + title
            + marker
            + "};",
            rf"\draw[draw={color},line width=.65pt] ([yshift=-1mm]{name}.west |- titles) -- ([yshift=-1mm]{name}.east |- titles);",
        ]
    if training:
        lines += [
            r"\coordinate (floor) at ([yshift=-9mm]current bounding box.south);",
            r"\node[inner sep=2pt,fill=white,align=center] (optimizer) at (head.south |- floor) {Loss + optimizer"
            + ("" if any(trainable) else r"\\No trainable stages")
            + "};",
            r"\draw[fh/data,line width=.5pt] (head.south) -- node[fh/label,midway,left]{predictions} (optimizer.north);",
            r"\coordinate (updates) at ([yshift=-5mm]optimizer.south);",
        ]
        if trainable[0] or trainable[1]:
            leftmost = "encoder" if trainable[0] else "quantum"
            lines.append(
                r"\draw[fh/feedback,draw=fhInk,line width=.5pt] (optimizer.south) -- (optimizer.south |- updates) -- "
                + f"({leftmost}.south |- updates);"
            )
        for name, enabled in zip(
            ("encoder", "quantum", "head"), trainable, strict=True
        ):
            if not enabled:
                continue
            if name == "head":
                lines += [
                    r"\coordinate (headfeedback) at ([xshift=6mm]head.east);",
                    r"\draw[fh/data,fh/feedback,line width=.5pt] (optimizer.east) -| (headfeedback |- optimizer.east) -- (headfeedback) -- (head.east);",
                    r"\node[fh/label,anchor=north east,yshift=-1mm] at (headfeedback |- optimizer.east) {head update};",
                ]
            else:
                target = "quantum-size" if name == "quantum" else name
                lines.append(
                    rf"\draw[fh/data,fh/feedback,line width=.5pt] ({name}.south |- updates) -- node[fh/label,midway,right]{{{name} update}} ({target}.south);"
                )
    return "\n".join([*lines, r"\end{fhdiagram}"])


def _width_guard(register: str) -> str:
    return (
        rf"\ifdim\wd{register}>{_CANVAS_MM}mm"
        r"\PackageError{flagquantum}{Figure panel exceeds 181.5 mm. Reduce gates_per_panel, "
        r"shorten labels, choose quantum_view=overview or summarize classical stages}{Text is never shrunk to fit.}\fi"
    )


def _paper_panel(content: str, heading: str) -> str:
    return (
        rf"\begin{{minipage}}{{{_CANVAS_MM}mm}}"
        + _FONT
        + (r"\textbf{" + heading + r"}\par\vspace{3pt}" if heading else "")
        + r"\centering"
        + r"\setbox\fhPanelA=\hbox{"
        + content
        + "}"
        + _width_guard(r"\fhPanelA")
        + r"\noindent\box\fhPanelA\par\end{minipage}"
    )


def _circuit_grid(ir: CircuitIR, panel_size: int) -> str:
    pictures: list[tuple[str, str]] = []
    for start in range(0, max(1, len(ir.instructions)), panel_size):
        end = min(start + panel_size, len(ir.instructions))
        last = end == len(ir.instructions)
        panel = replace(
            ir,
            instructions=ir.instructions[start:end],
            observables=ir.observables if last else (),
            measurements=ir.measurements if last else (),
        )
        title = f"gates {start+1}--{end}" if end else "no gates"
        picture = (
            r"\begin{tikzpicture}[anchor=center,font={"
            + _FONT
            + r"}]\begin{yquant}"
            + _circuit_body(panel)
            + r"\end{yquant}\end{tikzpicture}"
        )
        pictures.append((title, picture))
    rows = []
    for start in range(0, len(pictures), 2):
        pair = pictures[start : start + 2]
        lines = [r"\begingroup"]
        for register, (_, picture) in zip(
            (r"\fhPanelA", r"\fhPanelB"), pair, strict=False
        ):
            lines += [
                r"\setbox" + register + r"=\hbox{" + picture + "}",
                _width_guard(register),
            ]
        if len(pair) == 1:
            lines.append(pair[0][0] + r"\par\vspace{3pt}\noindent\copy\fhPanelA\par")
        else:
            lines += [
                r"\fhPanelWide=0pt\ifdim\wd\fhPanelA>86mm\fhPanelWide=1pt\fi\ifdim\wd\fhPanelB>86mm\fhPanelWide=1pt\fi",
                r"\ifdim\fhPanelWide=0pt",
                r"\begin{minipage}[t]{88mm}\centering "
                + pair[0][0]
                + r"\par\vspace{3pt}\noindent\copy\fhPanelA\par\end{minipage}\hfill",
                r"\begin{minipage}[t]{88mm}\centering "
                + pair[1][0]
                + r"\par\vspace{3pt}\noindent\copy\fhPanelB\par\end{minipage}",
                r"\else "
                + pair[0][0]
                + r"\par\vspace{3pt}\noindent\copy\fhPanelA\par\vspace{8pt}"
                + pair[1][0]
                + r"\par\vspace{3pt}\noindent\copy\fhPanelB\par\fi",
            ]
        rows.append("\n".join([*lines, r"\endgroup"]))
    return (
        rf"\begin{{minipage}}{{{_CANVAS_MM}mm}}"
        + _FONT
        + r"\textbf{b}\quad Quantum circuit\par\vspace{5pt}\centering"
        + r"\par\vspace{8pt}".join(rows)
        + r"\par\end{minipage}"
    )
