"""Explicit hybrid architecture, shape continuity and readable circuit panels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Literal

from .hybrid_layers import ClassicalLayer, _positive_count
from .latex_drawer import _as_ir, _circuit_body, _escape_text
from .latex_figure import LatexFigure, _document


def _layers(
    stage: str | Sequence[ClassicalLayer], name: str
) -> tuple[ClassicalLayer, ...]:
    if isinstance(stage, str):
        if not stage.strip():
            raise ValueError(f"{name} label cannot be empty")
        return ()
    layers = tuple(stage)
    if not layers or any(not isinstance(layer, ClassicalLayer) for layer in layers):
        raise TypeError(
            f"{name} must be a label or a nonempty sequence of ClassicalLayer"
        )
    for previous, current in pairwise(layers):
        if previous.output_shape != current.input_shape:
            raise ValueError(
                f"{name} shape mismatch: {previous.label} outputs {previous.output_shape}, "
                f"but {current.label} expects {current.input_shape}"
            )
    return layers


def _vector_count(
    layers: tuple[ClassicalLayer, ...], count: int | None, name: str
) -> int | None:
    if count is not None:
        count = _positive_count(count, name)
    if layers:
        shape = layers[-1].output_shape
        if len(shape) != 1:
            raise ValueError(f"{name} requires a final vector, got {shape}")
        if count is not None and count != shape[0]:
            raise ValueError(f"{name}={count} conflicts with declared output {shape}")
        count = shape[0]
    return count


def _shape_tex(shape: tuple[int, ...]) -> str:
    return r" $\times$ ".join(str(n) for n in shape)


def _stage_tex(
    label: str,
    layers: tuple[ClassicalLayer, ...],
    trainable: bool,
    count: int | None,
    unit: str,
) -> str:
    if not layers:
        description = "unspecified " + unit if count is None else f"{count} {unit}"
        return (
            r"\begin{tikzpicture}\fhPNNLayer[height=14,width=2,depth=10]{volume}{(0,0,0)}{"
            + _escape_text(label)
            + "}{"
            + description
            + r"}\end{tikzpicture}"
        )
    lines = [r"\begin{tikzpicture}[font=\sffamily\small]"]
    for i, layer in enumerate(layers):
        position = "" if i == 0 else f",below=5mm of layer{i-1}"
        state = "trainable" if trainable and layer.trainable else "fixed"
        lines.append(
            r"\node[draw=fhClassical!60,fill=white,rounded corners=2pt,"
            r"text width=3.3cm,align=center,inner sep=5pt"
            + position
            + f"] (layer{i}) {{"
            + r"\textbf{"
            + _escape_text(layer.label)
            + r"}\\[3pt]{\fontsize{8}{10}\selectfont "
            + _shape_tex(layer.input_shape)
            + r" $\rightarrow$ "
            + _shape_tex(layer.output_shape)
            + r"}\\[-1pt]{\fontsize{8}{10}\selectfont "
            + state
            + "}};"
        )
        if i:
            lines.append(rf"\draw[fh/data] (layer{i-1}.south) -- (layer{i}.north);")
    return "\n".join([*lines, r"\end{tikzpicture}"])


def _architecture(
    encoder: str,
    head: str,
    quantum: str,
    training: bool,
    trainable: tuple[bool, bool, bool],
    declared_readout: bool,
) -> str:
    lines = [
        r"\begin{fhdiagram}[fh/label/.append style={font=\sffamily\fontsize{8}{10}\selectfont}]",
        r"\node[fh/block,inner sep=10pt] (encoder) {\begin{tabular}{@{}c@{}}\textbf{Classical encoder}\\[7pt]"
        + encoder
        + r"\end{tabular}};",
        r"\node[fh/quantum,right=14mm of encoder,inner sep=10pt] (quantum) {"
        + quantum
        + "};",
        r"\node[fh/block,right=14mm of quantum,inner sep=10pt] (head) {\begin{tabular}{@{}c@{}}\textbf{Classical head}\\[7pt]"
        + head
        + r"\end{tabular}};",
        r"\fhConnect{encoder.east}{quantum.west}{features}",
        r"\fhConnect"
        + ("[fh/measurement]" if declared_readout else "")
        + r"{quantum.east}{head.west}{readout}",
    ]
    if training:
        lines += [
            r"\node[fh/block,fh/optimizer,below=14mm of head] (optimizer)"
            r" {\textbf{Loss + optimizer}\\[3pt]{\fontsize{8}{10}\selectfont "
            + ("Trainable parameters only" if any(trainable) else "No trainable stages")
            + "}};",
            r"\draw[fh/data] (head.south) -- node[fh/label,midway,right]{predictions} (optimizer.north);",
            r"\coordinate (updates) at ([yshift=-10mm]current bounding box.south);",
        ]
        if trainable[0] or trainable[1]:
            leftmost = "encoder" if trainable[0] else "quantum"
            lines.append(
                r"\draw[fh/feedback,draw=fhInk,line width=.7pt]"
                r" (optimizer.south) -- (optimizer.south |- updates) -- "
                + f"({leftmost}.south |- updates);"
            )
        for name, enabled in zip(
            ("encoder", "quantum", "head"), trainable, strict=True
        ):
            if not enabled:
                continue
            if name == "head":
                lines.append(
                    r"\draw[fh/data,fh/feedback] (optimizer.east) -- ++(12mm,0)"
                    r" coordinate (headfeedback) |- (head.east);"
                    r"\node[fh/label,rotate=90,anchor=north] at"
                    r" ($(headfeedback)!.5!(headfeedback |- head.east)$) {head update};"
                )
            else:
                lines.append(
                    rf"\draw[fh/data,fh/feedback] ({name}.south |- updates)"
                    + rf" -- node[fh/label,midway,right]{{{name} update}} ({name}.south);"
                )
    lines += [
        r"\node[anchor=north,font=\sffamily\fontsize{8}{10}\selectfont,align=center]"
        r" at ([yshift=-6mm]current bounding box.south)"
        r" {Blue: classical\quad Purple: quantum\quad Shapes exclude batch\\[2pt]"
        r" Solid: data\quad Double: declared readout\quad Dashed: parameter updates};",
        r"\node[anchor=south west,font=\sffamily\large\bfseries,text=fhInk]"
        r" at ([yshift=6mm]current bounding box.north west) {Quantum--classical architecture};",
        r"\end{fhdiagram}",
    ]
    return "\n".join(lines)


def draw_hybrid(
    program: object,
    *,
    encoder: str | Sequence[ClassicalLayer] = "CNN encoder",
    head: str | Sequence[ClassicalLayer] = "Dense head",
    encoder_features: int | None = None,
    head_outputs: int | None = None,
    training: bool = False,
    standalone: bool = True,
    display: bool = False,
    engine: str | None = None,
    timeout: float = 120.0,
    readout: str | None = None,
    readout_features: int | None = None,
    encoder_trainable: bool = True,
    quantum_trainable: bool = True,
    head_trainable: bool = True,
    quantum_view: Literal["auto", "overview", "detail"] = "auto",
    gates_per_panel: int = 8,
) -> LatexFigure:
    """Draw explicit classical layers around an authoritative native circuit.

    Args:
        program: Native circuit, CircuitIR or its serialized mapping.
        encoder: Label or ordered classical layers before the quantum circuit.
        head: Label or ordered classical layers after quantum readout.
        encoder_features: Encoder vector size; derived from detailed layers.
        head_outputs: Head vector size; derived from detailed layers.
        training: Show loss/optimizer and updates to trainable stages only.
        standalone: Include a complete LaTeX document.
        display: Immediately display through IPython.
        engine: Optional rendering executable name/path.
        timeout: Maximum seconds per rendering subprocess.
        readout: Explicit readout description; None means unspecified.
        readout_features: Declared readout vector size, required for detailed heads.
        encoder_trainable: Enable encoder parameter updates; False freezes the stage.
        quantum_trainable: Whether to draw quantum parameter updates.
        head_trainable: Enable head parameter updates; False freezes the stage.
        quantum_view: Auto inlines short circuits and panels larger ones; overview
            omits details; detail always adds ordered circuit panels.
        gates_per_panel: Maximum instructions in each continuation panel.

    Returns:
        An editable, lazily compiled LatexFigure. Shapes exclude the batch axis.

    Raises:
        ValueError: Shapes/counts conflict or a circuit operation is unsupported.
        TypeError: A stage or trainability flag has an invalid type.

    This checks declared shapes, not nn.Module operation formulas or gradients.
    It never traces or executes a model. Readout labels are caller declarations.
    No assumed counts, measurement operations or parameter updates are invented.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.drawer import ClassicalLayer, draw_hybrid
        >>> head = [ClassicalLayer("Linear", (2,), (1,), trainable=True)]
        >>> figure = draw_hybrid(fq.Circuit(2).h(0), head=head,
        ...                      readout="Z expectations", readout_features=2,
        ...                      training=True, encoder_trainable=False)
        >>> "Loss + optimizer" in figure.source and "encoder update" not in figure.source
        True
    """
    for name, value in (
        ("training", training),
        ("encoder_trainable", encoder_trainable),
        ("quantum_trainable", quantum_trainable),
        ("head_trainable", head_trainable),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    if quantum_view not in ("auto", "overview", "detail"):
        raise ValueError("quantum_view must be 'auto', 'overview' or 'detail'")
    panel_size = _positive_count(gates_per_panel, "gates_per_panel")
    encoder_layers, head_layers = _layers(encoder, "encoder"), _layers(head, "head")
    features = _vector_count(encoder_layers, encoder_features, "encoder_features")
    outputs = _vector_count(head_layers, head_outputs, "head_outputs")
    if readout is not None and (not isinstance(readout, str) or not readout.strip()):
        raise ValueError("readout must be a nonempty plain-text label")
    if readout_features is not None:
        readout_features = _positive_count(readout_features, "readout_features")
        if readout is None:
            raise ValueError("readout_features requires an explicit readout label")
    if head_layers:
        if readout is None or readout_features is None:
            raise ValueError("detailed head requires readout and readout_features")
        if head_layers[0].input_shape != (readout_features,):
            raise ValueError("head input shape conflicts with readout_features")
    ir = _as_ir(program)
    circuit = _circuit_body(ir)  # Validate all operations, even in overview mode.
    quantum_updates = quantum_trainable and any(n.params for n in ir.instructions)
    readout_label = "Readout unspecified" if readout is None else _escape_text(readout)
    if readout_features is not None:
        readout_label += f" / {readout_features} values"
    inline = quantum_view == "auto" and len(ir.instructions) <= 6 and ir.n_wires <= 4
    quantum = (
        r"\begin{tabular}{@{}c@{}}\textbf{Quantum circuit}\\[6pt]"
        + (
            r"\begin{tikzpicture}[font=\sffamily\small]\begin{yquant}"
            + circuit
            + r"\end{yquant}\end{tikzpicture}"
            if inline
            else f"{ir.n_wires} qubits / {len(ir.instructions)} gates"
        )
        + r"\\[4pt]{\fontsize{8}{10}\selectfont "
        + ("Trainable parameters declared" if quantum_updates else "Fixed circuit")
        + r"}\\[6pt]\parbox{3.5cm}{\centering\fontsize{8}{10}\selectfont "
        + readout_label
        + r"}\end{tabular}"
    )
    trainable = (
        encoder_trainable
        and (not encoder_layers or any(n.trainable for n in encoder_layers)),
        quantum_updates,
        head_trainable and (not head_layers or any(n.trainable for n in head_layers)),
    )
    architecture = _architecture(
        _stage_tex(
            str(encoder) if not encoder_layers else "",
            encoder_layers,
            encoder_trainable,
            features,
            "features",
        ),
        _stage_tex(
            str(head) if not head_layers else "",
            head_layers,
            head_trainable,
            outputs,
            "outputs",
        ),
        quantum,
        training,
        trainable,
        readout is not None,
    )
    rows = [architecture]
    if quantum_view != "overview" and not inline:
        for start in range(0, max(1, len(ir.instructions)), panel_size):
            end = min(start + panel_size, len(ir.instructions))
            last = end == len(ir.instructions)
            panel = replace(
                ir,
                instructions=ir.instructions[start:end],
                observables=ir.observables if last else (),
                measurements=ir.measurements if last else (),
            )
            heading = (
                f"Circuit detail / gates {start + 1}--{end}"
                if end
                else "Circuit detail / no gates"
            )
            rows.append(
                r"\begin{tabular}{@{}c@{}}\textbf{" + heading + r"}\\[3pt]"
                r"{\fontsize{8}{10}\selectfont Same registers continue between panels; no reset.}\\[7pt]"
                r"\begin{tikzpicture}[font=\sffamily\small]\begin{yquant}"
                + _circuit_body(panel)
                + r"\end{yquant}\end{tikzpicture}\end{tabular}"
            )
    body = (
        r"{\sffamily\begin{tabular}{@{}c@{}}"
        + "\n"
        + (r"\\[24pt]" + "\n").join(rows)
        + "\n"
        + r"\end{tabular}}"
        + "\n"
    )
    figure = LatexFigure(
        _document(body) if standalone else body, standalone, engine, timeout
    )
    if display:
        figure.display()
    return figure
