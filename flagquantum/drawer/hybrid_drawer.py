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
            r"\parbox{38mm}{\centering\textbf{"
            + _escape_text(label)
            + r"}\\[4pt]{"
            + description
            + "}}"
        )
    lines = [r"\begin{tikzpicture}[font=\sffamily\fontsize{9}{11}\selectfont]"]
    for i, layer in enumerate(layers):
        position = "" if i == 0 else f",below=3mm of layer{i-1}"
        state = r"\,\textsuperscript{*}" if trainable and layer.trainable else ""
        lines.append(
            r"\node[draw=fhClassical!70,fill=white,rounded corners=1pt,"
            r"text width=38mm,align=center,inner sep=3pt"
            + position
            + f"] (layer{i}) {{"
            + r"\textbf{"
            + _escape_text(layer.label)
            + "}"
            + state
            + r"\\[2pt]"
            + _shape_tex(layer.input_shape)
            + r" $\rightarrow$ "
            + _shape_tex(layer.output_shape)
            + "};"
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
        r"\begin{fhdiagram}[font=\sffamily\fontsize{9}{11}\selectfont,"
        r"fh/label/.append style={font=\sffamily\fontsize{9}{11}\selectfont}]",
        r"\node[fh/block,inner sep=5pt] (encoder) {\begin{tabular}{@{}c@{}}\textbf{Classical encoder}\\[5pt]"
        + encoder
        + r"\end{tabular}};",
        r"\node[fh/quantum,right=10mm of encoder,inner sep=5pt] (quantum) {"
        + quantum
        + "};",
        r"\node[fh/block,right=10mm of quantum,inner sep=5pt] (head) {\begin{tabular}{@{}c@{}}\textbf{Classical head}\\[5pt]"
        + head
        + r"\end{tabular}};",
        r"\fhConnect{encoder.east}{quantum.west}{$x$}",
        r"\fhConnect"
        + ("[fh/measurement]" if declared_readout else "")
        + r"{quantum.east}{head.west}{$z$}",
    ]
    if training:
        lines += [
            r"\node[fh/block,fh/optimizer,below=8mm of head,inner sep=5pt] (optimizer)"
            r" {Loss + optimizer"
            + ("" if any(trainable) else r"\\No trainable stages")
            + "};",
            r"\draw[fh/data] (head.south) -- node[fh/label,midway,right]{predictions} (optimizer.north);",
            r"\coordinate (updates) at ([yshift=-7mm]current bounding box.south);",
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
                    r"\coordinate (headfeedback) at ([xshift=8mm]head.east);"
                    r"\draw[fh/data,fh/feedback] (optimizer.east)"
                    r" -| (headfeedback |- optimizer.east) -- (headfeedback) -- (head.east);"
                    r"\node[fh/label,rotate=90,anchor=south] at"
                    r" ($(headfeedback)!.5!(headfeedback |- optimizer.east)$) {head update};"
                )
            else:
                lines.append(
                    rf"\draw[fh/data,fh/feedback] ({name}.south |- updates)"
                    + rf" -- node[fh/label,midway,right]{{{name} update}} ({name}.south);"
                )
    lines.append(r"\end{fhdiagram}")
    return "\n".join(lines)


def _paper_panel(content: str, heading: str) -> str:
    # Measure at the intended font size; never shrink an oversized circuit.
    return (
        r"\begin{minipage}{166mm}\sffamily\fontsize{9}{11}\selectfont"
        + r"\textbf{"
        + heading
        + r"}\par\vspace{5pt}\centering"
        + r"\setbox0=\hbox{"
        + content
        + r"}\ifdim\wd0>166mm"
        + r"\PackageError{flagquantum}{Figure panel exceeds 166 mm. "
        r"Reduce gates_per_panel, shorten labels or use quantum_view=overview}"
        r"{The drawer preserves publication font sizes instead of scaling.}\fi"
        + r"\noindent\box0\par\end{minipage}"
    )


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
        quantum_view: Auto and detail separate architecture and ordered circuit
            panels at publication font size; overview omits circuit detail.
        gates_per_panel: Maximum instructions in each continuation panel.

    Returns:
        An editable, lazily compiled LatexFigure for a 170 mm paper text area.
        Body text is 9 pt; shapes exclude the batch axis. A star marks trainable
        classical layers; dashed arrows show eligible parameter updates.

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
    _circuit_body(ir)  # Validate all operations, even in overview mode.
    quantum_updates = quantum_trainable and any(n.params for n in ir.instructions)
    readout_label = "Readout unspecified" if readout is None else _escape_text(readout)
    if readout_features is not None:
        readout_label += f" / {readout_features} values"
    quantum = (
        r"\begin{tabular}{@{}c@{}}\textbf{Quantum circuit}"
        + (r"\,\textsuperscript{*}" if quantum_updates else "")
        + r"\\[5pt]"
        + f"{ir.n_wires} qubits / {len(ir.instructions)} gates"
        + r"\\[5pt]\parbox{36mm}{\centering "
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
    rows = [_paper_panel(architecture, "(a) Hybrid model")]
    if quantum_view != "overview":
        for index, start in enumerate(
            range(0, max(1, len(ir.instructions)), panel_size)
        ):
            end = min(start + panel_size, len(ir.instructions))
            last = end == len(ir.instructions)
            panel = replace(
                ir,
                instructions=ir.instructions[start:end],
                observables=ir.observables if last else (),
                measurements=ir.measurements if last else (),
            )
            heading = "(b) Quantum circuit" if index == 0 else "(b) Continued"
            if len(ir.instructions) > panel_size:
                heading += f" / gates {start + 1}--{end}"
            elif not end:
                heading += " / no gates"
            rows.append(
                _paper_panel(
                    r"\begin{tikzpicture}[font=\sffamily\fontsize{9}{11}\selectfont]"
                    r"\begin{yquant}"
                    + _circuit_body(panel)
                    + r"\end{yquant}\end{tikzpicture}",
                    heading,
                )
            )
    body = (
        r"\begin{minipage}{166mm}\sffamily\noindent"
        + "\n"
        + (r"\par\vspace{14pt}\noindent" + "\n").join(rows)
        + "\n"
        + r"\par\end{minipage}"
        + "\n"
    )
    figure = LatexFigure(
        _document(body) if standalone else body, standalone, engine, timeout
    )
    if display:
        figure.display()
    return figure
