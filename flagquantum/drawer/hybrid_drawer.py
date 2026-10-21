"""Explicit hybrid architecture, shape continuity and readable circuit panels."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

from ._hybrid_layout import (
    _architecture,
    _circuit_grid,
    _paper_panel,
    _quantum_tex,
    _readout_tex,
    _stage_tex,
    _width_guard,
)
from .hybrid_layers import ClassicalLayer, _positive_count
from .latex_drawer import _as_ir, _circuit_body
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
        quantum_view: Auto inlines up to eight gates on four qubits; larger circuits
            use measured continuation rows. Detail forces panels; overview omits them.
        gates_per_panel: Maximum instructions in each continuation panel.

    Returns:
        An editable, lazily compiled LatexFigure for a 183 mm paper text area.
        Helvetica body text is 7 pt; shapes exclude the batch axis. A star marks trainable
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
    inline = quantum_view == "auto" and len(ir.instructions) <= 8 and ir.n_wires <= 4
    quantum = _quantum_tex(ir, inline)
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
            include_input=True,
        ),
        _stage_tex(
            str(head) if not head_layers else "",
            head_layers,
            head_trainable,
            outputs,
            "outputs",
            include_input=False,
        ),
        quantum,
        _readout_tex(readout, readout_features),
        f"{ir.n_wires} {'qubit' if ir.n_wires == 1 else 'qubits'} / "
        f"{len(ir.instructions)} {'gate' if len(ir.instructions) == 1 else 'gates'}",
        training,
        trainable,
        readout is not None,
    )
    rows = [
        _paper_panel(
            architecture, "a" if quantum_view != "overview" and not inline else ""
        )
    ]
    if quantum_view != "overview" and not inline:
        rows.append(_circuit_grid(ir, panel_size))
    body = (
        r"\begin{minipage}{181.5mm}\noindent"
        + (r"\par\vspace{10pt}\noindent" + "\n").join(rows)
        + r"\par\end{minipage}"
        + "\n"
    )
    body = (
        r"\setbox\fhFigure=\hbox{"
        + body.rstrip()
        + "}"
        + _width_guard(r"\fhFigure")
        + r"\ifdim\dimexpr\ht\fhFigure+\dp\fhFigure\relax>168.5mm"
        r"\PackageError{flagquantum}{Figure exceeds 170 mm height. Use overview or split circuit detail}{Text is never shrunk to fit.}\fi\box\fhFigure"
        + "\n"
    )
    figure = LatexFigure(
        _document(body) if standalone else body, standalone, engine, timeout
    )
    if display:
        figure.display()
    return figure
