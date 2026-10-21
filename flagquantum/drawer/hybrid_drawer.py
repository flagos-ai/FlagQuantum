"""Explicit encoder/circuit/head architecture figures, without model tracing."""

from __future__ import annotations

import operator

from .latex_drawer import _circuit_body, _escape_text
from .latex_figure import LatexFigure, _document


def _feature_count(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    count = operator.index(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def draw_hybrid(
    program: object,
    *,
    encoder: str = "CNN encoder",
    head: str = "Dense head",
    encoder_features: int = 32,
    head_outputs: int = 10,
    training: bool = False,
    standalone: bool = True,
    display: bool = False,
    engine: str | None = None,
    timeout: float = 120.0,
) -> LatexFigure:
    """Draw an explicitly described classical/quantum/classical model.

    The circuit is authoritative; classical labels and counts are supplied by
    the caller. This does not trace or execute an nn.Module. Training adds an
    illustrative loss/optimizer and parameter-update paths for all three stages.

    Args:
        program: Native circuit or CircuitIR embedded in the quantum stage.
        encoder: Plain-text classical encoder label.
        head: Plain-text classical head label.
        encoder_features: Number of features emitted by the encoder.
        head_outputs: Number of outputs emitted by the head.
        training: Include loss, optimizer and illustrative parameter updates.
        standalone: Include a complete LaTeX document.
        display: Immediately display through IPython.
        engine: Optional rendering executable name/path.
        timeout: Maximum seconds per rendering subprocess.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.drawer import draw_hybrid
        >>> figure = draw_hybrid(fq.Circuit(2).h(0).cx(0, 1), training=True)
        >>> "Loss + optimizer" in figure.source
        True
    """
    features = _feature_count(encoder_features, "encoder_features")
    outputs = _feature_count(head_outputs, "head_outputs")
    if not isinstance(training, bool):
        raise TypeError("training must be a bool")
    circuit = _circuit_body(program)
    lines = [
        r"\begin{fhdiagram}[fh/label baseline=-1.8]",
        r"\node[anchor=west,font=\sffamily\Large\bfseries,text=fhInk] at (-1.5,3.3) {Quantum--classical model};",
        r"\node[anchor=west,font=\sffamily\small,text=fhInk] at (-1.5,2.6) {Explicit architecture / circuit detail / editable source};",
        r"\fhPNNLayer[height=18,width=2,depth=16]{encoder}{(0,0,0)}"
        + "{"
        + _escape_text(encoder)
        + "}{"
        + str(features)
        + " features}",
        r"\fhQuantumBlock{quantum}{(5.5,0)}{Quantum layer}{" + "\n" + circuit + "}",
        r"\fhPNNLayer[height=15,width=2,depth=3]{head}{(11,0,0)}"
        + "{"
        + _escape_text(head)
        + "}{"
        + str(outputs)
        + " outputs}",
        r"\fhConnect{encoder-east}{quantum}{encoded features}",
        r"\fhConnect[fh/measurement]{quantum}{head-west}{readout}",
    ]
    if training:
        lines += [
            r"\fhBlock[fh/optimizer]{optimizer}{(5.5,-3.2)}{Loss + optimizer}{Joint parameter update}",
            r"\draw[fh/data,rounded corners=5pt] (head-east) -- (12.9,0) |- node[fh/label,pos=.8,above]{predictions} (optimizer.east);",
            r"\draw[fh/data,fh/feedback] (optimizer.north) -- node[fh/label,midway,right]{quantum update} (quantum.south);",
            r"\draw[fh/data,fh/feedback] (optimizer.west) -- node[fh/label,midway,above]{encoder update} (-1.5,-3.2) |- (encoder-west);",
            r"\draw[fh/data,fh/feedback] (optimizer.south) -- (5.5,-4.35) -- node[fh/label,midway,above]{head update} (14,-4.35) -- (14,1.75) -| (head-north);",
            r"\fhLegend{(6,-5.4)}",
            r"\path (-1.7,-5.9) rectangle (14.2,3.65);",
        ]
    else:
        lines += [r"\fhLegend{(5.5,-3.1)}", r"\path (-1.7,-3.7) rectangle (12.5,3.65);"]
    lines.append(r"\end{fhdiagram}")
    body = "\n".join(lines) + "\n"
    figure = LatexFigure(
        _document(body) if standalone else body, standalone, engine, timeout
    )
    if display:
        figure.display()
    return figure
