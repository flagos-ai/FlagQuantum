"""CircuitIR-to-yquant source generation, independent of optional render tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING

import torch

from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import get_operator_schema
from ..core.parameters import Parameter, ParameterExpression
from .latex_figure import LatexFigure, _document

if TYPE_CHECKING:
    from .register_tex import RegisterLayout


def _escape_text(value: object) -> str:
    escapes = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(character, character) for character in str(value))


def _parameter_tex(value: object) -> str:
    if isinstance(value, Parameter):
        match = re.fullmatch(r"(theta|phi|alpha|beta|gamma|x)(?:_(\d+))?", value.name)
        if match:
            base, index = match.groups()
            label = base if base == "x" else "\\" + base
            return label if index is None else label + "_{" + index + "}"
        return r"\mbox{" + _escape_text(value.name) + "}"
    if isinstance(value, ParameterExpression):
        if value.op == "neg" and len(value.args) == 1:
            return "(-" + _parameter_tex(value.args[0]) + ")"
        if value.op in ("add", "sub", "mul") and len(value.args) == 2:
            symbol = {"add": "+", "sub": "-", "mul": r"\cdot "}[value.op]
            return (
                "("
                + _parameter_tex(value.args[0])
                + symbol
                + _parameter_tex(value.args[1])
                + ")"
            )
        raise ValueError(f"Unsupported parameter expression: {value.op}")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                "LaTeX drawing requires scalar gate parameters; draw a symbolic circuit for batched models"
            )
        return _parameter_tex(value.detach().cpu().item())
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    raise ValueError(f"Unsupported gate parameter type: {type(value).__name__}")


def _as_ir(program: object) -> CircuitIR:
    if isinstance(program, CircuitIR):
        return program
    if isinstance(program, Mapping):
        return CircuitIR.from_dict(program)
    method = getattr(program, "to_ir", None)
    if callable(method):
        ir = method()
        if isinstance(ir, CircuitIR):
            return ir
    raise TypeError(
        "LaTeX drawing expects a FlagQuantum Circuit, CircuitIR or its serialized mapping"
    )


def _gate(instruction: Instruction) -> str:
    name = instruction.name
    schema = get_operator_schema(name)
    metadata = instruction.metadata
    if (
        schema is None
        or not schema.unitary
        or instruction.matrix is not None
        or any(
            metadata.get(key)
            for key in ("is_channel", "is_dynamic", "inverse", "adjoint")
        )
        or any(
            key in metadata
            for key in ("condition", "conditions", "condition_clauses", "controls")
        )
    ):
        raise ValueError(
            f"LaTeX drawing does not support gate {name!r} with these semantics"
        )
    qubits = [f"q[{index}]" for index in instruction.wires]
    if name in ("h", "x", "y", "z"):
        return f"{name} {qubits[0]};"
    if name in ("cx", "ccx"):
        return f"cnot {qubits[-1]} | {', '.join(qubits[:-1])};"
    if name in ("cy", "cz"):
        return f"{name[-1]} {qubits[1]} | {qubits[0]};"
    if name in ("swap", "cswap"):
        targets = qubits if name == "swap" else qubits[1:]
        control = "" if name == "swap" else f" | {qubits[0]}"
        return f"swap ({', '.join(targets)}){control};"
    controlled = name in ("ch", "csx", "csxdg", "crx", "cry", "crz", "cphase")
    base = name[1:] if controlled else name
    names = {
        "i": "I",
        "h": "H",
        "iswap": r"\mathrm{iSWAP}",
        "iswapdg": r"\mathrm{iSWAP}^{\dagger}",
        "dcx": r"\mathrm{DCX}",
        "ecr": r"\mathrm{ECR}",
        "s": "S",
        "t": "T",
        "sdg": r"S^{\dagger}",
        "tdg": r"T^{\dagger}",
        "sx": r"\sqrt{X}",
        "sxdg": r"\sqrt{X}^{\dagger}",
        "rx": "R_x",
        "ry": "R_y",
        "rz": "R_z",
        "phase": "P",
        "rxx": "R_{xx}",
        "ryy": "R_{yy}",
        "rzz": "R_{zz}",
        "u1": "U_1",
        "u2": "U_2",
        "u3": "U_3",
    }
    if base not in names:
        raise ValueError(f"LaTeX drawing has no rendering rule for gate {name!r}")
    label = names[base]
    if schema.parameters:
        label += (
            "("
            + ",".join(
                _parameter_tex(instruction.params[key]) for key in schema.parameters
            )
            + ")"
        )
    targets = qubits[1:] if controlled else qubits
    target = targets[0] if len(targets) == 1 else "(" + ", ".join(targets) + ")"
    control = f" | {qubits[0]}" if controlled else ""
    return "box {$" + label + "$} " + target + control + ";"


def _circuit_body(program: object) -> str:
    ir = _as_ir(program)
    lines = [rf"qubit {{$q_{{\idx}}$}} q[{ir.n_wires}];"]
    lines.extend(_gate(instruction) for instruction in ir.instructions)
    for observable in ir.observables:
        label = "Observable: " + _escape_text(observable.name) + ", coefficient="
        label += _escape_text(observable.coefficient)
        targets = ", ".join(f"q[{index}]" for index in observable.wires)
        lines.append("text {" + label + "} (" + targets + ");")
    for measurement in ir.measurements:
        label = "Readout: " + _escape_text(measurement.kind)
        if measurement.shots is not None:
            label += f", shots={measurement.shots}"
        targets = ", ".join(f"q[{index}]" for index in measurement.wires)
        lines.append("text {" + label + "} (" + targets + ");")
    return "\n".join(lines) + "\n"


def draw_latex(
    program: object,
    *,
    layout: RegisterLayout | None = None,
    comparison: tuple[object, RegisterLayout] | None = None,
    row_spacing: float = 4.5,
    inputs: Mapping[int, str] | None = None,
    outputs: Mapping[int, str] | None = None,
    standalone: bool = True,
    display: bool = False,
    engine: str | None = None,
    timeout: float = 120.0,
) -> LatexFigure:
    """Export a Circuit or CircuitIR without executing or compiling it.

    Args:
        program: A native circuit, IR, or serialized IR mapping.
        layout: Register geometry and symbols for named operations.
        comparison: Optional second circuit and layout, drawn to the right.
        row_spacing: Distance between register rows in millimetres.
        inputs: Trusted LaTeX labels keyed by input qubit.
        outputs: Trusted LaTeX labels keyed by output qubit.
        standalone: Include a complete LaTeX document rather than a fragment.
        display: Immediately display through IPython, compiling on demand.
        engine: Optional executable name/path for rendering later.
        timeout: Maximum seconds per rendering subprocess.

    Returns:
        A lazy figure exposing editable source and Notebook display.

    Raises:
        ValueError: An operation cannot be represented without semantic loss.
        TypeError: The input is not a supported circuit or IR.

    Examples:
        >>> import flagquantum as fq
        >>> figure = fq.Circuit(1).h(0).draw(format="latex")
        >>> "h q[0];" in figure.source
        True
    """
    ir = _as_ir(program)
    if comparison is not None and layout is None:
        raise ValueError("A comparison requires a layout for the first circuit.")
    if (
        layout is not None
        or inputs is not None
        or outputs is not None
        or any(
            instruction.name == "measure"
            or instruction.metadata.get("conditions")
            or instruction.metadata.get("quantum_controls")
            for instruction in ir.instructions
        )
    ):
        from .paper_tex import paper_circuit_tex

        if ir.observables or ir.measurements:
            raise ValueError(
                "Paper layout requires readout to be represented as circuit instructions."
            )
        if layout is None:
            body = paper_circuit_tex(ir, inputs or {}, outputs or {})
        else:
            from .register_tex import register_circuit_tex

            body = register_circuit_tex(ir, layout, comparison, row_spacing=row_spacing)
        source = (
            "\\documentclass[border=2pt]{standalone}\n"
            "\\usepackage{tikz}\n\\begin{document}\n" + body + "\\end{document}\n"
            if standalone
            else body
        )
    else:
        body = "\\begin{tikzpicture}\n\\begin{yquant}\n" + _circuit_body(ir)
        body += "\\end{yquant}\n\\end{tikzpicture}\n"
        source = _document(body) if standalone else body
    figure = LatexFigure(source, standalone, engine, timeout)
    if display:
        figure.display()
    return figure
