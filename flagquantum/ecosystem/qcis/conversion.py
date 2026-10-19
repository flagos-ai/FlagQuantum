"""Parse the supported QCIS gate subset without executing external tools."""

from __future__ import annotations

import math
import re
from typing import NoReturn

from ...core.ir import CircuitIR, Instruction, MeasurementNode
from ..contracts import (
    InteropConversionError,
    InteropConversionIssue,
    InteropConversionReport,
    InteropImportResult,
)
from ._expressions import Angle, parse_angle

_FIXED = {
    "X": "x",
    "Y": "y",
    "Z": "z",
    "H": "h",
    "S": "s",
    "SD": "sdg",
    "T": "t",
    "TD": "tdg",
    "CZ": "cz",
}
_HALF = {
    "X2P": ("rx", math.pi / 2),
    "X2M": ("rx", -math.pi / 2),
    "Y2P": ("ry", math.pi / 2),
    "Y2M": ("ry", -math.pi / 2),
}
_ROTATIONS = {"RX": "rx", "RY": "ry", "RZ": "rz"}
_QUBIT = re.compile(r"Q(?:0|[1-9][0-9]*)\Z", re.ASCII)


def _report(issues: list[InteropConversionIssue]) -> InteropConversionReport:
    return InteropConversionReport("qcis", "from_qcis", None, tuple(issues))


def _refuse(code: str, message: str, *, line: int, name: str) -> NoReturn:
    issue = InteropConversionIssue(
        code,
        f"QCIS line {line} ({name}): {message}",
        "error",
        operation_name=name,
    )
    raise InteropConversionError(issue.message, _report([issue]))


def _qubits(tokens: list[str], *, line: int, name: str) -> tuple[int, ...]:
    if not tokens or any(_QUBIT.fullmatch(token) is None for token in tokens):
        _refuse(
            "invalid_qubit",
            "expected Q followed by a non-negative integer",
            line=line,
            name=name,
        )
    qubits = tuple(
        _integer(token[1:], operand=token, code="invalid_qubit", line=line, name=name)
        for token in tokens
    )
    if len(set(qubits)) != len(qubits):
        _refuse(
            "duplicate_qubit",
            "an instruction cannot repeat a qubit",
            line=line,
            name=name,
        )
    return qubits


def _integer(token: str, *, operand: str, code: str, line: int, name: str) -> int:
    try:
        return int(token)
    except ValueError:
        _refuse(
            code,
            f"integer operand {operand[:48]!r} ({len(operand)} characters) exceeds the supported conversion limit",
            line=line,
            name=name,
        )


def _angle(token: str, *, line: int, name: str) -> Angle:
    try:
        return parse_angle(token)
    except ValueError as exc:
        _refuse(
            "invalid_parameter",
            f"angle operand {token[:80]!r}: {exc}",
            line=line,
            name=name,
        )


def _operands(text: str, *, line: int, name: str) -> list[str]:
    # Parenthesized angles may contain spaces; top-level whitespace separates operands.
    depth = 0
    start = 0
    result: list[str] = []
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth < 0:
            _refuse(
                "invalid_parameter",
                "unmatched closing parenthesis",
                line=line,
                name=name,
            )
        if char.isspace() and depth == 0:
            if start < index:
                result.append(text[start:index])
            start = index + 1
    if depth:
        _refuse("invalid_parameter", "unclosed parenthesis", line=line, name=name)
    if start < len(text):
        result.append(text[start:])
    return result


def import_qcis(source: str, *, allow_lossy: bool = False) -> InteropImportResult:
    """Import numeric or parameterized QCIS gates and ordered terminal Z measurements.

    Hardware labels are compacted in ascending numeric order. The original
    labels and timing annotations remain in ``ir.metadata['interop']``.
    A lossless report covers only the supported ideal circuit semantics,
    not the full QCIS hardware instruction set or device execution semantics.

    Args:
        source: QCIS text, not a filename. Angles accept numeric values or
            names and arithmetic; wrap expressions containing spaces in parentheses.
        allow_lossy: Explicitly permit dropping timing/alignment semantics of
            I and B for ideal simulation. Unsupported operations still fail.

    Returns:
        Canonical complex128 IR and its conversion diagnostics.

    Raises:
        TypeError: An argument has the wrong Python type.
        InteropConversionError: Syntax, gates, measurement order, or timing
            semantics cannot be represented under the requested policy.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.ecosystem.qcis import import_qcis
        >>> imported = import_qcis("X2P Q7\\nRZ Q7 0.3\\nM Q7")
        >>> imported.report.lossless
        True
        >>> fq.run(imported.ir).probabilities.shape
        torch.Size([1, 2])

        Bind symbolic angles before execution using the existing Circuit API:

        >>> imported = import_qcis("RXY Q7 phi (theta/2)")
        >>> circuit = fq.Circuit.from_ir(imported.ir)
        >>> circuit.parameter_names
        ('phi', 'theta')
        >>> bound = circuit.bind_parameters({"phi": 0.2, "theta": 0.6})
        >>> fq.run(bound, outputs=fq.probabilities()).probabilities.shape
        torch.Size([1, 2])
    """
    if not isinstance(source, str):
        raise TypeError("QCIS source must be str")
    if not isinstance(allow_lossy, bool):
        raise TypeError("allow_lossy must be bool")
    instructions: list[Instruction] = []
    issues: list[InteropConversionIssue] = []
    measured: list[int] = []
    labels: set[int] = set()
    timing: list[dict[str, str | int]] = []
    for line, raw in enumerate(source.splitlines(), start=1):
        text = re.split(r"#|//", raw, maxsplit=1)[0].strip()
        if not text:
            continue
        name = text.split(maxsplit=1)[0]
        if name not in {*_FIXED, *_HALF, *_ROTATIONS, "RXY", "M", "I", "B"}:
            _refuse(
                "unsupported_operation",
                "unsupported QCIS instruction",
                line=line,
                name=name,
            )
        operands = _operands(text[len(name) :].strip(), line=line, name=name)
        if name in {"M", "B"}:
            qubits = _qubits(operands, line=line, name=name)
        else:
            expected = (
                3 if name == "RXY" else 2 if name in {*_ROTATIONS, "I", "CZ"} else 1
            )
            if len(operands) != expected:
                _refuse(
                    "invalid_arity",
                    f"expected {expected} operands",
                    line=line,
                    name=name,
                )
            qubits = _qubits(
                operands if name == "CZ" else operands[:1], line=line, name=name
            )
        labels.update(qubits)
        if name == "M":
            if set(measured).intersection(qubits):
                _refuse(
                    "duplicate_measurement",
                    "each qubit may be measured only once",
                    line=line,
                    name=name,
                )
            measured.extend(qubits)
            continue
        if measured and name != "B":
            _refuse(
                "measurement_not_terminal",
                "quantum operations cannot follow measurement",
                line=line,
                name=name,
            )
        if name in {"I", "B"}:
            annotation: dict[str, str | int] = {"line": line, "source": text}
            if name == "I":
                if re.fullmatch(r"[0-9]+", operands[-1], flags=re.ASCII) is None:
                    _refuse(
                        "invalid_duration",
                        "idle duration must be a non-negative integer in 0.5 ns ticks",
                        line=line,
                        name=name,
                    )
                annotation["duration_ticks"] = _integer(
                    operands[-1],
                    operand=operands[-1],
                    code="invalid_duration",
                    line=line,
                    name=name,
                )
            timing.append(annotation)
            issues.append(
                InteropConversionIssue(
                    "timing_not_simulated",
                    f"QCIS line {line} ({name}): ideal simulation omits timing/alignment of {text!r}",
                    "warning",
                    operation_name=name,
                )
            )
            if name == "I":
                instructions.append(Instruction("i", qubits))
        elif name in _HALF:
            opcode, half_theta = _HALF[name]
            instructions.append(
                Instruction(opcode, qubits, params={"theta": half_theta})
            )
        elif name == "RXY":
            phi = _angle(operands[1], line=line, name=name)
            theta = _angle(operands[2], line=line, name=name)
            # RXY(phi, theta) = RZ(phi) RX(theta) RZ(-phi), rightmost first.
            instructions.extend(
                (
                    Instruction("rz", qubits, params={"theta": -phi}),
                    Instruction("rx", qubits, params={"theta": theta}),
                    Instruction("rz", qubits, params={"theta": phi}),
                )
            )
        elif name in _ROTATIONS:
            instructions.append(
                Instruction(
                    _ROTATIONS[name],
                    qubits,
                    params={"theta": _angle(operands[-1], line=line, name=name)},
                )
            )
        else:
            instructions.append(Instruction(_FIXED[name], qubits))
    if not labels:
        _refuse(
            "empty_source", "source contains no instructions", line=1, name="<source>"
        )
    report = _report(issues)
    if not allow_lossy and not report.lossless:
        raise InteropConversionError(
            report.issues[0].message
            + "; use allow_lossy=True only for ideal simulation",
            report,
        )
    ordered = sorted(labels)
    mapping = {label: index for index, label in enumerate(ordered)}
    remapped = tuple(
        Instruction(
            item.name, tuple(mapping[q] for q in item.wires), params=item.params
        )
        for item in instructions
    )
    ir = CircuitIR(
        n_wires=len(ordered),
        instructions=remapped,
        dtype="complex128",
        measurements=(
            (MeasurementNode("probabilities", tuple(mapping[q] for q in measured)),)
            if measured
            else ()
        ),
        metadata={
            "interop": {
                "source": "qcis",
                "qubits": tuple(f"Q{q}" for q in ordered),
                "timing_annotations": tuple(timing),
                "conversion_report": report.to_dict(),
            }
        },
    )
    return InteropImportResult(ir, report)
