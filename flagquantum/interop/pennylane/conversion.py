"""Loss-aware static conversion between PennyLane QuantumScript and FlagQuantum IR."""

from __future__ import annotations

from importlib import import_module
from numbers import Number
from typing import Any, Literal

from ...core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ...core.operator_schema import get_operator_schema
from ...core.parameters import Parameter, ParameterExpression
from .models import (
    PennyLaneConversionError,
    PennyLaneConversionIssue,
    PennyLaneConversionReport,
    PennyLaneDependencyError,
    PennyLaneExportResult,
    PennyLaneImportResult,
)

_PENNYLANE_TO_FLAGQUANTUM = {
    "Identity": "i",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "S": "s",
    "T": "t",
    "SX": "sx",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "PhaseShift": "phase",
    "CNOT": "cx",
    "CY": "cy",
    "CZ": "cz",
    "SWAP": "swap",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "ControlledPhaseShift": "cphase",
    "IsingXX": "rxx",
    "IsingYY": "ryy",
    "IsingZZ": "rzz",
    "Toffoli": "ccx",
    "CSWAP": "cswap",
}
_FLAGQUANTUM_TO_PENNYLANE = {
    value: key for key, value in _PENNYLANE_TO_FLAGQUANTUM.items()
}


def _pennylane() -> Any:
    try:
        return import_module("pennylane")
    except ImportError as exc:
        raise PennyLaneDependencyError(
            "PennyLane interoperability requires the optional dependency; install "
            "it with `pip install 'flagquantum[pennylane]'`."
        ) from exc


def _report(
    direction: Literal["from_pennylane", "to_pennylane"],
    version: str | None,
    issues: list[PennyLaneConversionIssue],
) -> PennyLaneConversionReport:
    return PennyLaneConversionReport(direction, version, tuple(issues))


def _raise_if_lossy(report: PennyLaneConversionReport, *, allow_lossy: bool) -> None:
    if allow_lossy or report.lossless:
        return
    codes = ", ".join(issue.code for issue in report.issues)
    raise PennyLaneConversionError(
        "PennyLane conversion is not lossless; inspect error.report or explicitly "
        f"set allow_lossy=True. Issues: {codes}",
        report,
    )


def _real_scalar(
    value: Any, issues: list[PennyLaneConversionIssue], index: int, name: str
) -> float | None:
    requires_grad = bool(getattr(value, "requires_grad", False))
    if requires_grad:
        issues.append(
            PennyLaneConversionIssue(
                "trainable_parameter_not_supported",
                "PennyLane trainable parameters are outside the static IR-only v1 boundary.",
                "error",
                index,
                name,
            )
        )
        return None
    try:
        converted = complex(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
        issues.append(
            PennyLaneConversionIssue(
                "unsupported_parameter_value",
                f"parameter value {value!r} is not a scalar",
                "error",
                index,
                name,
            )
        )
        return None
    if converted.imag != 0:
        issues.append(
            PennyLaneConversionIssue(
                "complex_gate_parameter",
                f"parameter value {value!r} is complex",
                "error",
                index,
                name,
            )
        )
        return None
    return float(converted.real)


def _wire_map(script: Any, issues: list[PennyLaneConversionIssue]) -> dict[Any, int]:
    labels = tuple(script.wires)
    if not labels:
        raise ValueError("PennyLane QuantumScript must reference at least one wire")
    contiguous = all(
        isinstance(wire, int) and not isinstance(wire, bool) for wire in labels
    )
    contiguous = contiguous and set(labels) == set(range(len(labels)))
    if contiguous:
        return {wire: int(wire) for wire in labels}
    issues.append(
        PennyLaneConversionIssue(
            "wire_labels_flattened",
            "PennyLane wire labels are deterministically flattened in script.wires order.",
            "warning",
        )
    )
    return {wire: index for index, wire in enumerate(labels)}


def import_pennylane(
    quantum_script: Any, *, allow_lossy: bool = False
) -> PennyLaneImportResult:
    """Convert an immutable PennyLane ``QuantumScript`` to canonical IR."""

    qml = _pennylane()
    if not isinstance(quantum_script, qml.tape.QuantumScript):
        raise TypeError("from_pennylane requires pennylane.tape.QuantumScript")
    issues: list[PennyLaneConversionIssue] = []
    wires = _wire_map(quantum_script, issues)
    shots = getattr(quantum_script.shots, "total_shots", None)
    if shots is not None:
        issues.append(
            PennyLaneConversionIssue(
                "finite_shots_not_represented",
                "QuantumScript shots are execution policy and are not embedded in CircuitIR.",
                "error",
            )
        )
    if quantum_script.measurements:
        issues.append(
            PennyLaneConversionIssue(
                "measurements_not_represented",
                "PennyLane measurement processes require a separate execution plan.",
                "error",
            )
        )
    instructions: list[Instruction] = []
    for index, operation in enumerate(quantum_script.operations):
        raw_name = str(operation.name)
        opcode = _PENNYLANE_TO_FLAGQUANTUM.get(raw_name)
        if opcode is None:
            issues.append(
                PennyLaneConversionIssue(
                    "unsupported_operation",
                    f"PennyLane operation {raw_name!r} has no v1 lowering.",
                    "error",
                    index,
                    raw_name,
                )
            )
            continue
        schema = get_operator_schema(opcode)
        assert schema is not None
        if len(operation.data) != len(schema.parameters):
            issues.append(
                PennyLaneConversionIssue(
                    "parameter_arity_mismatch",
                    f"operation {raw_name!r} exposes {len(operation.data)} parameter(s), expected {len(schema.parameters)}",
                    "error",
                    index,
                    raw_name,
                )
            )
            continue
        values = [
            _real_scalar(value, issues, index, raw_name) for value in operation.data
        ]
        if any(value is None for value in values):
            continue
        instructions.append(
            Instruction(
                opcode,
                tuple(wires[wire] for wire in operation.wires),
                params=dict(zip(schema.parameters, values)),
            )
        )
    report = _report("from_pennylane", str(qml.__version__), issues)
    ir = CircuitIR(
        n_wires=len(wires),
        instructions=tuple(instructions),
        dtype="complex128",
        shape=(1, 2 ** len(wires)),
        metadata={
            "interop": {
                "source": "pennylane",
                "framework_version": str(qml.__version__),
                "wire_labels": tuple(str(wire) for wire in quantum_script.wires),
                "conversion_report": report.to_dict(),
            }
        },
    )
    _raise_if_lossy(report, allow_lossy=allow_lossy)
    return PennyLaneImportResult(ir, report)


def from_pennylane(quantum_script: Any, *, allow_lossy: bool = False) -> CircuitIR:
    return import_pennylane(quantum_script, allow_lossy=allow_lossy).ir


def export_pennylane(
    program: Any, *, allow_lossy: bool = False
) -> PennyLaneExportResult:
    """Convert FlagQuantum IR to an immutable PennyLane ``QuantumScript``."""

    qml = _pennylane()
    ir = ensure_circuit_ir(program)
    issues: list[PennyLaneConversionIssue] = []
    if ir.observables or ir.measurements:
        issues.append(
            PennyLaneConversionIssue(
                "execution_requests_not_represented",
                "IR observables and measurements require a separate PennyLane execution plan.",
                "error",
            )
        )
    used_wires = {wire for instruction in ir.instructions for wire in instruction.wires}
    if used_wires != set(range(ir.n_wires)):
        issues.append(
            PennyLaneConversionIssue(
                "idle_wire_extent_not_represented",
                "QuantumScript cannot preserve CircuitIR wire extent when wires are idle.",
                "error",
            )
        )
    operations: list[Any] = []
    for index, instruction in enumerate(ir.instructions):
        class_name = _FLAGQUANTUM_TO_PENNYLANE.get(instruction.name)
        if class_name is None or instruction.matrix is not None:
            issues.append(
                PennyLaneConversionIssue(
                    "unsupported_operation",
                    f"FlagQuantum operation {instruction.name!r} has no v1 lowering.",
                    "error",
                    index,
                    instruction.name,
                )
            )
            continue
        schema = get_operator_schema(instruction.name)
        assert schema is not None
        params: list[Any] = []
        blocked = False
        for parameter in schema.parameters:
            value = instruction.params.get(parameter)
            if isinstance(value, (Parameter, ParameterExpression)) or not (
                isinstance(value, Number) or hasattr(value, "item")
            ):
                issues.append(
                    PennyLaneConversionIssue(
                        "symbolic_parameter_not_supported",
                        "Bind FlagQuantum symbolic parameters before PennyLane export.",
                        "error",
                        index,
                        instruction.name,
                    )
                )
                blocked = True
                break
            scalar = _real_scalar(value, issues, index, instruction.name)
            if scalar is None:
                blocked = True
                break
            params.append(scalar)
        if blocked:
            continue
        operation_type = getattr(qml, class_name)
        wire_value: Any = (
            instruction.wires[0] if len(instruction.wires) == 1 else instruction.wires
        )
        operations.append(operation_type(*params, wires=wire_value))
    report = _report("to_pennylane", str(qml.__version__), issues)
    script = qml.tape.QuantumScript(operations, measurements=(), shots=None)
    _raise_if_lossy(report, allow_lossy=allow_lossy)
    return PennyLaneExportResult(script, report)


def to_pennylane(program: Any, *, allow_lossy: bool = False) -> Any:
    return export_pennylane(program, allow_lossy=allow_lossy).quantum_script


__all__ = ("export_pennylane", "from_pennylane", "import_pennylane", "to_pennylane")
