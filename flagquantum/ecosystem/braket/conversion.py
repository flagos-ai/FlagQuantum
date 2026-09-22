"""Loss-aware static conversion between Amazon Braket circuits and FlagQuantum IR."""

from __future__ import annotations

import math
from importlib import import_module, metadata
from numbers import Number
from typing import Any, Literal

from ...core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ...core.operator_schema import get_operator_schema
from ...core.parameters import Parameter, ParameterExpression
from .models import (
    BraketConversionError,
    BraketConversionIssue,
    BraketConversionReport,
    BraketDependencyError,
    BraketExportResult,
    BraketImportResult,
)

_BRAKET_GATE_TO_FLAGQUANTUM = {
    "I": "i",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "H": "h",
    "S": "s",
    "Si": "sdg",
    "T": "t",
    "Ti": "tdg",
    "V": "sx",
    "Vi": "sxdg",
    "Rx": "rx",
    "Ry": "ry",
    "Rz": "rz",
    "PhaseShift": "phase",
    "U": "u3",
    "CNot": "cx",
    "CY": "cy",
    "CZ": "cz",
    "Swap": "swap",
    "CPhaseShift": "cphase",
    "XX": "rxx",
    "YY": "ryy",
    "ZZ": "rzz",
    "CCNot": "ccx",
    "CSwap": "cswap",
}

_FLAGQUANTUM_TO_BRAKET_METHOD = {
    "i": "i",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "si",
    "t": "t",
    "tdg": "ti",
    "sx": "v",
    "sxdg": "vi",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "phase": "phaseshift",
    "u3": "u",
    "cx": "cnot",
    "cy": "cy",
    "cz": "cz",
    "swap": "swap",
    "cphase": "cphaseshift",
    "rxx": "xx",
    "ryy": "yy",
    "rzz": "zz",
    "ccx": "ccnot",
    "cswap": "cswap",
}


def _braket() -> Any:
    try:
        return import_module("braket.circuits")
    except ImportError as exc:
        raise BraketDependencyError(
            "Amazon Braket interoperability requires the optional dependency; "
            "install it with `pip install 'flagquantum[braket]'`."
        ) from exc


def _version() -> str:
    return metadata.version("amazon-braket-sdk")


def _report(
    direction: Literal["from_braket", "to_braket"],
    issues: list[BraketConversionIssue],
) -> BraketConversionReport:
    return BraketConversionReport(direction, _version(), tuple(issues))


def _raise_if_lossy(report: BraketConversionReport, *, allow_lossy: bool) -> None:
    if allow_lossy or report.lossless:
        return
    codes = ", ".join(issue.code for issue in report.issues)
    raise BraketConversionError(
        "Amazon Braket conversion is not lossless; inspect error.report or "
        f"explicitly set allow_lossy=True. Issues: {codes}",
        report,
    )


def _issue(
    issues: list[BraketConversionIssue],
    code: str,
    message: str,
    *,
    index: int | None = None,
    name: str | None = None,
) -> None:
    issues.append(BraketConversionIssue(code, message, "error", index, name))


def _real_scalar(
    value: Any,
    issues: list[BraketConversionIssue],
    index: int,
    name: str,
    *,
    symbolic_type: type[Any] | tuple[type[Any], ...] = (),
) -> float | None:
    if symbolic_type and isinstance(value, symbolic_type):
        _issue(
            issues,
            "symbolic_parameter_not_supported",
            "Bind symbolic parameters before Amazon Braket conversion.",
            index=index,
            name=name,
        )
        return None
    try:
        converted = complex(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
        _issue(
            issues,
            "non_real_parameter",
            f"parameter value {value!r} must be a real scalar",
            index=index,
            name=name,
        )
        return None
    if converted.imag != 0:
        _issue(
            issues,
            "non_real_parameter",
            f"parameter value {value!r} must be real",
            index=index,
            name=name,
        )
        return None
    if not math.isfinite(converted.real):
        _issue(
            issues,
            "non_finite_parameter",
            f"parameter value {value!r} must be finite",
            index=index,
            name=name,
        )
        return None
    return float(converted.real)


def import_braket(circuit: Any, *, allow_lossy: bool = False) -> BraketImportResult:
    """Convert a static Amazon Braket ``Circuit`` to FlagQuantum IR."""

    braket = _braket()
    if not isinstance(circuit, braket.Circuit):
        raise TypeError("from_braket requires braket.circuits.Circuit")
    issues: list[BraketConversionIssue] = []
    if circuit.result_types:
        _issue(
            issues,
            "result_type_not_represented",
            "Braket result types require a separate execution plan.",
        )
    if getattr(circuit, "gate_definitions", {}):
        _issue(
            issues,
            "gate_calibration_not_represented",
            "Braket gate calibrations are not represented in CircuitIR.",
        )
    qubits = tuple(int(qubit) for qubit in circuit.qubits)
    if not qubits:
        _issue(
            issues,
            "wire_extent_not_represented",
            "An empty Braket circuit does not carry a FlagQuantum wire extent.",
        )
        report = _report("from_braket", issues)
        raise BraketConversionError(
            "Amazon Braket conversion requires at least one referenced qubit.",
            report,
        )

    instructions: list[Instruction] = []
    symbolic_type = braket.FreeParameterExpression
    for index, external in enumerate(circuit.instructions):
        operator = external.operator
        name = type(operator).__name__
        if name == "Measure":
            _issue(
                issues,
                "measurement_not_represented",
                "Braket measurements require a separate execution plan.",
                index=index,
                name=name,
            )
            continue
        opcode = _BRAKET_GATE_TO_FLAGQUANTUM.get(name)
        if opcode is None:
            _issue(
                issues,
                "unsupported_instruction",
                f"Braket instruction {name!r} has no static v1 lowering.",
                index=index,
                name=name,
            )
            continue
        schema = get_operator_schema(opcode)
        assert schema is not None
        values = tuple(getattr(operator, "parameters", ()))
        if len(values) != len(schema.parameters):
            _issue(
                issues,
                "unsupported_instruction",
                f"Braket gate {name!r} exposes an unexpected parameter shape.",
                index=index,
                name=name,
            )
            continue
        params: dict[str, float] = {}
        blocked = False
        for parameter, value in zip(schema.parameters, values, strict=True):
            scalar = _real_scalar(
                value,
                issues,
                index,
                name,
                symbolic_type=symbolic_type,
            )
            if scalar is None:
                blocked = True
                break
            params[parameter] = scalar
        if blocked:
            continue
        instructions.append(
            Instruction(
                opcode,
                tuple(int(qubit) for qubit in external.target),
                params=params,
            )
        )

    report = _report("from_braket", issues)
    ir = CircuitIR(
        n_wires=max(qubits) + 1,
        instructions=tuple(instructions),
        dtype="complex128",
        shape=(1, 2 ** (max(qubits) + 1)),
        metadata={
            "interop": {
                "source": "braket",
                "framework_version": report.framework_version,
                "conversion_report": report.to_dict(),
            }
        },
    )
    _raise_if_lossy(report, allow_lossy=allow_lossy)
    return BraketImportResult(ir, report)


def from_braket(circuit: Any, *, allow_lossy: bool = False) -> CircuitIR:
    return import_braket(circuit, allow_lossy=allow_lossy).ir


def export_braket(program: Any, *, allow_lossy: bool = False) -> BraketExportResult:
    """Convert FlagQuantum IR to a static Amazon Braket ``Circuit``."""

    braket = _braket()
    ir = ensure_circuit_ir(program)
    issues: list[BraketConversionIssue] = []
    if ir.observables or ir.measurements:
        _issue(
            issues,
            "measurement_not_represented",
            "IR observables and measurements require a separate Braket execution plan.",
        )
    circuit = braket.Circuit()
    used_wires: set[int] = set()
    for index, instruction in enumerate(ir.instructions):
        method_name = _FLAGQUANTUM_TO_BRAKET_METHOD.get(instruction.name)
        if method_name is None or instruction.matrix is not None:
            _issue(
                issues,
                "unsupported_operation",
                f"FlagQuantum operation {instruction.name!r} has no Braket v1 lowering.",
                index=index,
                name=instruction.name,
            )
            continue
        schema = get_operator_schema(instruction.name)
        assert schema is not None
        params: list[float] = []
        blocked = False
        for parameter in schema.parameters:
            value = instruction.params.get(parameter)
            if isinstance(value, (Parameter, ParameterExpression)) or not (
                isinstance(value, Number) or hasattr(value, "item")
            ):
                _issue(
                    issues,
                    "symbolic_parameter_not_supported",
                    "Bind FlagQuantum symbolic parameters before Braket export.",
                    index=index,
                    name=instruction.name,
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
        getattr(circuit, method_name)(*instruction.wires, *params)
        used_wires.update(instruction.wires)

    for wire in sorted(set(range(ir.n_wires)) - used_wires):
        circuit.i(wire)
    report = _report("to_braket", issues)
    _raise_if_lossy(report, allow_lossy=allow_lossy)
    return BraketExportResult(circuit, report)


def to_braket(program: Any, *, allow_lossy: bool = False) -> Any:
    return export_braket(program, allow_lossy=allow_lossy).circuit


__all__ = ("export_braket", "from_braket", "import_braket", "to_braket")
