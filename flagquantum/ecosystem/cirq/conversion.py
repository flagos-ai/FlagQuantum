"""Loss-aware static conversion between Cirq circuits and FlagQuantum IR."""

from __future__ import annotations

import math
from importlib import import_module
from numbers import Number
from typing import Any, Literal

from ...core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ...core.operator_schema import get_operator_schema
from ...core.parameters import Parameter, ParameterExpression
from .models import (
    CirqConversionError,
    CirqConversionIssue,
    CirqConversionReport,
    CirqDependencyError,
    CirqExportResult,
    CirqImportResult,
)

_CIRQ_SYMBOL_TO_FLAGQUANTUM = {
    "I": "i",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "H": "h",
    "S": "s",
    "T": "t",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "CNOT": "cx",
    "CZ": "cz",
    "SWAP": "swap",
    "CCX": "ccx",
    "CSWAP": "cswap",
}


def _cirq() -> Any:
    try:
        return import_module("cirq")
    except ImportError as exc:
        raise CirqDependencyError(
            "Cirq interoperability requires the optional dependency; install it "
            "with `pip install 'flagquantum[cirq]'`."
        ) from exc


def _report(
    direction: Literal["from_cirq", "to_cirq"],
    version: str | None,
    issues: list[CirqConversionIssue],
) -> CirqConversionReport:
    return CirqConversionReport(direction, version, tuple(issues))


def _raise_if_lossy(report: CirqConversionReport, *, allow_lossy: bool) -> None:
    if allow_lossy or report.lossless:
        return
    codes = ", ".join(issue.code for issue in report.issues)
    raise CirqConversionError(
        "Cirq conversion is not lossless; inspect error.report or explicitly set "
        f"allow_lossy=True. Issues: {codes}",
        report,
    )


def _issue(
    issues: list[CirqConversionIssue],
    code: str,
    message: str,
    *,
    index: int | None = None,
    name: str | None = None,
    severity: Literal["warning", "error"] = "error",
) -> None:
    issues.append(CirqConversionIssue(code, message, severity, index, name))


def _real_scalar(
    value: Any,
    issues: list[CirqConversionIssue],
    index: int,
    name: str,
) -> float | None:
    try:
        converted = complex(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
        _issue(
            issues,
            "unsupported_parameter_value",
            f"parameter value {value!r} is not a scalar",
            index=index,
            name=name,
        )
        return None
    if converted.imag != 0 or not math.isfinite(converted.real):
        _issue(
            issues,
            "unsupported_parameter_value",
            f"parameter value {value!r} must be a finite real scalar",
            index=index,
            name=name,
        )
        return None
    return float(converted.real)


def _close(value: Any, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _flagquantum_parameter_value(
    cirq: Any, value: Any, *, scale_by_pi: bool = False
) -> Any:
    """Translate Cirq's public symbolic form into owned arithmetic nodes."""

    sympy = import_module("sympy")
    symbolic = sympy.sympify(value)
    if scale_by_pi:
        symbolic = symbolic * sympy.pi
    cirq_symbols = tuple(cirq.parameter_symbols(symbolic))
    parameters_by_name = {
        str(symbol): Parameter(str(symbol)) for symbol in cirq_symbols
    }
    if len(parameters_by_name) != len(cirq_symbols):
        raise ValueError("distinct Cirq symbols share the same name")
    parameters_by_symbol = {
        symbol: parameters_by_name[str(symbol)] for symbol in cirq_symbols
    }

    def convert(node: Any) -> Any:
        if isinstance(node, sympy.Symbol):
            try:
                return parameters_by_symbol[node]
            except KeyError as exc:
                raise ValueError(f"unknown symbol {node!s}") from exc
        if bool(getattr(node, "is_Number", False)):
            converted = complex(node)
            if converted.imag != 0 or not math.isfinite(converted.real):
                raise ValueError(f"constant {node!s} is not a finite real scalar")
            return float(converted.real)
        if isinstance(node, sympy.Add):
            args = tuple(convert(arg) for arg in node.args)
            if not args:
                return 0.0
            result = args[0]
            for arg in args[1:]:
                result = result + arg
            return result
        if isinstance(node, sympy.Mul):
            args = tuple(convert(arg) for arg in node.args)
            if not args:
                return 1.0
            result = args[0]
            for arg in args[1:]:
                result = result * arg
            return result
        function_name = getattr(getattr(node, "func", None), "__name__", None)
        raise ValueError(
            f"operation {function_name or type(node).__name__!r} is unsupported"
        )

    return convert(symbolic)


def _cirq_parameter_value(value: Any, *, cache: dict[str, Any]) -> Any:
    """Translate owned parameter nodes into a SymPy value accepted by Cirq."""

    sympy = import_module("sympy")
    if isinstance(value, Parameter):
        return cache.setdefault(value.name, sympy.Symbol(value.name))
    if isinstance(value, ParameterExpression):
        expected_operands = 1 if value.op == "neg" else 2
        if value.op not in {"add", "sub", "mul", "neg"}:
            raise ValueError(
                f"unsupported FlagQuantum parameter expression {value.op!r}"
            )
        if len(value.args) != expected_operands:
            raise ValueError(
                f"FlagQuantum parameter expression {value.op!r} requires "
                f"{expected_operands} operands; got {len(value.args)}"
            )
        args = tuple(_cirq_parameter_value(arg, cache=cache) for arg in value.args)
        if value.op == "add":
            return args[0] + args[1]
        if value.op == "sub":
            return args[0] + (-args[1])
        if value.op == "mul":
            return args[0] * args[1]
        return -args[0]
    if isinstance(value, Number) or hasattr(value, "item"):
        raw = value.item() if hasattr(value, "item") else value
        converted = complex(raw)
        if converted.imag != 0 or not math.isfinite(converted.real):
            raise ValueError(f"constant {value!r} is not a finite real scalar")
        return float(converted.real)
    raise TypeError(f"unsupported FlagQuantum parameter value {value!r}")


def _qubit_map(
    cirq: Any, circuit: Any, issues: list[CirqConversionIssue]
) -> dict[Any, int]:
    qubits = tuple(sorted(circuit.all_qubits()))
    if not qubits:
        raise ValueError("Cirq circuit must reference at least one qubit")
    if all(isinstance(qubit, cirq.LineQubit) for qubit in qubits):
        indices = tuple(int(qubit.x) for qubit in qubits)
        if indices == tuple(range(len(qubits))):
            return {qubit: int(qubit.x) for qubit in qubits}
        _issue(
            issues,
            "non_contiguous_line_qubits",
            "Cirq LineQubit indices are flattened in ascending order.",
        )
    else:
        _issue(
            issues,
            "non_line_qubit",
            "Cirq qubits must be contiguous LineQubit instances; explicit lossy "
            "conversion flattens them in Cirq's deterministic Qid order.",
        )
    return {qubit: index for index, qubit in enumerate(qubits)}


def _pow_gate(
    cirq: Any,
    gate: Any,
    issues: list[CirqConversionIssue],
    index: int,
    name: str,
) -> tuple[str, dict[str, Any]] | None:
    shift = _real_scalar(gate.global_shift, issues, index, name)
    if shift is None:
        return None
    if cirq.is_parameterized(gate.exponent):
        rotations = (
            (cirq.XPowGate, "rx"),
            (cirq.YPowGate, "ry"),
            (cirq.ZPowGate, "rz"),
        )
        for gate_type, opcode in rotations:
            if isinstance(gate, gate_type) and _close(shift, -0.5):
                try:
                    theta = _flagquantum_parameter_value(
                        cirq, gate.exponent, scale_by_pi=True
                    )
                except (TypeError, ValueError) as exc:
                    _issue(
                        issues,
                        "unsupported_parameter_expression",
                        "Cirq parameter expression is outside the FlagQuantum v1 "
                        f"arithmetic subset: {exc}",
                        index=index,
                        name=name,
                    )
                    return None
                return opcode, {"theta": theta}
        _issue(
            issues,
            "unsupported_parameter_expression",
            "Only symbolic rx, ry, and rz rotation angles preserve Cirq semantics.",
            index=index,
            name=name,
        )
        return None
    exponent = _real_scalar(gate.exponent, issues, index, name)
    if exponent is None:
        return None
    if isinstance(gate, cirq.XPowGate):
        if _close(shift, 0.0) and _close(exponent, 1.0):
            return "x", {}
        if _close(shift, -0.5):
            return "rx", {"theta": math.pi * exponent}
    if isinstance(gate, cirq.YPowGate):
        if _close(shift, 0.0) and _close(exponent, 1.0):
            return "y", {}
        if _close(shift, -0.5):
            return "ry", {"theta": math.pi * exponent}
    if isinstance(gate, cirq.ZPowGate):
        if _close(shift, 0.0):
            for fixed_exponent, opcode in ((1.0, "z"), (0.5, "s"), (0.25, "t")):
                if _close(exponent, fixed_exponent):
                    return opcode, {}
        if _close(shift, -0.5):
            return "rz", {"theta": math.pi * exponent}
    fixed_types = (
        (cirq.HPowGate, "h"),
        (cirq.CXPowGate, "cx"),
        (cirq.CZPowGate, "cz"),
        (cirq.SwapPowGate, "swap"),
        (cirq.CCXPowGate, "ccx"),
    )
    for gate_type, opcode in fixed_types:
        if isinstance(gate, gate_type) and _close(exponent, 1.0) and _close(shift, 0.0):
            return opcode, {}
    return None


def _lower_gate(
    cirq: Any,
    gate: Any,
    issues: list[CirqConversionIssue],
    index: int,
) -> tuple[str, dict[str, Any]] | None:
    name = type(gate).__name__
    if isinstance(gate, cirq.MeasurementGate):
        _issue(
            issues,
            "measurement_not_represented",
            "Cirq measurements require a separate execution plan.",
            index=index,
            name=name,
        )
        return None
    if isinstance(gate, cirq.GlobalPhaseGate):
        _issue(
            issues,
            "global_phase_not_represented",
            "Cirq global phase operations are not represented in CircuitIR.",
            index=index,
            name=name,
        )
        return None
    if isinstance(gate, cirq.IdentityGate) and gate.num_qubits() == 1:
        return "i", {}
    if isinstance(gate, cirq.CSwapGate):
        return "cswap", {}
    if isinstance(
        gate,
        (
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
            cirq.HPowGate,
            cirq.CXPowGate,
            cirq.CZPowGate,
            cirq.SwapPowGate,
            cirq.CCXPowGate,
        ),
    ):
        issue_count = len(issues)
        lowered = _pow_gate(cirq, gate, issues, index, name)
        if lowered is not None:
            return lowered
        if len(issues) != issue_count:
            return None
    _issue(
        issues,
        "unsupported_operation",
        f"Cirq gate {name!r} has no v1 lowering.",
        index=index,
        name=name,
    )
    return None


def import_cirq(circuit: Any, *, allow_lossy: bool = False) -> CirqImportResult:
    """Convert a static Cirq ``Circuit`` to canonical FlagQuantum IR."""

    cirq = _cirq()
    if not isinstance(circuit, cirq.Circuit):
        raise TypeError("from_cirq requires cirq.Circuit")
    issues: list[CirqConversionIssue] = []
    wires = _qubit_map(cirq, circuit, issues)
    instructions: list[Instruction] = []
    operation_index = 0
    for moment in circuit:
        if len(moment.operations) > 1:
            _issue(
                issues,
                "moment_structure_flattened",
                "Parallel Cirq Moment structure is flattened into stable operation order.",
                severity="warning",
            )
        for operation in moment.operations:
            gate = getattr(operation, "gate", None)
            name = type(gate).__name__ if gate is not None else type(operation).__name__
            if (
                getattr(operation, "tags", ())
                or getattr(operation, "classical_controls", ())
                or gate is None
            ):
                _issue(
                    issues,
                    "unsupported_operation",
                    f"Cirq operation {name!r} carries control metadata, tags, or no "
                    "gate and cannot be lowered.",
                    index=operation_index,
                    name=name,
                )
                operation_index += 1
                continue
            lowered = _lower_gate(cirq, gate, issues, operation_index)
            if lowered is not None:
                opcode, params = lowered
                instructions.append(
                    Instruction(
                        opcode,
                        tuple(wires[qubit] for qubit in operation.qubits),
                        params=params,
                    )
                )
            operation_index += 1
    report = _report("from_cirq", str(cirq.__version__), issues)
    ir = CircuitIR(
        n_wires=len(wires),
        instructions=tuple(instructions),
        dtype="complex128",
        shape=(1, 2 ** len(wires)),
        metadata={
            "interop": {
                "source": "cirq",
                "framework_version": str(cirq.__version__),
                "qubits": tuple(
                    str(qubit) for qubit in sorted(wires, key=lambda item: wires[item])
                ),
                "conversion_report": report.to_dict(),
            }
        },
    )
    _raise_if_lossy(report, allow_lossy=allow_lossy)
    return CirqImportResult(ir, report)


def from_cirq(circuit: Any, *, allow_lossy: bool = False) -> CircuitIR:
    return import_cirq(circuit, allow_lossy=allow_lossy).ir


def _export_gate(cirq: Any, instruction: Instruction, params: list[Any]) -> Any:
    gates = {
        "i": cirq.I,
        "x": cirq.X,
        "y": cirq.Y,
        "z": cirq.Z,
        "h": cirq.H,
        "s": cirq.S,
        "t": cirq.T,
        "cx": cirq.CNOT,
        "cz": cirq.CZ,
        "swap": cirq.SWAP,
        "ccx": cirq.CCX,
        "cswap": cirq.CSWAP,
    }
    if instruction.name == "rx":
        return cirq.rx(params[0])
    if instruction.name == "ry":
        return cirq.ry(params[0])
    if instruction.name == "rz":
        return cirq.rz(params[0])
    return gates[instruction.name]


def export_cirq(program: Any, *, allow_lossy: bool = False) -> CirqExportResult:
    """Convert FlagQuantum IR to a static Cirq ``Circuit``."""

    cirq = _cirq()
    ir = ensure_circuit_ir(program)
    issues: list[CirqConversionIssue] = []
    if ir.observables or ir.measurements:
        _issue(
            issues,
            "measurement_not_represented",
            "IR observables and measurements require a separate Cirq execution plan.",
        )
    used_wires = {wire for instruction in ir.instructions for wire in instruction.wires}
    if used_wires != set(range(ir.n_wires)):
        _issue(
            issues,
            "idle_wire_extent_not_represented",
            "Cirq Circuit cannot preserve CircuitIR wire extent when wires are idle.",
        )
    qubits = cirq.LineQubit.range(ir.n_wires)
    operations: list[Any] = []
    parameter_cache: dict[str, Any] = {}
    for index, instruction in enumerate(ir.instructions):
        if (
            instruction.name not in _CIRQ_SYMBOL_TO_FLAGQUANTUM.values()
            or instruction.matrix is not None
        ):
            _issue(
                issues,
                "unsupported_operation",
                f"FlagQuantum operation {instruction.name!r} has no Cirq v1 lowering.",
                index=index,
                name=instruction.name,
            )
            continue
        schema = get_operator_schema(instruction.name)
        assert schema is not None
        params: list[Any] = []
        blocked = False
        for parameter in schema.parameters:
            value = instruction.params.get(parameter)
            if isinstance(value, (Parameter, ParameterExpression)):
                try:
                    params.append(_cirq_parameter_value(value, cache=parameter_cache))
                except (TypeError, ValueError) as exc:
                    _issue(
                        issues,
                        "unsupported_parameter_expression",
                        "FlagQuantum parameter expression is outside the Cirq v1 "
                        f"arithmetic subset: {exc}",
                        index=index,
                        name=instruction.name,
                    )
                    blocked = True
                    break
                continue
            if not (isinstance(value, Number) or hasattr(value, "item")):
                _issue(
                    issues,
                    "unsupported_parameter_value",
                    f"parameter value {value!r} is not a scalar",
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
        gate = _export_gate(cirq, instruction, params)
        operations.append(gate.on(*(qubits[wire] for wire in instruction.wires)))
    report = _report("to_cirq", str(cirq.__version__), issues)
    circuit = cirq.Circuit(cirq.Moment([operation]) for operation in operations)
    _raise_if_lossy(report, allow_lossy=allow_lossy)
    return CirqExportResult(circuit, report)


def to_cirq(program: Any, *, allow_lossy: bool = False) -> Any:
    return export_cirq(program, allow_lossy=allow_lossy).circuit


__all__ = ("export_cirq", "from_cirq", "import_cirq", "to_cirq")
