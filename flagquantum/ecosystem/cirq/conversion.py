"""Loss-aware static conversion between Cirq circuits and FlagQuantum IR."""

from __future__ import annotations

import math
from collections.abc import Iterable
from importlib import import_module
from numbers import Number
from typing import Any, Literal

from ...compiler.formats.symbolic_parameters import symbolic_parameter_value
from ...core.classical_control import (
    instruction_condition_clauses,
    validate_classical_control,
)
from ...core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ...core.operator_schema import get_operator_schema
from ...core.parameters import Parameter, ParameterExpression
from ._register_feedback import RegisterFeedback
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
    "CH": "ch",
    "S": "s",
    "SDG": "sdg",
    "T": "t",
    "TDG": "tdg",
    "ZPowGate": "phase",
    "CZPowGate": "cphase",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "ZZPowGate": "rzz",
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
    discovered = {str(symbol) for symbol in cirq.parameter_symbols(symbolic)}
    cirq_symbols = tuple(symbolic.free_symbols)
    if {str(symbol) for symbol in cirq_symbols} != discovered:
        raise ValueError("Cirq and SymPy symbol discovery disagree")
    if len({str(symbol) for symbol in cirq_symbols}) != len(cirq_symbols):
        raise ValueError("distinct Cirq symbols share the same name")
    return symbolic_parameter_value(symbolic)


def _cirq_parameter_value(value: Any, *, cache: dict[str, Any]) -> Any:
    """Translate owned parameter nodes into a SymPy value accepted by Cirq."""

    sympy = import_module("sympy")
    if isinstance(value, Parameter):
        return cache.setdefault(value.name, sympy.Symbol(value.name))
    if isinstance(value, ParameterExpression):
        expected_operands = 1 if value.op in {"neg", "sin", "cos", "exp", "log"} else 2
        if value.op not in {
            "add",
            "sub",
            "mul",
            "neg",
            "sin",
            "cos",
            "exp",
            "log",
            "pow",
            "div",
        }:
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
        if value.op == "pow":
            return args[0] ** args[1]
        if value.op == "div":
            return args[0] / args[1]
        if value.op == "neg":
            return -args[0]
        return getattr(sympy, value.op)(args[0])
    if isinstance(value, Number) or hasattr(value, "item"):
        raw = value.item() if hasattr(value, "item") else value
        converted = complex(raw)
        if converted.imag != 0 or not math.isfinite(converted.real):
            raise ValueError(f"constant {value!r} is not a finite real scalar")
        return float(converted.real)
    raise TypeError(f"unsupported FlagQuantum parameter value {value!r}")


def _eigen_phase_operations(cirq: Any, operations: Iterable[Any]) -> Iterable[Any]:
    """Factor admitted eigen-gates into their canonical gate and scalar phase."""
    sympy = import_module("sympy")
    for original in operations:
        if original.tags:
            yield original
            continue
        operation = original.without_classical_controls()
        gate = operation.gate
        controls: tuple[Any, ...] = ()
        control_values = None
        if isinstance(gate, cirq.ControlledGate):
            controls = tuple(operation.qubits[: gate.num_controls()])
            control_values = gate.control_values
            gate = gate.sub_gate
        if len(controls) > 2:
            # Larger controlled rotations have no native scalar leaf. Preserve
            # the SDK decomposition before factoring phases, otherwise its
            # canonical rotation decomposition can normalize back to itself.
            yield original
            continue
        if (
            not isinstance(
                gate,
                (
                    cirq.XPowGate,
                    cirq.YPowGate,
                    cirq.ZPowGate,
                    cirq.ZZPowGate,
                    cirq.CZPowGate,
                ),
            )
            or _lower_gate(cirq, gate, [], 0) is not None
        ):
            yield original
            continue
        if (
            isinstance(gate, cirq.XPowGate)
            and gate.global_shift == 0
            and not cirq.is_parameterized(gate.exponent)
            and float(gate.exponent) % 2 in (0.5, 1.5)
        ):
            yield original
            continue
        if any(qubit.dimension != 2 for qubit in operation.qubits):
            yield original
            continue
        exponent = sympy.sympify(gate.exponent)
        shift = sympy.sympify(gate.global_shift)
        try:
            _flagquantum_parameter_value(cirq, exponent)
            _flagquantum_parameter_value(cirq, shift)
        except (TypeError, ValueError):
            yield original
            continue
        if isinstance(gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZZPowGate)):
            canonical_shift = sympy.Rational(-1, 2)
        else:
            canonical_shift = sympy.Integer(0)
        families = (
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
            cirq.ZZPowGate,
            cirq.CZPowGate,
        )
        family = next(kind for kind in families if isinstance(gate, kind))
        canonical = family(exponent=gate.exponent, global_shift=float(canonical_shift))
        angle = sympy.pi * exponent * (shift - canonical_shift)
        leaves = [canonical.on(*operation.qubits[len(controls) :])]
        if angle != 0:
            leaves.append(cirq.global_phase_operation(sympy.exp(sympy.I * angle)))
        for leaf in leaves:
            if controls:
                leaf = leaf.controlled_by(*controls, control_values=control_values)
            yield leaf.with_classical_controls(*original.classical_controls)


def _check_cirq_symbol_identity(cirq: Any, operations: Iterable[Any]) -> None:
    """Reject distinct symbols sharing one owned name across supported leaves."""
    sympy = import_module("sympy")
    symbols: dict[str, Any] = {}
    for operation in operations:
        gate = operation.gate
        while isinstance(gate, cirq.ControlledGate):
            gate = gate.sub_gate
        if isinstance(gate, cirq.GlobalPhaseGate):
            values: tuple[Any, ...] = (gate.coefficient,)
        elif isinstance(gate, cirq.EigenGate):
            values = (gate.exponent, gate.global_shift)
        else:
            continue
        for value in values:
            for symbol in sympy.sympify(value).free_symbols:
                name = str(symbol)
                if name in symbols and symbols[name] != symbol:
                    issues: list[CirqConversionIssue] = []
                    _issue(
                        issues,
                        "unsupported_parameter_expression",
                        "distinct Cirq symbols share the same name",
                    )
                    raise CirqConversionError(
                        "distinct Cirq symbols share the same name",
                        _report("from_cirq", str(cirq.__version__), issues),
                    )
                symbols[name] = symbol


def _global_phase_angle(cirq: Any, coefficient: Any) -> Any:
    """Derive an owned real angle before admitting a scalar coefficient."""
    sympy = import_module("sympy")
    if not cirq.is_parameterized(coefficient):
        value = complex(coefficient)
        if not (
            math.isfinite(value.real) and math.isfinite(value.imag)
        ) or not math.isclose(abs(value), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Global phase coefficient must be finite and unit modulus")
        return math.atan2(value.imag, value.real)
    symbolic = sympy.powsimp(sympy.sympify(coefficient), combine="exp")
    if isinstance(symbolic, sympy.Pow) and not symbolic.base.free_symbols:
        base = complex(symbolic.base)
        if (
            math.isfinite(base.real)
            and math.isfinite(base.imag)
            and math.isclose(abs(base), 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            return _flagquantum_parameter_value(
                cirq, sympy.arg(symbolic.base) * symbolic.exp
            )
    if symbolic.func != sympy.exp:
        raise ValueError(
            "Symbolic global phase requires a proven exp(i * real_angle) coefficient"
        )
    angle = sympy.simplify(symbolic.args[0] / sympy.I)
    # Owned expression leaves denote real values. Nonlinear domains are checked
    # by Core binding, while complex constants and foreign functions stop here.
    return _flagquantum_parameter_value(cirq, angle)


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
    if (
        isinstance(gate, (cirq.ZPowGate, cirq.CZPowGate))
        and shift == 0
        and cirq.is_parameterized(gate.exponent)
    ):
        try:
            return ("cphase" if isinstance(gate, cirq.CZPowGate) else "phase"), {
                "theta": _flagquantum_parameter_value(
                    cirq, gate.exponent, scale_by_pi=True
                )
            }
        except (TypeError, ValueError) as exc:
            _issue(
                issues,
                "unsupported_parameter_expression",
                str(exc),
                index=index,
                name=name,
            )
            return None
    if cirq.is_parameterized(gate.exponent):
        rotations = (
            (cirq.XPowGate, "rx"),
            (cirq.YPowGate, "ry"),
            (cirq.ZPowGate, "rz"),
            (cirq.ZZPowGate, "rzz"),
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
                        "Cirq parameter expression is outside the supported "
                        f"real expression vocabulary: {exc}",
                        index=index,
                        name=name,
                    )
                    return None
                return opcode, {"theta": theta}
        _issue(
            issues,
            "unsupported_parameter_expression",
            "Only symbolic rx, ry, rz, and rzz rotation angles preserve Cirq semantics.",
            index=index,
            name=name,
        )
        return None
    exponent = _real_scalar(gate.exponent, issues, index, name)
    if exponent is None:
        return None
    if isinstance(gate, cirq.XPowGate):
        if _close(shift, 0.0) and _close(abs(exponent), 1.0):
            return "x", {}
        if _close(shift, -0.5):
            return "rx", {"theta": math.pi * exponent}
    if isinstance(gate, cirq.YPowGate):
        if _close(shift, 0.0) and _close(abs(exponent), 1.0):
            return "y", {}
        if _close(shift, -0.5):
            return "ry", {"theta": math.pi * exponent}
    if isinstance(gate, cirq.ZPowGate):
        if _close(shift, 0.0):
            if _close(abs(exponent), 1.0):
                return "z", {}
            for fixed_exponent, opcode in (
                (0.5, "s"),
                (-0.5, "sdg"),
                (0.25, "t"),
                (-0.25, "tdg"),
            ):
                if _close(exponent, fixed_exponent):
                    return opcode, {}
        if shift == 0:
            return "phase", {"theta": math.pi * exponent}
        if _close(shift, -0.5):
            return "rz", {"theta": math.pi * exponent}
    if isinstance(gate, cirq.CZPowGate) and shift == 0:
        if not _close(abs(exponent), 1.0):
            return "cphase", {"theta": math.pi * exponent}
    if isinstance(gate, cirq.ZZPowGate) and _close(shift, -0.5):
        return "rzz", {"theta": math.pi * exponent}
    fixed_types = (
        (cirq.HPowGate, "h"),
        (cirq.CXPowGate, "cx"),
        (cirq.CZPowGate, "cz"),
        (cirq.SwapPowGate, "swap"),
        (cirq.CCXPowGate, "ccx"),
    )
    for gate_type, opcode in fixed_types:
        if (
            isinstance(gate, gate_type)
            and _close(abs(exponent), 1.0)
            and _close(shift, 0.0)
        ):
            return opcode, {}
    return None


def _lower_gate(
    cirq: Any,
    gate: Any,
    issues: list[CirqConversionIssue],
    index: int,
) -> tuple[str, dict[str, Any]] | None:
    name = type(gate).__name__
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
    if (
        isinstance(gate, cirq.ControlledGate)
        and isinstance(gate.sub_gate, cirq.HPowGate)
        and gate.sub_gate.exponent == 1
        and gate.sub_gate.global_shift == 0
        and gate == cirq.H.controlled()
    ):
        return "ch", {}
    if isinstance(
        gate,
        (
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
            cirq.ZZPowGate,
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


def _lower_measurement(
    cirq: Any,
    operation: Any,
    wires: dict[Any, int],
    issues: list[CirqConversionIssue],
    index: int,
    *,
    seen_keys: set[str],
    first_classical_bit: int,
) -> tuple[Instruction, ...]:
    """Lower one terminal Cirq measurement without retaining Cirq objects."""

    gate = operation.gate
    name = type(gate).__name__
    key = gate.key
    blocked = False
    if not isinstance(key, str) or not key:
        _issue(
            issues,
            "measurement_key_empty",
            "Cirq measurement keys must be non-empty strings.",
            index=index,
            name=name,
        )
        blocked = True
    elif key in seen_keys:
        _issue(
            issues,
            "duplicate_measurement_key",
            f"Cirq measurement key {key!r} is used by more than one gate.",
            index=index,
            name=name,
        )
        blocked = True
    if tuple(gate.invert_mask):
        _issue(
            issues,
            "measurement_invert_mask_not_representable",
            "Cirq measurement invert masks are not represented in FlagQuantum IR.",
            index=index,
            name=name,
        )
        blocked = True
    if dict(gate.confusion_map):
        _issue(
            issues,
            "measurement_confusion_map_not_representable",
            "Cirq measurement confusion maps are not represented in FlagQuantum IR.",
            index=index,
            name=name,
        )
        blocked = True
    qid_shape = tuple(cirq.qid_shape(operation))
    if any(dimension != 2 for dimension in qid_shape):
        _issue(
            issues,
            "measurement_qid_dimension_not_representable",
            "Cirq measurements are limited to two-dimensional qids.",
            index=index,
            name=name,
        )
        blocked = True
    if (
        len(operation.qubits) == 0
        or gate.num_qubits() != len(operation.qubits)
        or len(qid_shape) != len(operation.qubits)
    ):
        _issue(
            issues,
            "measurement_not_representable",
            "Cirq measurement gate shape or arity is not representable.",
            index=index,
            name=name,
        )
        blocked = True
    if blocked:
        return ()
    seen_keys.add(key)
    return tuple(
        Instruction(
            "measure",
            (wires[qubit],),
            metadata={
                "is_dynamic": True,
                "classical_bit": first_classical_bit + offset,
                "measurement_key": key,
            },
        )
        for offset, qubit in enumerate(operation.qubits)
    )


def import_cirq(circuit: Any, *, allow_lossy: bool = False) -> CirqImportResult:
    """Convert a static Cirq ``Circuit`` to canonical FlagQuantum IR."""

    cirq = _cirq()
    if not isinstance(circuit, cirq.Circuit):
        raise TypeError("from_cirq requires cirq.Circuit")
    return _import_static_cirq(cirq, circuit, allow_lossy=allow_lossy)


def _import_static_cirq(
    cirq: Any,
    circuit: Any,
    *,
    allow_lossy: bool = False,
    wire_order: tuple[Any, ...] | None = None,
) -> CirqImportResult:
    issues: list[CirqConversionIssue] = []
    wires = (
        _qubit_map(cirq, circuit, issues)
        if wire_order is None
        else {qubit: index for index, qubit in enumerate(wire_order)}
    )
    instructions: list[Instruction] = []
    _check_cirq_symbol_identity(cirq, circuit.all_operations())
    indexed_operations = tuple(
        operation
        for moment in circuit
        for operation in _eigen_phase_operations(cirq, moment.operations)
    )
    _check_cirq_symbol_identity(cirq, indexed_operations)
    last_non_measurement = max(
        (
            index
            for index, operation in enumerate(indexed_operations)
            if not isinstance(getattr(operation, "gate", None), cirq.MeasurementGate)
        ),
        default=-1,
    )
    seen_measurement_keys: set[str] = set()
    classical_bit = 0
    operation_index = 0
    for moment in circuit:
        measurement_only = all(
            isinstance(getattr(operation, "gate", None), cirq.MeasurementGate)
            for operation in moment.operations
        )
        if len(moment.operations) > 1 and not measurement_only:
            _issue(
                issues,
                "moment_structure_flattened",
                "Parallel Cirq Moment structure is flattened into stable operation order.",
                severity="warning",
            )
        for operation in _eigen_phase_operations(cirq, moment.operations):
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
            if isinstance(gate, cirq.MeasurementGate):
                if operation_index < last_non_measurement:
                    _issue(
                        issues,
                        "measurement_not_terminal",
                        "Cirq measurements must follow every non-measurement operation.",
                        index=operation_index,
                        name=name,
                    )
                else:
                    lowered_measurement = _lower_measurement(
                        cirq,
                        operation,
                        wires,
                        issues,
                        operation_index,
                        seen_keys=seen_measurement_keys,
                        first_classical_bit=classical_bit,
                    )
                    instructions.extend(lowered_measurement)
                    classical_bit += len(lowered_measurement)
                operation_index += 1
                continue
            if isinstance(gate, cirq.GlobalPhaseGate):
                from ...compiler.registers.primitives.global_phase import (
                    scalar_phase_ir,
                )

                try:
                    angle = _global_phase_angle(cirq, gate.coefficient)
                    if not wires:
                        raise ValueError("Global phase requires a declared anchor wire")
                    instructions.extend(scalar_phase_ir(0, angle))
                except (TypeError, ValueError) as exc:
                    _issue(
                        issues,
                        "global_phase_not_represented",
                        str(exc),
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
        "ch": cirq.H.controlled(),
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
    if instruction.name == "rzz":
        return cirq.ZZPowGate(exponent=params[0] / math.pi, global_shift=-0.5)
    if instruction.name == "phase":
        return cirq.ZPowGate(exponent=params[0] / math.pi)
    if instruction.name == "cphase":
        return cirq.CZPowGate(exponent=params[0] / math.pi)
    if instruction.name in {"crx", "cry", "crz"}:
        rotation = {"crx": cirq.rx, "cry": cirq.ry, "crz": cirq.rz}
        return rotation[instruction.name](params[0]).controlled()
    if instruction.name == "sdg":
        return cirq.S**-1
    if instruction.name == "tdg":
        return cirq.T**-1
    return gates[instruction.name]


def export_cirq(program: Any, *, allow_lossy: bool = False) -> CirqExportResult:
    """Convert FlagQuantum IR to a static Cirq ``Circuit``."""
    return _export_cirq(program, allow_lossy=allow_lossy)


def _export_cirq(
    program: Any,
    *,
    allow_lossy: bool,
    include_inverse_phases: bool = False,
    extent_declared_by_gate: bool = False,
    feedback_key_prefix: str | None = None,
    guarded_measurement_work: int | None = None,
    instruction_offsets: list[int] | None = None,
) -> CirqExportResult:
    cirq = _cirq()
    ir = ensure_circuit_ir(program)
    supported_gates = set(_CIRQ_SYMBOL_TO_FLAGQUANTUM.values())
    if include_inverse_phases:
        supported_gates.update(("sdg", "tdg"))
    if extent_declared_by_gate:
        supported_gates.update(("phase", "cphase", "crx", "cry", "crz"))
    issues: list[CirqConversionIssue] = []
    feedback = None
    if feedback_key_prefix is not None:
        if allow_lossy or not feedback_key_prefix.isidentifier():
            raise ValueError(
                "Cirq feedback export requires a key prefix and strict conversion"
            )
        validate_classical_control(ir.instructions, n_qubits=ir.n_wires)
        feedback = RegisterFeedback()
    if ir.observables or ir.measurements:
        _issue(
            issues,
            "measurement_not_represented",
            "IR observables and measurements require a separate Cirq execution plan.",
        )
    used_wires = {wire for instruction in ir.instructions for wire in instruction.wires}
    if guarded_measurement_work is not None:
        if (
            feedback is None
            or type(guarded_measurement_work) is not int
            or not 0 <= guarded_measurement_work < ir.n_wires
            or guarded_measurement_work in used_wires
        ):
            raise ValueError(
                "Cirq guarded measurement requires an unused clean wire and feedback"
            )
        if any(
            gate.name == "measure" and gate.metadata.get("guarded_effect") is True
            for gate in ir.instructions
        ):
            used_wires.add(guarded_measurement_work)
    if not extent_declared_by_gate and used_wires != set(range(ir.n_wires)):
        _issue(
            issues,
            "idle_wire_extent_not_represented",
            "Cirq Circuit cannot preserve CircuitIR wire extent when wires are idle.",
        )
    qubits = cirq.LineQubit.range(ir.n_wires)
    operations: list[Any] = []
    parameter_cache: dict[str, Any] = {}
    measurement_started = False
    completed_measurement_keys: set[str] = set()
    current_measurement_key: str | None = None
    current_measurement_qubits: list[Any] = []
    next_classical_bit = 0

    def flush_measurement() -> None:
        nonlocal current_measurement_key, current_measurement_qubits
        if current_measurement_key is not None and current_measurement_qubits:
            operations.append(
                cirq.measure(
                    *current_measurement_qubits,
                    key=current_measurement_key,
                )
            )
            completed_measurement_keys.add(current_measurement_key)
        current_measurement_key = None
        current_measurement_qubits = []

    for index, instruction in enumerate(ir.instructions):
        if instruction_offsets is not None:
            instruction_offsets.append(len(operations))
        clauses = (
            instruction_condition_clauses(instruction)
            if feedback_key_prefix is not None
            else ()
        )
        if feedback is not None:
            if instruction.name in {"measure", "reset"}:
                if len(instruction.wires) != 1 or instruction.matrix is not None:
                    _issue(
                        issues,
                        "unsupported_operation",
                        "Cirq register feedback requires a one-wire measurement or reset.",
                        index=index,
                        name=instruction.name,
                    )
                    break
                qubit = qubits[instruction.wires[0]]
                if instruction.name == "measure":
                    write_key = f"{feedback_key_prefix}_{index}"
                    if clauses:
                        if guarded_measurement_work is None:
                            _issue(
                                issues,
                                "unsupported_operation",
                                "Cirq guarded measurement requires a clean work wire.",
                                index=index,
                                name=instruction.name,
                            )
                            break
                        work = qubits[guarded_measurement_work]
                        controlled_copy = feedback.condition_operation(
                            cirq, cirq.CNOT(qubit, work), clauses
                        )
                        operations.extend(
                            (
                                controlled_copy,
                                cirq.measure(work, key=write_key),
                                cirq.reset(work),
                            )
                        )
                    else:
                        operations.append(cirq.measure(qubit, key=write_key))
                    feedback.record_measurement(
                        instruction.metadata["classical_bit"], write_key, clauses
                    )
                else:
                    reset = cirq.reset(qubit)
                    if clauses:
                        reset = feedback.condition_operation(cirq, reset, clauses)
                    operations.append(reset)
                continue
            if instruction.name == "classical_xor":
                feedback.assign(instruction, clauses)
                continue
        elif (
            "conditions" in instruction.metadata
            or "condition_clauses" in instruction.metadata
        ):
            _issue(
                issues,
                "unsupported_operation",
                "Cirq v1 export does not preserve FlagQuantum classical controls.",
                index=index,
                name=instruction.name,
            )
            continue
        if instruction.name == "measure":
            measurement_started = True
            key = instruction.metadata.get("measurement_key")
            classical_bit = instruction.metadata.get("classical_bit")
            blocked = False
            if not isinstance(key, str) or not key:
                _issue(
                    issues,
                    "measurement_key_empty",
                    "FlagQuantum measurement keys must be non-empty strings.",
                    index=index,
                    name=instruction.name,
                )
                blocked = True
            if len(instruction.wires) != 1:
                _issue(
                    issues,
                    "measurement_not_representable",
                    "Each FlagQuantum measure instruction must own exactly one wire.",
                    index=index,
                    name=instruction.name,
                )
                blocked = True
            if (
                not isinstance(classical_bit, int)
                or isinstance(classical_bit, bool)
                or classical_bit != next_classical_bit
            ):
                _issue(
                    issues,
                    "measurement_not_representable",
                    "FlagQuantum measurement classical bits must be circuit-global, "
                    "zero-based, contiguous, and ordered.",
                    index=index,
                    name=instruction.name,
                )
                blocked = True
            if blocked:
                continue
            if key != current_measurement_key:
                flush_measurement()
                if key in completed_measurement_keys:
                    _issue(
                        issues,
                        "duplicate_measurement_key",
                        f"FlagQuantum measurement key {key!r} forms multiple gates.",
                        index=index,
                        name=instruction.name,
                    )
                    continue
                current_measurement_key = key
            current_measurement_qubits.append(qubits[instruction.wires[0]])
            next_classical_bit += 1
            continue
        if measurement_started:
            flush_measurement()
            _issue(
                issues,
                "measurement_not_terminal",
                "FlagQuantum measurements must follow every non-measurement operation.",
                index=index,
                name=instruction.name,
            )
            continue
        if (
            extent_declared_by_gate
            and instruction.name == "any"
            and instruction.matrix is not None
        ):
            matrix = instruction.matrix.detach().cpu().numpy().copy()
            operation = cirq.MatrixGate(matrix).on(
                *(qubits[wire] for wire in instruction.wires)
            )
            if feedback is not None and clauses:
                operation = feedback.condition_operation(cirq, operation, clauses)
            operations.append(operation)
            continue
        if (
            extent_declared_by_gate
            and instruction.name == "kraus"
            and instruction.metadata.get("is_channel")
        ):
            from ._channel import _channel_operation

            operation = _channel_operation(instruction, qubits)
            if feedback is not None and clauses:
                operation = feedback.condition_operation(cirq, operation, clauses)
            operations.append(operation)
            continue
        if instruction.name not in supported_gates or instruction.matrix is not None:
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
        operation = gate.on(*(qubits[wire] for wire in instruction.wires))
        if feedback is not None and clauses:
            operation = feedback.condition_operation(cirq, operation, clauses)
        operations.append(operation)
    flush_measurement()
    if instruction_offsets is not None:
        instruction_offsets.append(len(operations))
    report = _report("to_cirq", str(cirq.__version__), issues)
    circuit = cirq.Circuit(cirq.Moment([operation]) for operation in operations)
    _raise_if_lossy(report, allow_lossy=allow_lossy)
    return CirqExportResult(circuit, report)


def to_cirq(program: Any, *, allow_lossy: bool = False) -> Any:
    return export_cirq(program, allow_lossy=allow_lossy).circuit


__all__ = ("export_cirq", "from_cirq", "import_cirq", "to_cirq")
