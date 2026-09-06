"""Loss-aware conversion between Qiskit circuits and FlagQuantum IR."""

from __future__ import annotations

from importlib import import_module
from numbers import Number
from typing import Any, Literal, Mapping

import torch

from ...core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ...core.operator_schema import canonical_opcode, get_operator_schema
from ...core.parameters import Parameter, ParameterExpression
from .models import (
    QiskitConversionError,
    QiskitConversionIssue,
    QiskitConversionReport,
    QiskitDependencyError,
    QiskitExportResult,
    QiskitImportResult,
)

_QISKIT_TO_FLAGQUANTUM = {
    "id": "i",
    "p": "phase",
    "u": "u3",
    "cp": "cphase",
}
_FLAGQUANTUM_TO_QISKIT = {
    "i": "id",
    "phase": "p",
    "u3": "u",
    "cphase": "cp",
}


def _qiskit_modules() -> tuple[Any, Any, Any]:
    try:
        qiskit = import_module("qiskit")
        circuit_module = import_module("qiskit.circuit")
        library_module = import_module("qiskit.circuit.library")
    except ImportError as exc:
        raise QiskitDependencyError(
            "Qiskit interoperability requires the optional dependency; install "
            "it with `pip install 'flagquantum[qiskit]'`."
        ) from exc
    return qiskit, circuit_module, library_module


def _report(
    direction: Literal["from_qiskit", "to_qiskit"],
    version: str | None,
    issues: list[QiskitConversionIssue],
) -> QiskitConversionReport:
    return QiskitConversionReport(direction, version, tuple(issues))


def _raise_if_lossy(report: QiskitConversionReport, *, allow_lossy: bool) -> None:
    if allow_lossy or report.lossless:
        return
    rendered = ", ".join(issue.code for issue in report.issues)
    raise QiskitConversionError(
        "Qiskit conversion is not lossless; inspect error.report or explicitly "
        f"set allow_lossy=True. Issues: {rendered}",
        report,
    )


def _numeric_parameter(
    value: Any,
    *,
    parameter_type: type[Any],
    expression_type: type[Any],
    issues: list[QiskitConversionIssue],
    operation_index: int,
    operation_name: str,
) -> Any | None:
    if isinstance(value, parameter_type):
        return Parameter(str(value.name))
    if isinstance(value, expression_type):
        issues.append(
            QiskitConversionIssue(
                "unsupported_parameter_expression",
                "Qiskit ParameterExpression is not representable by the FlagQuantum "
                "v1 arithmetic expression subset; bind or simplify it first.",
                "error",
                operation_index,
                operation_name,
            )
        )
        return None
    try:
        converted = complex(value)
    except (TypeError, ValueError):
        issues.append(
            QiskitConversionIssue(
                "unsupported_parameter_value",
                f"parameter value {value!r} is not a real scalar",
                "error",
                operation_index,
                operation_name,
            )
        )
        return None
    if converted.imag != 0:
        issues.append(
            QiskitConversionIssue(
                "complex_gate_parameter",
                f"parameter value {value!r} is complex",
                "error",
                operation_index,
                operation_name,
            )
        )
        return None
    return float(converted.real)


def _matrix_from_operation(operation: Any) -> torch.Tensor | None:
    try:
        matrix = operation.to_matrix()
    except (AttributeError, TypeError, ValueError):
        return None
    try:
        return torch.as_tensor(matrix)
    except (TypeError, ValueError):
        return None


def import_qiskit(circuit: Any, *, allow_lossy: bool = False) -> QiskitImportResult:
    """Convert a Qiskit ``QuantumCircuit`` into canonical FlagQuantum IR.

    Unsupported operations are never silently approximated. The default is
    fail-closed; ``allow_lossy=True`` returns a deliberately partial IR and a
    machine-readable report describing every skipped operation.
    """

    qiskit, circuit_module, _library_module = _qiskit_modules()
    quantum_circuit_type = circuit_module.QuantumCircuit
    parameter_type = circuit_module.Parameter
    expression_type = circuit_module.ParameterExpression
    if not isinstance(circuit, quantum_circuit_type):
        raise TypeError("from_qiskit requires qiskit.QuantumCircuit")
    if int(circuit.num_qubits) <= 0:
        raise ValueError("Qiskit circuit must contain at least one qubit")

    issues: list[QiskitConversionIssue] = []
    instructions: list[Instruction] = []
    qregs = tuple(
        (str(register.name), int(register.size)) for register in circuit.qregs
    )
    cregs = tuple(
        (str(register.name), int(register.size)) for register in circuit.cregs
    )
    standard_qregs = (("q", int(circuit.num_qubits)),)
    standard_cregs = (("c", int(circuit.num_clbits)),) if circuit.num_clbits else ()
    if qregs != standard_qregs or cregs != standard_cregs:
        issues.append(
            QiskitConversionIssue(
                "register_layout_flattened",
                "named, multiple, or aliased Qiskit registers are flattened to stable "
                "qubit and classical-bit indices in FlagQuantum IR v1.",
                "warning",
            )
        )
    metadata_keys = set(circuit.metadata or {})
    flagquantum_metadata_keys = {
        "flagquantum_interop",
        "flagquantum_ir_hash",
        "flagquantum_ir_version",
    }
    unknown_metadata_keys = metadata_keys - flagquantum_metadata_keys
    if unknown_metadata_keys:
        issues.append(
            QiskitConversionIssue(
                "circuit_metadata_dropped",
                "arbitrary Qiskit circuit metadata is not copied into executable "
                "FlagQuantum IR metadata; unsupported keys: "
                + ", ".join(sorted(unknown_metadata_keys)),
                "warning",
            )
        )
    for index, item in enumerate(circuit.data):
        operation = item.operation
        raw_name = str(operation.name).lower()
        wires = tuple(int(circuit.find_bit(bit).index) for bit in item.qubits)
        classical_bits = tuple(int(circuit.find_bit(bit).index) for bit in item.clbits)
        if raw_name == "barrier":
            issues.append(
                QiskitConversionIssue(
                    "barrier_dropped",
                    "Qiskit barriers constrain later compilation and are not part of "
                    "FlagQuantum IR v1.",
                    "warning",
                    index,
                    raw_name,
                )
            )
            continue
        if raw_name == "measure":
            if len(wires) != 1 or len(classical_bits) != 1:
                issues.append(
                    QiskitConversionIssue(
                        "invalid_measurement_shape",
                        "measurement must map one qubit to one classical bit",
                        "error",
                        index,
                        raw_name,
                    )
                )
                continue
            instructions.append(
                Instruction(
                    "measure",
                    wires,
                    metadata={
                        "is_dynamic": True,
                        "classical_bit": classical_bits[0],
                    },
                )
            )
            continue
        if raw_name == "reset":
            instructions.append(
                Instruction("reset", wires, metadata={"is_dynamic": True})
            )
            continue
        if getattr(operation, "label", None) is not None and raw_name != "unitary":
            issues.append(
                QiskitConversionIssue(
                    "operation_label_dropped",
                    "Qiskit operation labels are not represented for built-in "
                    "FlagQuantum opcodes.",
                    "warning",
                    index,
                    raw_name,
                )
            )
        if getattr(operation, "condition", None) is not None or hasattr(
            operation, "blocks"
        ):
            issues.append(
                QiskitConversionIssue(
                    "unsupported_control_flow",
                    "Qiskit control flow must be lowered to the supported dynamic "
                    "FlagQuantum subset before conversion.",
                    "error",
                    index,
                    raw_name,
                )
            )
            continue

        name = canonical_opcode(_QISKIT_TO_FLAGQUANTUM.get(raw_name, raw_name))
        schema = get_operator_schema(name)
        if schema is None:
            matrix = _matrix_from_operation(operation)
            if matrix is None:
                issues.append(
                    QiskitConversionIssue(
                        "unsupported_operation",
                        f"Qiskit operation {raw_name!r} has no FlagQuantum opcode or "
                        "static unitary matrix.",
                        "error",
                        index,
                        raw_name,
                    )
                )
                continue
            if len(wires) != 1:
                issues.append(
                    QiskitConversionIssue(
                        "multi_qubit_unitary_wire_order_unverified",
                        "custom multi-qubit matrices require an explicit Qiskit-to-"
                        "FlagQuantum basis-order conversion that IR v1 does not yet "
                        "define.",
                        "error",
                        index,
                        raw_name,
                    )
                )
                continue
            instructions.append(
                Instruction(
                    raw_name,
                    wires,
                    matrix=matrix,
                    metadata={
                        "interop_source": "qiskit",
                        "qiskit_label": getattr(operation, "label", None),
                    },
                )
            )
            continue
        if len(operation.params) != len(schema.parameters):
            issues.append(
                QiskitConversionIssue(
                    "parameter_arity_mismatch",
                    f"operation {raw_name!r} exposes {len(operation.params)} "
                    f"parameter(s), expected {len(schema.parameters)}",
                    "error",
                    index,
                    raw_name,
                )
            )
            continue
        converted = [
            _numeric_parameter(
                value,
                parameter_type=parameter_type,
                expression_type=expression_type,
                issues=issues,
                operation_index=index,
                operation_name=raw_name,
            )
            for value in operation.params
        ]
        if any(value is None for value in converted):
            continue
        instructions.append(
            Instruction(name, wires, params=dict(zip(schema.parameters, converted)))
        )

    global_phase: float | None
    converted_phase = _numeric_parameter(
        circuit.global_phase,
        parameter_type=parameter_type,
        expression_type=expression_type,
        issues=issues,
        operation_index=-1,
        operation_name="global_phase",
    )
    global_phase = float(converted_phase) if converted_phase is not None else None
    report = _report("from_qiskit", str(qiskit.__version__), issues)
    metadata: dict[str, Any] = {
        "interop": {
            "source": "qiskit",
            "framework_version": str(qiskit.__version__),
            "num_clbits": int(circuit.num_clbits),
            "circuit_name": str(circuit.name),
            "qregs": qregs,
            "cregs": cregs,
            "global_phase": global_phase,
            "flagquantum_source_provenance": {
                key: circuit.metadata[key]
                for key in sorted(metadata_keys & flagquantum_metadata_keys)
            },
            "conversion_report": report.to_dict(),
        }
    }
    ir = CircuitIR(
        n_wires=int(circuit.num_qubits),
        instructions=tuple(instructions),
        dtype="complex64",
        shape=(1, 2 ** int(circuit.num_qubits)),
        metadata=metadata,
    )
    _raise_if_lossy(report, allow_lossy=allow_lossy)
    return QiskitImportResult(ir, report)


def from_qiskit(circuit: Any, *, allow_lossy: bool = False) -> CircuitIR:
    """Return FlagQuantum IR for a Qiskit circuit."""

    return import_qiskit(circuit, allow_lossy=allow_lossy).ir


def _qiskit_parameter(
    value: Any,
    *,
    parameter_type: type[Any],
    cache: dict[str, Any],
) -> Any:
    if isinstance(value, Parameter):
        return cache.setdefault(value.name, parameter_type(value.name))
    if isinstance(value, ParameterExpression):
        args = tuple(
            _qiskit_parameter(item, parameter_type=parameter_type, cache=cache)
            for item in value.args
        )
        if value.op == "add":
            return args[0] + args[1]
        if value.op == "sub":
            return args[0] - args[1]
        if value.op == "mul":
            return args[0] * args[1]
        if value.op == "neg":
            return -args[0]
        raise ValueError(f"unsupported FlagQuantum parameter expression {value.op!r}")
    if isinstance(value, Number) or hasattr(value, "item"):
        return value.item() if hasattr(value, "item") else value
    raise TypeError(f"unsupported FlagQuantum parameter value {value!r}")


def _condition_pairs(instruction: Instruction) -> tuple[tuple[int, int], ...]:
    if "conditions" in instruction.metadata:
        return tuple(
            (int(bit), int(value)) for bit, value in instruction.metadata["conditions"]
        )
    legacy = instruction.metadata.get("condition")
    if isinstance(legacy, Mapping):
        return ((int(legacy["bit"]), int(legacy["equals"])),)
    return ()


def export_qiskit(program: Any, *, allow_lossy: bool = False) -> QiskitExportResult:
    """Convert FlagQuantum IR (or an object exposing ``to_ir``) to Qiskit."""

    qiskit, circuit_module, _library_module = _qiskit_modules()
    quantum_circuit_type = circuit_module.QuantumCircuit
    parameter_type = circuit_module.Parameter
    ir = ensure_circuit_ir(program)
    interop_metadata = ir.metadata.get("interop", {})
    if not isinstance(interop_metadata, Mapping):
        interop_metadata = {}
    width = int(interop_metadata.get("num_clbits", 0) or 0)
    for instruction in ir.instructions:
        if instruction.name == "measure":
            width = max(width, int(instruction.metadata["classical_bit"]) + 1)
        for bit, _expected in _condition_pairs(instruction):
            width = max(width, bit + 1)
    circuit_name = interop_metadata.get("circuit_name")
    circuit = quantum_circuit_type(
        ir.n_wires,
        width,
        name=str(circuit_name) if circuit_name is not None else None,
    )
    global_phase = interop_metadata.get("global_phase")
    if global_phase is not None:
        circuit.global_phase = global_phase

    issues: list[QiskitConversionIssue] = []
    if ir.observables:
        issues.append(
            QiskitConversionIssue(
                "observables_not_embedded",
                "CircuitIR observables are execution requests and cannot be embedded "
                "in a Qiskit QuantumCircuit.",
                "error",
            )
        )
    if ir.measurements:
        issues.append(
            QiskitConversionIssue(
                "measurement_requests_not_embedded",
                "CircuitIR measurement requests require a separate Qiskit execution "
                "plan and cannot be embedded in a QuantumCircuit.",
                "error",
            )
        )
    parameter_cache: dict[str, Any] = {}

    def apply(instruction: Instruction, index: int) -> bool:
        if instruction.name == "measure":
            circuit.measure(
                instruction.wires[0], int(instruction.metadata["classical_bit"])
            )
            return True
        if instruction.name == "reset":
            circuit.reset(instruction.wires[0])
            return True
        if instruction.matrix is not None:
            if len(instruction.wires) != 1:
                issues.append(
                    QiskitConversionIssue(
                        "multi_qubit_unitary_wire_order_unverified",
                        "custom multi-qubit matrices require an explicit FlagQuantum-"
                        "to-Qiskit basis-order conversion that IR v1 does not yet "
                        "define.",
                        "error",
                        index,
                        instruction.name,
                    )
                )
                return False
            matrix = instruction.matrix
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            circuit.unitary(
                matrix,
                list(instruction.wires),
                label=instruction.metadata.get("qiskit_label") or instruction.name,
            )
            return True
        schema = get_operator_schema(instruction.name)
        method_name = _FLAGQUANTUM_TO_QISKIT.get(instruction.name, instruction.name)
        method = getattr(circuit, method_name, None)
        if schema is None or method is None:
            issues.append(
                QiskitConversionIssue(
                    "unsupported_operation",
                    f"FlagQuantum operation {instruction.name!r} has no Qiskit "
                    "lowering.",
                    "error",
                    index,
                    instruction.name,
                )
            )
            return False
        try:
            params = tuple(
                _qiskit_parameter(
                    instruction.params[name],
                    parameter_type=parameter_type,
                    cache=parameter_cache,
                )
                for name in schema.parameters
            )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                QiskitConversionIssue(
                    "unsupported_parameter_expression",
                    str(exc),
                    "error",
                    index,
                    instruction.name,
                )
            )
            return False
        method(*params, *instruction.wires)
        return True

    for index, instruction in enumerate(ir.instructions):
        conditions = _condition_pairs(instruction)
        if not conditions:
            apply(instruction, index)
            continue
        if len(conditions) != 1:
            issues.append(
                QiskitConversionIssue(
                    "unsupported_multi_bit_condition",
                    "Qiskit export currently supports one classical condition bit.",
                    "error",
                    index,
                    instruction.name,
                )
            )
            continue
        bit, expected = conditions[0]
        if expected not in {0, 1}:
            issues.append(
                QiskitConversionIssue(
                    "unsupported_condition_value",
                    "Qiskit export requires a one-bit Boolean condition.",
                    "error",
                    index,
                    instruction.name,
                )
            )
            continue
        with circuit.if_test((circuit.clbits[bit], bool(expected))):
            apply(instruction, index)

    report = _report("to_qiskit", str(qiskit.__version__), issues)
    circuit.metadata = {
        **(dict(circuit.metadata) if circuit.metadata else {}),
        "flagquantum_interop": report.to_dict(),
        "flagquantum_ir_version": ir.version,
        "flagquantum_ir_hash": ir.content_hash,
    }
    _raise_if_lossy(report, allow_lossy=allow_lossy)
    return QiskitExportResult(circuit, report)


def to_qiskit(program: Any, *, allow_lossy: bool = False) -> Any:
    """Return a Qiskit circuit for FlagQuantum IR or ``fq.Circuit``."""

    return export_qiskit(program, allow_lossy=allow_lossy).circuit


__all__ = ("export_qiskit", "from_qiskit", "import_qiskit", "to_qiskit")
