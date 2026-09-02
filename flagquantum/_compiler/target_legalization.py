"""Deterministic, fail-closed QuantumIR-to-TargetIR legalization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

from .bindings import RuntimeBindingRef, SymbolicExpression, SymbolicParameter
from .diagnostics import Diagnostic, DiagnosticCode
from .ir.modules import QuantumModule
from .ir.operations import FrozenAttributes
from .ir.schemas import circuit_ir_v1_schema_registry
from .ir.types import QUBIT
from .ir.verifier import verify_module
from .target_capabilities import (
    GateCapability,
    MeasurementResult,
    ParameterConstraint,
    TargetCapabilities,
    TargetClass,
)
from .target_ir import TargetIR, TargetOperation


class TargetLegalizationStatus(str, Enum):
    LEGALIZED = "legalized"
    UNSUPPORTED = "unsupported"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class TargetLegalizationResult:
    status: TargetLegalizationStatus
    target_ir: TargetIR | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is TargetLegalizationStatus.LEGALIZED


def _failure(
    status: TargetLegalizationStatus,
    code: DiagnosticCode,
    message: str,
) -> TargetLegalizationResult:
    return TargetLegalizationResult(
        status,
        diagnostics=(
            Diagnostic(
                code,
                message,
                notes=("Batch B legalization never repairs or weakens capabilities",),
            ),
        ),
    )


def _static_scalar(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, FrozenAttributes) and value.get("kind") == "tensor":
        shape = tuple(value.get("shape", ()))
        data = value.get("data")
        if (
            shape == ()
            and isinstance(data, (int, float))
            and not isinstance(data, bool)
        ):
            return float(data)
    return None


def _parameter_is_dynamic(value: object) -> bool:
    return isinstance(
        value, (RuntimeBindingRef, SymbolicParameter, SymbolicExpression)
    ) or (
        isinstance(value, FrozenAttributes) and value.get("kind") == "runtime_binding"
    )


def _constraint_accepts(value: float, constraint: ParameterConstraint) -> bool:
    if constraint.minimum is not None and value < constraint.minimum:
        return False
    if constraint.maximum is not None and value > constraint.maximum:
        return False
    return True


def _validate_parameters(
    attributes: FrozenAttributes,
    gate: GateCapability,
    target: TargetCapabilities,
) -> str | None:
    schema = OPERATOR_SCHEMAS[gate.operation]
    unknown = set(attributes) - set(schema.parameters)
    if unknown:
        return (
            f"operation {gate.operation!r} has unknown attributes {sorted(unknown)!r}"
        )
    missing = set(schema.parameters) - set(attributes)
    if missing:
        return f"operation {gate.operation!r} lacks parameters {sorted(missing)!r}"
    constraints = {item.name: item for item in gate.parameters}
    for name in schema.parameters:
        value = attributes[name]
        if _parameter_is_dynamic(value):
            if not target.supports_parameter_binding:
                return f"operation {gate.operation!r} requires unsupported parameter binding"
            if name in constraints:
                return f"dynamic parameter {name!r} cannot be proven inside the target domain"
            continue
        scalar = _static_scalar(value)
        if scalar is None:
            return f"operation {gate.operation!r} parameter {name!r} is not scalar"
        constraint = constraints.get(name)
        if constraint is not None and not _constraint_accepts(scalar, constraint):
            return f"operation {gate.operation!r} parameter {name!r} is outside target domain"
    return None


def _validate_layout(
    logical_count: int,
    target: TargetCapabilities,
    requested: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if logical_count > target.logical_qubit_capacity:
        raise ValueError("program exceeds logical qubit capacity")
    layout = tuple(range(logical_count)) if requested is None else tuple(requested)
    if len(layout) != logical_count:
        raise ValueError("layout length must equal the program logical-qubit count")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in layout):
        raise ValueError("layout entries must be integers")
    if len(layout) != len(set(layout)):
        raise ValueError("layout physical qubits must be unique")
    if any(item < 0 or item >= target.physical_qubit_capacity for item in layout):
        raise ValueError("layout physical qubit is outside target capacity")
    return layout


def legalize_quantum_module(
    module: QuantumModule,
    target: TargetCapabilities,
    *,
    logical_to_physical: tuple[int, ...] | None = None,
    required_results: tuple[MeasurementResult, ...] = (),
    requested_shots: int | None = None,
) -> TargetLegalizationResult:
    """Legalize without decomposition, routing, provider lookup, or fallback."""

    if not isinstance(module, QuantumModule) or not isinstance(
        target, TargetCapabilities
    ):
        return _failure(
            TargetLegalizationStatus.INVALID_INPUT,
            DiagnosticCode.VALUE_TYPE_MISMATCH,
            "legalization requires QuantumModule and TargetCapabilities",
        )
    verification = verify_module(module, circuit_ir_v1_schema_registry())
    if not verification.ok:
        return TargetLegalizationResult(
            TargetLegalizationStatus.INVALID_INPUT,
            diagnostics=verification.diagnostics,
        )
    if len(module.body.blocks) != 1:
        return _failure(
            TargetLegalizationStatus.UNSUPPORTED,
            DiagnosticCode.REGION_COUNT_MISMATCH,
            "Batch B TargetIR requires one block",
        )
    block = module.body.blocks[0]
    if any(argument.type != QUBIT for argument in block.arguments):
        return _failure(
            TargetLegalizationStatus.UNSUPPORTED,
            DiagnosticCode.VALUE_TYPE_MISMATCH,
            "Batch B TargetIR supports qubit block arguments only",
        )
    try:
        layout = _validate_layout(len(block.arguments), target, logical_to_physical)
    except ValueError as exc:
        return _failure(
            TargetLegalizationStatus.UNSUPPORTED,
            DiagnosticCode.VALUE_TYPE_MISMATCH,
            str(exc),
        )
    results = tuple(required_results)
    if any(not isinstance(item, MeasurementResult) for item in results):
        return _failure(
            TargetLegalizationStatus.INVALID_INPUT,
            DiagnosticCode.VALUE_TYPE_MISMATCH,
            "required results must use MeasurementResult",
        )
    if len(results) != len(set(results)):
        return _failure(
            TargetLegalizationStatus.INVALID_INPUT,
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "required results must be unique",
        )
    missing_results = set(results) - set(target.measurement_results)
    if missing_results:
        return _failure(
            TargetLegalizationStatus.UNSUPPORTED,
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            f"target lacks required results {sorted(item.value for item in missing_results)!r}",
        )
    if requested_shots is not None and (
        isinstance(requested_shots, bool)
        or not isinstance(requested_shots, int)
        or requested_shots <= 0
    ):
        return _failure(
            TargetLegalizationStatus.INVALID_INPUT,
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "requested shots must be a positive integer",
        )
    if requested_shots is not None and (
        target.maximum_shots is None or requested_shots > target.maximum_shots
    ):
        return _failure(
            TargetLegalizationStatus.UNSUPPORTED,
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "requested shots exceed or cannot be proven against the target limit",
        )
    if target.maximum_program_operations is None or (
        len(block.operations) > target.maximum_program_operations
    ):
        return _failure(
            TargetLegalizationStatus.UNSUPPORTED,
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "program exceeds the target operation limit",
        )
    gates = {item.operation: item for item in target.native_gates}
    logical_by_value = {
        argument.id: index for index, argument in enumerate(block.arguments)
    }
    output = []
    for operation in block.operations:
        opcode = operation.name.removeprefix("quantum.")
        gate = gates.get(opcode)
        if gate is None:
            return _failure(
                TargetLegalizationStatus.UNSUPPORTED,
                DiagnosticCode.UNKNOWN_OPERATION,
                f"operation {operation.name!r} is not native for the target",
            )
        try:
            logicals = tuple(logical_by_value[item.id] for item in operation.operands)
        except KeyError:
            return _failure(
                TargetLegalizationStatus.INVALID_INPUT,
                DiagnosticCode.USE_BEFORE_DEFINITION,
                f"operation {operation.name!r} consumes an unknown value",
            )
        physicals = tuple(layout[item] for item in logicals)
        if len(physicals) != len(set(physicals)):
            return _failure(
                TargetLegalizationStatus.UNSUPPORTED,
                DiagnosticCode.OPERAND_ARITY_MISMATCH,
                f"operation {operation.name!r} aliases a physical qubit",
            )
        if len(physicals) > 1:
            if target.topology is None:
                if target.target_class is not TargetClass.LOCAL_RUNTIME:
                    return _failure(
                        TargetLegalizationStatus.UNSUPPORTED,
                        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                        "non-local multi-qubit legality requires explicit topology",
                    )
            elif len(physicals) != 2:
                return _failure(
                    TargetLegalizationStatus.UNSUPPORTED,
                    DiagnosticCode.OPERAND_ARITY_MISMATCH,
                    "Batch B topology legality supports at most two-qubit operations",
                )
            elif not target.topology.has_edge(*physicals):
                return _failure(
                    TargetLegalizationStatus.UNSUPPORTED,
                    DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                    f"directed target edge {physicals[0]}->{physicals[1]} is unavailable",
                )
        parameter_error = _validate_parameters(operation.attributes, gate, target)
        if parameter_error is not None:
            return _failure(
                TargetLegalizationStatus.UNSUPPORTED,
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                parameter_error,
            )
        output.append(
            TargetOperation(
                opcode,
                physicals,
                operation.attributes,
                location=operation.location,
            )
        )
        for result, logical in zip(operation.results, logicals, strict=True):
            logical_by_value[result.id] = logical
    target_ir = TargetIR(
        module.program_identity,
        target.semantic_fingerprint,
        layout,
        tuple(output),
        results,
        requested_shots,
    )
    return TargetLegalizationResult(TargetLegalizationStatus.LEGALIZED, target_ir)


__all__ = [
    "TargetLegalizationResult",
    "TargetLegalizationStatus",
    "legalize_quantum_module",
]
