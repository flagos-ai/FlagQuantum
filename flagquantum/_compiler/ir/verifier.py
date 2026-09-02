"""Fail-closed structural and linear-value verifier for private QuantumIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flagquantum._compiler.diagnostics import (
    Diagnostic,
    DiagnosticCode,
    DiagnosticLocation,
)

from .modules import Block, QuantumModule, Region
from .operations import FrozenAttributes, Operation
from .schemas import OperationSchema, OperationSchemaRegistry, UnknownOperationError
from .types import QUBIT, IRType
from .values import ValueId, ValueRef


@dataclass(frozen=True)
class VerificationResult:
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    def require_valid(self) -> None:
        if self.diagnostics:
            codes = ", ".join(item.code.value for item in self.diagnostics)
            raise ValueError(f"QuantumIR verification failed: {codes}")


def _location(operation: Operation) -> DiagnosticLocation | None:
    if operation.location is None:
        return None
    return DiagnosticLocation(
        operation.location.source,
        operation.location.line,
        operation.location.column,
    )


def _diagnostic(
    code: DiagnosticCode,
    message: str,
    operation: Operation,
    *notes: str,
) -> Diagnostic:
    return Diagnostic(code, message, _location(operation), tuple(notes))


def _attribute_matches(value: Any, kinds: tuple[str, ...]) -> bool:
    for kind in kinds:
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "bool" and isinstance(value, bool):
            return True
        if kind == "float" and isinstance(value, float):
            return True
        if kind == "complex" and isinstance(value, complex):
            return True
        if kind == "static" and (
            value is None
            or isinstance(value, (bool, int, float, complex, tuple, FrozenAttributes))
        ):
            return True
        marker = getattr(value, "attribute_kind", None)
        if marker == kind and kind in {"symbolic", "binding"}:
            return True
    return False


def _check_schema_contract(
    operation: Operation, schema: OperationSchema
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    known_attributes = {item.name: item for item in schema.attributes}
    for name in operation.attributes:
        if name not in known_attributes:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.UNKNOWN_ATTRIBUTE,
                    f"operation {operation.name!r} has unknown attribute {name!r}",
                    operation,
                    "semantic attributes must be declared by the operation schema",
                )
            )
    for name, spec in known_attributes.items():
        if name not in operation.attributes:
            if spec.required:
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.MISSING_ATTRIBUTE,
                        f"operation {operation.name!r} requires attribute {name!r}",
                        operation,
                    )
                )
            continue
        value = operation.attributes[name]
        if not _attribute_matches(value, spec.kinds):
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                    f"attribute {name!r} does not match schema kinds {spec.kinds!r}",
                    operation,
                    f"received immutable value type {type(value).__name__}",
                )
            )

    if schema.variadic_qubit_arity:
        arity = operation.attributes.get("arity")
        if (
            not isinstance(arity, int)
            or isinstance(arity, bool)
            or arity <= 0
            or len(operation.operands) != arity
            or len(operation.results) != arity
        ):
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.INVALID_VARIADIC_ARITY,
                    f"operation {operation.name!r} has inconsistent variadic arity",
                    operation,
                    "arity must be a positive int matching operand and result counts",
                )
            )
    else:
        if len(operation.operands) != len(schema.operands):
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.OPERAND_ARITY_MISMATCH,
                    f"operation {operation.name!r} expects {len(schema.operands)} "
                    f"operands, received {len(operation.operands)}",
                    operation,
                )
            )
        if len(operation.results) != len(schema.results):
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.RESULT_ARITY_MISMATCH,
                    f"operation {operation.name!r} expects {len(schema.results)} "
                    f"results, received {len(operation.results)}",
                    operation,
                )
            )
    if len(operation.regions) != schema.region_count:
        diagnostics.append(
            _diagnostic(
                DiagnosticCode.REGION_COUNT_MISMATCH,
                f"operation {operation.name!r} expects {schema.region_count} regions, "
                f"received {len(operation.regions)}",
                operation,
            )
        )
    return diagnostics


def _expected_type(
    schema: OperationSchema, index: int, *, result: bool
) -> IRType | None:
    values = schema.results if result else schema.operands
    if schema.variadic_qubit_arity:
        return QUBIT
    return values[index] if index < len(values) else None


def _check_value_type(
    value: ValueRef,
    expected: IRType | None,
    operation: Operation,
    *,
    role: str,
    index: int,
) -> Diagnostic | None:
    if expected is None or value.type == expected:
        return None
    return _diagnostic(
        DiagnosticCode.VALUE_TYPE_MISMATCH,
        f"{role} {index} of {operation.name!r} has type {value.type.name!r}; "
        f"expected {expected.name!r}",
        operation,
    )


def _verify_block(block: Block, registry: OperationSchemaRegistry) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    definitions: dict[ValueId, IRType] = {}
    available_linear: set[ValueId] = set()
    released: set[ValueId] = set()
    terminated = False

    for argument in block.arguments:
        if argument.id in definitions:
            diagnostics.append(
                Diagnostic(
                    DiagnosticCode.DUPLICATE_DEFINITION,
                    f"block argument {argument.id} is defined more than once",
                )
            )
            continue
        definitions[argument.id] = argument.type
        if argument.linear:
            available_linear.add(argument.id)

    for operation in block.operations:
        if terminated:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.OPERATION_AFTER_TERMINATOR,
                    f"operation {operation.name!r} appears after a block terminator",
                    operation,
                )
            )
        try:
            schema = registry[operation.name]
        except UnknownOperationError:
            diagnostics.append(
                _diagnostic(
                    DiagnosticCode.UNKNOWN_OPERATION,
                    f"operation {operation.name!r} is not registered",
                    operation,
                )
            )
            continue

        diagnostics.extend(_check_schema_contract(operation, schema))

        for index, operand in enumerate(operation.operands):
            expected = _expected_type(schema, index, result=False)
            mismatch = _check_value_type(
                operand, expected, operation, role="operand", index=index
            )
            if mismatch:
                diagnostics.append(mismatch)
            if operand.id not in definitions:
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.USE_BEFORE_DEFINITION,
                        f"operand {operand.id} is used before definition",
                        operation,
                    )
                )
                continue
            if definitions[operand.id] != operand.type:
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.VALUE_TYPE_MISMATCH,
                        f"operand {operand.id} type disagrees with its definition",
                        operation,
                    )
                )
            if operand.linear:
                if operand.id in released:
                    diagnostics.append(
                        _diagnostic(
                            DiagnosticCode.USE_AFTER_RELEASE,
                            f"linear value {operand.id} is used after release",
                            operation,
                        )
                    )
                elif operand.id not in available_linear:
                    diagnostics.append(
                        _diagnostic(
                            DiagnosticCode.LINEAR_VALUE_REUSED,
                            f"linear value {operand.id} has already been consumed",
                            operation,
                        )
                    )
                else:
                    available_linear.remove(operand.id)
                    if "release" in schema.effects:
                        released.add(operand.id)

        for index, result in enumerate(operation.results):
            expected = _expected_type(schema, index, result=True)
            mismatch = _check_value_type(
                result, expected, operation, role="result", index=index
            )
            if mismatch:
                diagnostics.append(mismatch)
            if result.id in definitions:
                diagnostics.append(
                    _diagnostic(
                        DiagnosticCode.DUPLICATE_DEFINITION,
                        f"result {result.id} is defined more than once",
                        operation,
                    )
                )
                continue
            definitions[result.id] = result.type
            if result.linear:
                available_linear.add(result.id)

        if "terminator" in schema.effects:
            terminated = True

        for region in operation.regions:
            diagnostics.extend(_verify_region(region, registry))

    return diagnostics


def _verify_region(
    region: Region, registry: OperationSchemaRegistry
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for block in region.blocks:
        diagnostics.extend(_verify_block(block, registry))
    return diagnostics


def verify_module(
    module: QuantumModule, registry: OperationSchemaRegistry
) -> VerificationResult:
    """Verify a module without side effects, printing, or exception control flow."""

    return VerificationResult(tuple(_verify_region(module.body, registry)))


__all__ = ["VerificationResult", "verify_module"]
