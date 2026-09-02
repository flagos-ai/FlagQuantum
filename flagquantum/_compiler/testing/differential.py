"""Explicit, removable differential bridge for Phase 1 evidence only."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import torch

from flagquantum.core.ir import CircuitIR
from flagquantum.core.parameters import Parameter, ParameterExpression

from ..bindings import RuntimeBindingRef, SymbolicExpression, SymbolicParameter
from ..diagnostics import Diagnostic, DiagnosticCode
from ..exporters.circuit_ir import (
    SealedCircuitIRRoundTrip,
    export_circuit_ir,
    seal_circuit_ir_round_trip,
)
from ..ir.modules import QuantumModule
from ..ir.operations import FrozenAttributes
from ..ir.schemas import circuit_ir_v1_schema_registry
from ..ir.verifier import verify_module


class DifferentialStatus(str, Enum):
    READY = "ready"
    IMPORT_FAILED = "import_failed"
    LOWERING_FAILED = "lowering_failed"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True)
class ExecutionObservation:
    dtype: str | None = None
    device: str | None = None
    backend: str | None = None
    mode: str | None = None
    fallback: bool | None = None
    result_ordering: tuple[tuple[str, tuple[int, ...], str | None], ...] = ()
    failure: str | None = None


@dataclass(frozen=True)
class DifferentialReport:
    status: DifferentialStatus
    profile: str
    source_content_hash: str | None
    internal_program_identity: str | None
    candidate_content_hash: str | None
    binding_identity_preserved: bool
    requested_backend: str | None
    requested_mode: str | None
    legacy: ExecutionObservation
    candidate: ExecutionObservation
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is DifferentialStatus.READY


@dataclass(frozen=True)
class DifferentialExecution:
    report: DifferentialReport
    legacy_result: object | None = None
    candidate_result: object | None = None


@dataclass(frozen=True)
class DifferentialLoweringResult:
    circuit_ir: CircuitIR | None
    binding_identity_preserved: bool
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.circuit_ir is not None and not self.diagnostics


def _failure(message: str) -> Diagnostic:
    return Diagnostic(
        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
        message,
        notes=("Phase 1 differential lowering never falls back or repairs input",),
    )


def _validate_binding(reference: Any, value: Any) -> Diagnostic | None:
    shape = tuple(int(size) for size in getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", "")).removeprefix("torch.")
    if shape != reference.shape:
        return _failure(f"binding {reference.slot!r} shape changed")
    if dtype != reference.dtype:
        return _failure(f"binding {reference.slot!r} dtype changed")
    return None


class _LoweringError(Exception):
    pass


def _restore_attribute(
    value: object,
    artifact: SealedCircuitIRRoundTrip,
    consumed_bindings: set[str],
    *,
    evaluate_binding_expressions: bool = False,
) -> object:
    if isinstance(value, RuntimeBindingRef):
        try:
            bound = artifact.imported.bindings[value.slot]
            declared = next(
                reference
                for reference in artifact.imported.bindings.references
                if reference.slot == value.slot
            )
        except KeyError as exc:
            raise _LoweringError(f"binding {value.slot!r} is missing") from exc
        except StopIteration as exc:
            raise _LoweringError(
                f"binding descriptor {value.slot!r} is missing"
            ) from exc
        invalid = _validate_binding(declared, bound)
        if invalid is not None:
            raise _LoweringError(invalid.message)
        if declared != value:
            raise _LoweringError(f"binding {value.slot!r} descriptor changed")
        consumed_bindings.add(value.slot)
        return bound
    if isinstance(value, SymbolicParameter):
        return Parameter(value.name)
    if isinstance(value, SymbolicExpression):
        args = tuple(
            _restore_attribute(
                item,
                artifact,
                consumed_bindings,
                evaluate_binding_expressions=evaluate_binding_expressions,
            )
            for item in value.args
        )
        if evaluate_binding_expressions and not any(
            isinstance(item, (Parameter, ParameterExpression)) for item in args
        ):
            if value.op == "add":
                return args[0] + args[1]  # type: ignore[operator]
            if value.op == "sub":
                return args[0] - args[1]  # type: ignore[operator]
            if value.op == "mul":
                return args[0] * args[1]  # type: ignore[operator]
            if value.op == "neg":
                return -args[0]  # type: ignore[operator]
        return ParameterExpression(value.op, args)
    if isinstance(value, FrozenAttributes):
        if value.get("kind") != "tensor":
            raise _LoweringError("unsupported structured operation attribute")
        dtype = getattr(torch, str(value["dtype"]), None)
        if dtype is None:
            raise _LoweringError(f"unsupported tensor dtype {value['dtype']!r}")
        return torch.tensor(value["data"], dtype=dtype).reshape(tuple(value["shape"]))
    if value is None or isinstance(value, (bool, int, float, complex)):
        return value
    raise _LoweringError(f"unsupported operation attribute type {type(value).__name__}")


def _lower_operations(
    artifact: SealedCircuitIRRoundTrip,
    restored: CircuitIR,
    consumed_bindings: set[str],
    *,
    module: QuantumModule | None = None,
    allow_rewrites: bool = False,
) -> tuple[object, ...]:
    module = module or artifact.imported.module
    if len(module.body.blocks) != 1:
        raise _LoweringError("differential lowering requires one entry block")
    block = module.body.blocks[0]
    if not allow_rewrites and len(block.operations) != len(restored.instructions):
        raise _LoweringError("operation count changed after import")
    value_wires = {argument.id: wire for wire, argument in enumerate(block.arguments)}
    instructions = []
    for index, operation in enumerate(block.operations):
        if allow_rewrites:
            location = operation.location
            if (
                location is None
                or location.source != "CircuitIR.instructions"
                or location.line < 1
                or location.line > len(restored.instructions)
            ):
                raise _LoweringError(
                    "optimized operation lacks sealed CircuitIR source provenance"
                )
            skeleton = restored.instructions[location.line - 1]
        else:
            skeleton = restored.instructions[index]
        try:
            wires = tuple(value_wires[operand.id] for operand in operation.operands)
        except KeyError as exc:
            raise _LoweringError("operation consumes an unknown linear value") from exc
        if len(operation.results) != len(wires):
            raise _LoweringError("operation cannot preserve linear wire ownership")
        for result, wire in zip(operation.results, wires, strict=True):
            value_wires[result.id] = wire
        if operation.name == "quantum.custom_unitary":
            matrix = operation.attributes["matrix"]
            if not isinstance(matrix, FrozenAttributes):
                raise _LoweringError("custom unitary matrix attribute is invalid")
            dtype = getattr(torch, str(matrix["dtype"]), None)
            if dtype is None:
                raise _LoweringError("custom unitary matrix dtype is unsupported")
            instructions.append(
                replace(
                    skeleton,
                    name=str(operation.attributes["symbolic_name"]),
                    wires=wires,
                    params={},
                    matrix=torch.tensor(matrix["data"], dtype=dtype),
                )
            )
            continue
        if not operation.name.startswith("quantum."):
            raise _LoweringError(
                f"operation {operation.name!r} is outside the test lowering profile"
            )
        params = {
            key: _restore_attribute(
                value,
                artifact,
                consumed_bindings,
                evaluate_binding_expressions=allow_rewrites,
            )
            for key, value in operation.attributes.items()
        }
        instructions.append(
            replace(
                skeleton,
                name=operation.name.removeprefix("quantum."),
                wires=wires,
                params=params,
                matrix=None,
            )
        )
    return tuple(instructions)


def lower_sealed_for_differential(
    artifact: SealedCircuitIRRoundTrip,
) -> DifferentialLoweringResult:
    """Lower verified QuantumIR operations onto a sealed legacy source envelope."""

    exported = export_circuit_ir(artifact)
    if not exported.ok or exported.circuit_ir is None:
        return DifferentialLoweringResult(None, False, exported.diagnostics)
    restored = exported.circuit_ir
    consumed_bindings: set[str] = set()
    try:
        instructions = _lower_operations(artifact, restored, consumed_bindings)
    except (KeyError, _LoweringError) as exc:
        return DifferentialLoweringResult(None, False, (_failure(str(exc)),))
    observables = list(restored.observables)
    identity_preserved = True
    if len(observables) != len(artifact.imported.request.observables):
        return DifferentialLoweringResult(
            None, False, (_failure("observable count changed after import"),)
        )
    try:
        for index, request in enumerate(artifact.imported.request.observables):
            coefficient = _restore_attribute(
                request.coefficient, artifact, consumed_bindings
            )
            observables[index] = replace(
                observables[index],
                name=request.name,
                wires=request.wires,
                coefficient=coefficient,
            )
    except _LoweringError as exc:
        return DifferentialLoweringResult(None, False, (_failure(str(exc)),))
    expected_bindings = set(artifact.imported.bindings)
    if consumed_bindings != expected_bindings:
        return DifferentialLoweringResult(
            None, False, (_failure("runtime binding table contains unused slots"),)
        )
    for reference in artifact.imported.bindings.references:
        value = artifact.imported.bindings[reference.slot]
        if reference.slot.startswith("op"):
            operation_index, parameter = reference.slot[2:].split(".", 1)
            identity_preserved &= (
                instructions[int(operation_index)].params[parameter] is value
            )
        elif reference.slot.startswith("observable"):
            observable_index = int(reference.slot[10:].split(".", 1)[0])
            identity_preserved &= observables[observable_index].coefficient is value
    lowered = replace(
        restored,
        instructions=instructions,
        observables=tuple(observables),
    )
    return DifferentialLoweringResult(lowered, identity_preserved)


def lower_module_for_differential(
    artifact: SealedCircuitIRRoundTrip,
    module: QuantumModule,
) -> DifferentialLoweringResult:
    """Lower one verified Phase 2 module against its immutable source envelope."""

    verification = verify_module(module, circuit_ir_v1_schema_registry())
    if not verification.ok:
        return DifferentialLoweringResult(None, False, verification.diagnostics)
    exported = export_circuit_ir(artifact)
    if not exported.ok or exported.circuit_ir is None:
        return DifferentialLoweringResult(None, False, exported.diagnostics)
    restored = exported.circuit_ir
    consumed_bindings: set[str] = set()
    try:
        instructions = _lower_operations(
            artifact,
            restored,
            consumed_bindings,
            module=module,
            allow_rewrites=True,
        )
    except (KeyError, _LoweringError, TypeError) as exc:
        return DifferentialLoweringResult(None, False, (_failure(str(exc)),))
    observables = list(restored.observables)
    if len(observables) != len(artifact.imported.request.observables):
        return DifferentialLoweringResult(
            None, False, (_failure("observable count changed after import"),)
        )
    try:
        for index, request in enumerate(artifact.imported.request.observables):
            coefficient = _restore_attribute(
                request.coefficient,
                artifact,
                consumed_bindings,
                evaluate_binding_expressions=True,
            )
            observables[index] = replace(
                observables[index],
                name=request.name,
                wires=request.wires,
                coefficient=coefficient,
            )
    except _LoweringError as exc:
        return DifferentialLoweringResult(None, False, (_failure(str(exc)),))
    expected_bindings = set(artifact.imported.bindings)
    if consumed_bindings != expected_bindings:
        return DifferentialLoweringResult(
            None,
            False,
            (_failure("runtime binding table contains unused slots"),),
        )
    lowered = replace(
        restored,
        instructions=instructions,
        observables=tuple(observables),
    )
    return DifferentialLoweringResult(lowered, True)


def _first_tensor(result: object) -> Any | None:
    if hasattr(result, "dtype") and hasattr(result, "device"):
        return result
    for name in ("state", "value", "samples"):
        value = getattr(result, name, None)
        if hasattr(value, "dtype") and hasattr(value, "device"):
            return value
    for measurement in getattr(result, "measurements", ()):
        value = getattr(measurement, "value", None)
        if hasattr(value, "dtype") and hasattr(value, "device"):
            return value
    return None


def _observation(result: object | None, failure: str | None) -> ExecutionObservation:
    if result is None:
        return ExecutionObservation(failure=failure)
    tensor = _first_tensor(result)
    runtime = getattr(result, "runtime", {})
    if not isinstance(runtime, Mapping):
        runtime = {}
    ordering = tuple(
        (
            str(getattr(item, "kind", "")),
            tuple(int(wire) for wire in getattr(item, "wires", ())),
            (
                str(getattr(item, "metadata", {}).get("name"))
                if getattr(item, "metadata", {}).get("name") is not None
                else None
            ),
        )
        for item in getattr(result, "measurements", ())
    )
    fallback_raw = runtime.get("fallback")
    return ExecutionObservation(
        dtype=(
            str(tensor.dtype).removeprefix("torch.") if tensor is not None else None
        ),
        device=(str(tensor.device) if tensor is not None else None),
        backend=(None if runtime.get("backend") is None else str(runtime["backend"])),
        mode=(None if runtime.get("mode") is None else str(runtime["mode"])),
        fallback=(None if fallback_raw is None else bool(fallback_raw)),
        result_ordering=ordering,
        failure=failure,
    )


def execute_differential(
    source: CircuitIR,
    executor: Callable[[CircuitIR], object],
    *,
    requested_backend: str | None = None,
    requested_mode: str | None = None,
) -> DifferentialExecution:
    """Execute legacy and explicit internal-test paths without global switches."""

    sealed = seal_circuit_ir_round_trip(source)
    if not sealed.ok or sealed.artifact is None:
        report = DifferentialReport(
            DifferentialStatus.IMPORT_FAILED,
            "circuit_ir_v1_static",
            source.content_hash,
            None,
            None,
            False,
            requested_backend,
            requested_mode,
            ExecutionObservation(),
            ExecutionObservation(),
            sealed.diagnostics,
        )
        return DifferentialExecution(report)
    lowered = lower_sealed_for_differential(sealed.artifact)
    if not lowered.ok or lowered.circuit_ir is None:
        report = DifferentialReport(
            DifferentialStatus.LOWERING_FAILED,
            sealed.artifact.profile,
            source.content_hash,
            sealed.artifact.internal_program_identity,
            None,
            lowered.binding_identity_preserved,
            requested_backend,
            requested_mode,
            ExecutionObservation(),
            ExecutionObservation(),
            lowered.diagnostics,
        )
        return DifferentialExecution(report)
    legacy_result: object | None = None
    candidate_result: object | None = None
    legacy_failure: str | None = None
    candidate_failure: str | None = None
    try:
        legacy_result = executor(source)
    except Exception as exc:  # noqa: BLE001 - failure is evidence, not control flow
        legacy_failure = f"{type(exc).__name__}: {exc}"
    try:
        candidate_result = executor(lowered.circuit_ir)
    except Exception as exc:  # noqa: BLE001 - failure is evidence, not control flow
        candidate_failure = f"{type(exc).__name__}: {exc}"
    status = (
        DifferentialStatus.READY
        if legacy_failure is None and candidate_failure is None
        else DifferentialStatus.EXECUTION_FAILED
    )
    report = DifferentialReport(
        status,
        sealed.artifact.profile,
        source.content_hash,
        sealed.artifact.internal_program_identity,
        lowered.circuit_ir.content_hash,
        lowered.binding_identity_preserved,
        requested_backend,
        requested_mode,
        _observation(legacy_result, legacy_failure),
        _observation(candidate_result, candidate_failure),
    )
    return DifferentialExecution(report, legacy_result, candidate_result)


__all__ = [
    "DifferentialExecution",
    "DifferentialLoweringResult",
    "DifferentialReport",
    "DifferentialStatus",
    "ExecutionObservation",
    "execute_differential",
    "lower_module_for_differential",
    "lower_sealed_for_differential",
]
