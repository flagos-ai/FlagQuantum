"""Restricted exact CircuitIR round-trip exporter for Phase 1 evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum

import torch

from flagquantum.core.ir import CircuitIR, IRSerializationError, IRValidationError
from flagquantum.core.parameters import Parameter, ParameterExpression

from ..bindings import RuntimeBindingRef, SymbolicExpression, SymbolicParameter
from ..diagnostics import Diagnostic, DiagnosticCode
from ..import_models import ImportedCircuitProgram, ImportStatus
from ..importers.circuit_ir import import_circuit_ir
from ..ir.modules import QuantumModule
from ..ir.operations import FrozenAttributes
from ..ir.schemas import circuit_ir_v1_schema_registry
from ..ir.verifier import verify_module


class ExportStatus(str, Enum):
    EXPORTED_EXACT = "exported_exact"
    UNSUPPORTED_WITH_DIAGNOSTICS = "unsupported_with_diagnostics"
    INVALID_ARTIFACT = "invalid_artifact"


@dataclass(frozen=True)
class SealedCircuitIRRoundTrip:
    """Source envelope bound to the exact internal program produced at import."""

    imported: ImportedCircuitProgram
    canonical_source_json: str
    source_content_hash: str
    internal_program_identity: str
    profile: str = "circuit_ir_v1_static"


@dataclass(frozen=True)
class RoundTripSealResult:
    status: ImportStatus
    artifact: SealedCircuitIRRoundTrip | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is ImportStatus.SUPPORTED_EXACT


@dataclass(frozen=True)
class CircuitExportResult:
    status: ExportStatus
    circuit_ir: CircuitIR | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is ExportStatus.EXPORTED_EXACT


def _failure(message: str) -> CircuitExportResult:
    diagnostic = Diagnostic(
        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
        message,
        notes=("restricted exporter never performs lossy recovery",),
    )
    return CircuitExportResult(
        ExportStatus.INVALID_ARTIFACT,
        diagnostics=(diagnostic,),
    )


def seal_circuit_ir_round_trip(source: object) -> RoundTripSealResult:
    """Import and seal a source payload only when Batch C supports it exactly."""

    imported_result = import_circuit_ir(source)
    if not imported_result.ok or imported_result.imported is None:
        return RoundTripSealResult(
            imported_result.status,
            diagnostics=imported_result.diagnostics,
        )
    if not isinstance(source, CircuitIR):
        return RoundTripSealResult(
            ImportStatus.INVALID_INPUT,
            diagnostics=(
                Diagnostic(
                    DiagnosticCode.VALUE_TYPE_MISMATCH,
                    "round-trip sealing requires CircuitIR",
                ),
            ),
        )
    imported = imported_result.imported
    artifact = SealedCircuitIRRoundTrip(
        imported,
        source.to_json(),
        source.content_hash,
        imported.internal_program_identity,
    )
    return RoundTripSealResult(ImportStatus.SUPPORTED_EXACT, artifact)


def export_circuit_ir(artifact: object) -> CircuitExportResult:
    """Restore a sealed source only after every identity boundary is verified."""

    if not isinstance(artifact, SealedCircuitIRRoundTrip):
        return _failure("restricted CircuitIR exporter requires a sealed artifact")
    if artifact.profile != "circuit_ir_v1_static":
        return _failure(f"unsupported round-trip profile {artifact.profile!r}")
    if (
        artifact.imported.internal_program_identity
        != artifact.internal_program_identity
    ):
        return _failure("internal program identity changed after sealing")
    if artifact.imported.source.circuit_ir_content_hash != artifact.source_content_hash:
        return _failure("imported source identity does not match sealed source")
    encoded_hash = hashlib.sha256(
        artifact.canonical_source_json.encode("utf-8")
    ).hexdigest()
    if encoded_hash != artifact.source_content_hash:
        return _failure("sealed canonical source payload hash mismatch")
    try:
        restored = CircuitIR.from_json(artifact.canonical_source_json)
    except (
        IRSerializationError,
        IRValidationError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        return _failure(f"sealed canonical source payload is invalid: {exc}")
    if restored.content_hash != artifact.source_content_hash:
        return _failure("restored CircuitIR content hash mismatch")
    if restored.to_json() != artifact.canonical_source_json:
        return _failure(
            "restored CircuitIR is not canonically identical to sealed source"
        )
    return CircuitExportResult(ExportStatus.EXPORTED_EXACT, restored)


class _ExportError(Exception):
    pass


def _restore_attribute(
    value: object,
    artifact: SealedCircuitIRRoundTrip,
    consumed_bindings: set[str],
    *,
    evaluate_binding_expressions: bool,
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
            raise _ExportError(f"binding {value.slot!r} is missing") from exc
        except StopIteration as exc:
            raise _ExportError(f"binding descriptor {value.slot!r} is missing") from exc
        shape = tuple(int(size) for size in getattr(bound, "shape", ()))
        dtype = str(getattr(bound, "dtype", "")).removeprefix("torch.")
        if shape != declared.shape:
            raise _ExportError(f"binding {value.slot!r} shape changed")
        if dtype != declared.dtype:
            raise _ExportError(f"binding {value.slot!r} dtype changed")
        if declared != value:
            raise _ExportError(f"binding {value.slot!r} descriptor changed")
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
            raise _ExportError("unsupported structured operation attribute")
        dtype = getattr(torch, str(value["dtype"]), None)
        if dtype is None:
            raise _ExportError(f"unsupported tensor dtype {value['dtype']!r}")
        return torch.tensor(value["data"], dtype=dtype).reshape(tuple(value["shape"]))
    if value is None or isinstance(value, (bool, int, float, complex)):
        return value
    raise _ExportError(f"unsupported operation attribute type {type(value).__name__}")


def _restore_operations(
    artifact: SealedCircuitIRRoundTrip,
    restored: CircuitIR,
    module: QuantumModule,
    consumed_bindings: set[str],
    *,
    allow_rewrites: bool,
) -> tuple[object, ...]:
    if len(module.body.blocks) != 1:
        raise _ExportError("CircuitIR export requires one entry block")
    block = module.body.blocks[0]
    if not allow_rewrites and len(block.operations) != len(restored.instructions):
        raise _ExportError("operation count changed after import")
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
                raise _ExportError(
                    "transformed operation lacks sealed CircuitIR source provenance"
                )
            skeleton = restored.instructions[location.line - 1]
        else:
            skeleton = restored.instructions[index]
        try:
            wires = tuple(value_wires[operand.id] for operand in operation.operands)
        except KeyError as exc:
            raise _ExportError("operation consumes an unknown linear value") from exc
        if len(operation.results) != len(wires):
            raise _ExportError("operation cannot preserve linear wire ownership")
        for result, wire in zip(operation.results, wires, strict=True):
            value_wires[result.id] = wire
        if operation.name == "quantum.custom_unitary":
            matrix = operation.attributes["matrix"]
            if not isinstance(matrix, FrozenAttributes):
                raise _ExportError("custom unitary matrix attribute is invalid")
            dtype = getattr(torch, str(matrix["dtype"]), None)
            if dtype is None:
                raise _ExportError("custom unitary matrix dtype is unsupported")
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
            raise _ExportError(
                f"operation {operation.name!r} is outside the CircuitIR export profile"
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


def _export_module(
    artifact: SealedCircuitIRRoundTrip,
    module: QuantumModule,
    *,
    allow_rewrites: bool,
) -> CircuitExportResult:
    verification = verify_module(module, circuit_ir_v1_schema_registry())
    if not verification.ok:
        return CircuitExportResult(
            ExportStatus.INVALID_ARTIFACT,
            diagnostics=verification.diagnostics,
        )
    exported = export_circuit_ir(artifact)
    if not exported.ok or exported.circuit_ir is None:
        return exported
    restored = exported.circuit_ir
    consumed_bindings: set[str] = set()
    try:
        instructions = _restore_operations(
            artifact,
            restored,
            module,
            consumed_bindings,
            allow_rewrites=allow_rewrites,
        )
        observables = list(restored.observables)
        if len(observables) != len(artifact.imported.request.observables):
            raise _ExportError("observable count changed after import")
        for index, request in enumerate(artifact.imported.request.observables):
            coefficient = _restore_attribute(
                request.coefficient,
                artifact,
                consumed_bindings,
                evaluate_binding_expressions=allow_rewrites,
            )
            observables[index] = replace(
                observables[index],
                name=request.name,
                wires=request.wires,
                coefficient=coefficient,
            )
        if consumed_bindings != set(artifact.imported.bindings):
            raise _ExportError("runtime binding table contains unused slots")
    except (KeyError, TypeError, _ExportError) as exc:
        return _failure(str(exc))
    return CircuitExportResult(
        ExportStatus.EXPORTED_EXACT,
        replace(
            restored,
            instructions=instructions,
            observables=tuple(observables),
        ),
    )


def export_imported_circuit_ir(
    artifact: SealedCircuitIRRoundTrip,
) -> CircuitExportResult:
    """Reconstruct CircuitIR from the exact imported internal module."""

    return _export_module(artifact, artifact.imported.module, allow_rewrites=False)


def export_transformed_circuit_ir(
    artifact: SealedCircuitIRRoundTrip,
    module: QuantumModule,
) -> CircuitExportResult:
    """Reconstruct CircuitIR from one verified, provenance-preserving transform."""

    return _export_module(artifact, module, allow_rewrites=True)


__all__ = [
    "CircuitExportResult",
    "ExportStatus",
    "RoundTripSealResult",
    "SealedCircuitIRRoundTrip",
    "export_circuit_ir",
    "export_imported_circuit_ir",
    "export_transformed_circuit_ir",
    "seal_circuit_ir_round_trip",
]
