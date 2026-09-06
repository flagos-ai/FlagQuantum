"""Fail-closed importer from public CircuitIR 1.0 to private QuantumIR."""

from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock
from typing import Any

import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, get_operator_schema
from flagquantum.core.parameters import Parameter, ParameterExpression

from ..bindings import (
    BindingTable,
    RuntimeBindingRef,
    SymbolicExpression,
    SymbolicParameter,
)
from ..diagnostics import Diagnostic, DiagnosticCode, DiagnosticLocation
from ..import_models import (
    CircuitImportResult,
    ImportConstraints,
    ImportedCircuitProgram,
    ImportStatus,
    InstructionSemantics,
    InternalExecutionRequest,
    InternalMeasurementRequest,
    InternalObservableRequest,
    SourceIdentity,
    SourceProvenance,
)
from ..ir.modules import Block, QuantumModule, Region
from ..ir.operations import FrozenAttributes, Operation, SourceLocation
from ..ir.schemas import circuit_ir_v1_schema_registry
from ..ir.types import QUBIT
from ..ir.values import ValueId, ValueRef
from ..ir.verifier import verify_module

_INSTRUCTION_PROVENANCE = {"source", "qiskit_label"}
_INSTRUCTION_SEMANTICS = {
    "diagonal",
    "global_phase",
    "mpo",
    "operator_schmidt_rank",
    "pauli",
    "split",
}
_DYNAMIC_KEYS = {"classical_bit", "condition", "conditions", "is_dynamic"}
_CIRCUIT_PROVENANCE = {
    "circuit_name",
    "interop",
    "qiskit_label",
    "routing",
    "routing_strategy",
    "routing_strategy_selection",
    "source",
    "statevector_dependency_schedule_changed",
    "statevector_dependency_scheduled",
}
_CIRCUIT_CONSTRAINTS = {
    "batch_size",
    "logical_state_shape",
    "runtime_config",
}
_CIRCUIT_REQUEST = {"num_clbits"}
_MEASUREMENT_METADATA = {
    "format",
    "max_marginal_wires",
    "max_postselection_draw_multiplier",
    "name",
    "postselect",
    "seed",
}
_OBSERVABLE_METADATA = {"name", "pauli", "source"}
_EMPTY_ATTRIBUTES = FrozenAttributes()
_QUANTUM_OPERATION_NAMES = {opcode: f"quantum.{opcode}" for opcode in OPERATOR_SCHEMAS}
_SUCCESS_CACHE_CAPACITY = 16
_SUCCESS_CACHE_LOCK = RLock()
_SUCCESS_CACHE: OrderedDict[
    int,
    tuple[
        weakref.ReferenceType[CircuitIR],
        tuple[object, ...],
        ImportedCircuitProgram,
    ],
] = OrderedDict()


class _UnsupportedImportError(Exception):
    def __init__(self, diagnostic: Diagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


def _fingerprint_value(value: object) -> object:
    """Build a lossless comparison value without JSON encoding or hashing."""

    if isinstance(value, Parameter):
        return ("parameter", value.name)
    if isinstance(value, ParameterExpression):
        return (
            "expression",
            value.op,
            tuple(_fingerprint_value(item) for item in value.args),
        )
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return (
            "tensor",
            id(value),
            bool(value.requires_grad),
            str(tensor.dtype),
            tuple(tensor.shape),
            _fingerprint_value(tensor.tolist()),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), _fingerprint_value(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_fingerprint_value(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_fingerprint_value(item) for item in value))
    if isinstance(value, complex):
        return ("complex", value.real, value.imag)
    if value is None or isinstance(value, (str, int, float, bool)):
        return (type(value).__name__, value)
    if hasattr(value, "tolist"):
        return (
            "array_like",
            str(getattr(value, "dtype", "")),
            tuple(getattr(value, "shape", ())),
            _fingerprint_value(value.tolist()),
        )
    return (type(value), id(value), repr(value))


def _source_fingerprint(program: CircuitIR) -> tuple[object, ...]:
    return (
        program.version,
        program.n_wires,
        program.dtype,
        tuple(program.shape),
        tuple(
            (
                instruction.name,
                tuple(instruction.wires),
                _fingerprint_value(instruction.params),
                _fingerprint_value(instruction.matrix),
                _fingerprint_value(instruction.metadata),
            )
            for instruction in program.instructions
        ),
        tuple(
            (
                observable.name,
                tuple(observable.wires),
                _fingerprint_value(observable.coefficient),
                _fingerprint_value(observable.metadata),
            )
            for observable in program.observables
        ),
        tuple(
            (
                measurement.kind,
                tuple(measurement.wires),
                measurement.shots,
                _fingerprint_value(measurement.metadata),
            )
            for measurement in program.measurements
        ),
        _fingerprint_value(program.metadata),
    )


def _cached_success(
    program: CircuitIR,
    source_fingerprint: tuple[object, ...],
) -> ImportedCircuitProgram | None:
    key = id(program)
    with _SUCCESS_CACHE_LOCK:
        cached = _SUCCESS_CACHE.get(key)
        if cached is None:
            return None
        source_ref, cached_fingerprint, imported = cached
        if source_ref() is not program or cached_fingerprint != source_fingerprint:
            del _SUCCESS_CACHE[key]
            return None
        _SUCCESS_CACHE.move_to_end(key)
        return imported


def _remember_success(
    program: CircuitIR,
    source_fingerprint: tuple[object, ...],
    imported: ImportedCircuitProgram,
) -> None:
    key = id(program)

    def discard(source_ref: weakref.ReferenceType[CircuitIR]) -> None:
        with _SUCCESS_CACHE_LOCK:
            cached = _SUCCESS_CACHE.get(key)
            if cached is not None and cached[0] is source_ref:
                del _SUCCESS_CACHE[key]

    source_ref = weakref.ref(program, discard)
    with _SUCCESS_CACHE_LOCK:
        _SUCCESS_CACHE[key] = (source_ref, source_fingerprint, imported)
        _SUCCESS_CACHE.move_to_end(key)
        while len(_SUCCESS_CACHE) > _SUCCESS_CACHE_CAPACITY:
            _SUCCESS_CACHE.popitem(last=False)


def _source_location(index: int) -> DiagnosticLocation:
    return DiagnosticLocation("CircuitIR.instructions", index + 1)


def _unsupported(
    code: DiagnosticCode,
    message: str,
    *,
    index: int | None = None,
    notes: tuple[str, ...] = (),
) -> _UnsupportedImportError:
    location = _source_location(index) if index is not None else None
    return _UnsupportedImportError(Diagnostic(code, message, location, notes))


def _static_tensor(value: torch.Tensor) -> FrozenAttributes:
    tensor = value.detach().cpu()
    data = tensor.item() if tensor.ndim == 0 else tensor.tolist()
    return FrozenAttributes(
        {
            "kind": "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": tuple(tensor.shape),
            "data": data,
        }
    )


def _convert_parameter(
    value: Any,
    *,
    slot: str,
    binding_entries: list[tuple[RuntimeBindingRef, Any]],
) -> object:
    if isinstance(value, Parameter):
        return SymbolicParameter(value.name)
    if isinstance(value, ParameterExpression):
        args = tuple(
            _convert_parameter(
                arg,
                slot=f"{slot}.arg{index}",
                binding_entries=binding_entries,
            )
            for index, arg in enumerate(value.args)
        )
        try:
            return SymbolicExpression(value.op, args)
        except ValueError as exc:
            raise _unsupported(
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                str(exc),
            ) from exc
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise _unsupported(
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                f"gate parameter {slot!r} must be a scalar tensor",
            )
        if value.requires_grad:
            reference = RuntimeBindingRef(
                slot,
                tuple(value.shape),
                str(value.dtype).removeprefix("torch."),
            )
            binding_entries.append((reference, value))
            return reference
        return _static_tensor(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise _unsupported(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            f"gate parameter {slot!r} must be finite",
        )
    if isinstance(value, complex) and not (
        math.isfinite(value.real) and math.isfinite(value.imag)
    ):
        raise _unsupported(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            f"gate parameter {slot!r} must be finite",
        )
    if isinstance(value, (bool, int, float, complex)):
        return value
    raise _unsupported(
        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
        f"unsupported gate parameter type {type(value).__name__} at {slot!r}",
    )


def _matrix_attribute(matrix: Any, *, arity: int, dtype: str) -> FrozenAttributes:
    if isinstance(matrix, torch.Tensor) and matrix.requires_grad:
        raise _unsupported(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "trainable custom matrices are outside circuit_ir_v1_static",
        )
    try:
        tensor = torch.as_tensor(matrix).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise _unsupported(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "custom matrix is not a concrete numeric tensor",
        ) from exc
    expected = 2**arity
    if tensor.ndim != 2 or tuple(tensor.shape) != (expected, expected):
        raise _unsupported(
            DiagnosticCode.INVALID_VARIADIC_ARITY,
            f"custom matrix for {arity} wires must have shape {(expected, expected)}",
        )
    if not torch.isfinite(tensor).all():
        raise _unsupported(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "custom matrix elements must be finite",
        )
    complex_dtype = torch.complex128 if dtype == "complex128" else torch.complex64
    candidate = tensor.to(complex_dtype)
    identity = torch.eye(expected, dtype=complex_dtype)
    tolerance = 1e-12 if complex_dtype == torch.complex128 else 1e-5
    if not torch.allclose(
        candidate.mH @ candidate, identity, atol=tolerance, rtol=tolerance
    ):
        raise _unsupported(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "custom matrix must be unitary under the imported dtype contract",
        )
    data = tuple(tuple(complex(item) for item in row) for row in candidate.tolist())
    return FrozenAttributes(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": (expected, expected),
            "data": data,
        }
    )


def _kraus_attribute(kraus: Any, *, arity: int, dtype: str) -> FrozenAttributes:
    try:
        if isinstance(kraus, torch.Tensor):
            tensor = kraus.detach().cpu()
        else:
            tensor = torch.stack(tuple(torch.as_tensor(item) for item in kraus))
            tensor = tensor.detach().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise _unsupported(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "channel Kraus data must be a concrete tensor sequence",
        ) from exc
    expected = 2**arity
    if (
        tensor.ndim != 3
        or tensor.shape[0] < 1
        or tuple(tensor.shape[1:]) != (expected, expected)
    ):
        raise _unsupported(
            DiagnosticCode.INVALID_VARIADIC_ARITY,
            "channel Kraus data must contain one or more square operators "
            f"with shape {(expected, expected)}",
        )
    if not torch.isfinite(tensor).all():
        raise _unsupported(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "channel Kraus operators must be finite",
        )
    complex_dtype = torch.complex128 if dtype == "complex128" else torch.complex64
    candidate = tensor.to(complex_dtype)
    effect = sum(operator.mH @ operator for operator in candidate)
    identity = torch.eye(expected, dtype=complex_dtype)
    low_precision = tensor.dtype in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.complex64,
    }
    tolerance = 1e-5 if low_precision else 1e-12
    if not torch.allclose(effect, identity, atol=tolerance, rtol=tolerance):
        raise _unsupported(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            "channel Kraus operators must define a trace-preserving map",
        )
    data = tuple(
        tuple(tuple(complex(item) for item in row) for row in operator)
        for operator in candidate.tolist()
    )
    return FrozenAttributes(
        {
            "kind": "kraus",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": tuple(tensor.shape),
            "data": data,
        }
    )


def _instruction_operation(
    instruction: Instruction,
    *,
    index: int,
    dtype: str,
    current_values: dict[int, ValueRef],
    next_value: int,
    binding_entries: list[tuple[RuntimeBindingRef, Any]],
    provenance: dict[str, object],
    instruction_semantics: list[InstructionSemantics],
) -> tuple[Operation, int]:
    metadata = instruction.metadata
    if metadata:
        dynamic = sorted(key for key in _DYNAMIC_KEYS if metadata.get(key) is not None)
        if dynamic and (
            metadata.get("is_dynamic") or any(key != "is_dynamic" for key in dynamic)
        ):
            raise _unsupported(
                DiagnosticCode.UNKNOWN_OPERATION,
                f"dynamic instruction {instruction.name!r} is outside circuit_ir_v1_static",
                index=index,
                notes=(f"dynamic metadata: {', '.join(dynamic)}",),
            )
    semantic = get_operator_schema(instruction.name)
    if semantic is None and instruction.matrix is None:
        raise _unsupported(
            DiagnosticCode.UNKNOWN_OPERATION,
            f"operation {instruction.name!r} is not a supported canonical opcode",
            index=index,
        )
    if semantic is not None and not semantic.channel and instruction.matrix is not None:
        raise _unsupported(
            DiagnosticCode.UNKNOWN_ATTRIBUTE,
            f"registered opcode {instruction.name!r} cannot also override its matrix",
            index=index,
        )
    if metadata and "is_channel" in metadata:
        marker = metadata["is_channel"]
        if not isinstance(marker, bool):
            raise _unsupported(
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                "is_channel metadata must be a bool",
                index=index,
            )
        expected_channel = semantic.channel if semantic is not None else False
        if marker is not expected_channel:
            raise _unsupported(
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                f"is_channel metadata contradicts opcode {instruction.name!r}",
                index=index,
            )
    metadata_keys = set(metadata) if metadata else set()
    for key in _INSTRUCTION_PROVENANCE & metadata_keys:
        provenance[f"instruction.{index}.{key}"] = metadata[key]
    instruction_semantics.append(
        InstructionSemantics(
            index,
            (
                FrozenAttributes(
                    {
                        key: metadata[key]
                        for key in _INSTRUCTION_SEMANTICS & metadata_keys
                    }
                )
                if metadata_keys & _INSTRUCTION_SEMANTICS
                else _EMPTY_ATTRIBUTES
            ),
        )
    )

    operands = tuple(current_values[wire] for wire in instruction.wires)
    results = tuple(
        ValueRef(ValueId(next_value + offset), QUBIT)
        for offset in range(len(instruction.wires))
    )
    attributes: dict[str, object]
    if semantic is None:
        attributes = {
            "symbolic_name": instruction.name,
            "matrix": _matrix_attribute(
                instruction.matrix, arity=len(instruction.wires), dtype=dtype
            ),
            "arity": len(instruction.wires),
        }
        name = "quantum.custom_unitary"
    else:
        extra_params = (
            sorted(set(instruction.params) - set(semantic.parameters))
            if instruction.params
            else []
        )
        if extra_params:
            raise _unsupported(
                DiagnosticCode.UNKNOWN_ATTRIBUTE,
                f"operation {instruction.name!r} has undeclared parameters",
                index=index,
                notes=(f"unknown parameters: {', '.join(extra_params)}",),
            )
        if semantic.channel and instruction.matrix is not None:
            attributes = {
                "kraus": _kraus_attribute(
                    instruction.matrix,
                    arity=len(instruction.wires),
                    dtype=dtype,
                )
            }
        else:
            attributes = (
                {
                    parameter: _convert_parameter(
                        instruction.params[parameter],
                        slot=f"op{index}.{parameter}",
                        binding_entries=binding_entries,
                    )
                    for parameter in semantic.parameters
                }
                if semantic.parameters
                else _EMPTY_ATTRIBUTES
            )
        name = _QUANTUM_OPERATION_NAMES[instruction.name]
    operation = Operation(
        name,
        operands,
        results,
        attributes,
        location=SourceLocation("CircuitIR.instructions", index + 1),
    )
    for wire, result in zip(instruction.wires, results, strict=True):
        current_values[wire] = result
    return operation, next_value + len(results)


def _measurement_request(node: Any) -> InternalMeasurementRequest:
    metadata = dict(node.metadata)
    unknown = sorted(set(metadata) - _MEASUREMENT_METADATA)
    if unknown:
        raise _unsupported(
            DiagnosticCode.UNKNOWN_ATTRIBUTE,
            "measurement request has unsupported metadata",
            notes=(f"unclassified keys: {', '.join(unknown)}",),
        )
    postselection = metadata.get("postselect")
    if postselection is not None and not isinstance(postselection, Mapping):
        postselection = {"value": postselection}
    return InternalMeasurementRequest(
        node.kind,
        tuple(node.wires),
        node.shots,
        seed=None if metadata.get("seed") is None else int(metadata["seed"]),
        name=None if metadata.get("name") is None else str(metadata["name"]),
        format=None if metadata.get("format") is None else str(metadata["format"]),
        postselection=(
            FrozenAttributes(postselection) if postselection is not None else None
        ),
        max_marginal_wires=(
            None
            if metadata.get("max_marginal_wires") is None
            else int(metadata["max_marginal_wires"])
        ),
        max_postselection_draw_multiplier=(
            None
            if metadata.get("max_postselection_draw_multiplier") is None
            else int(metadata["max_postselection_draw_multiplier"])
        ),
    )


def _observable_request(
    node: Any,
    *,
    index: int,
    binding_entries: list[tuple[RuntimeBindingRef, Any]],
) -> InternalObservableRequest:
    metadata = dict(node.metadata)
    unknown = sorted(set(metadata) - _OBSERVABLE_METADATA)
    if unknown:
        raise _unsupported(
            DiagnosticCode.UNKNOWN_ATTRIBUTE,
            "observable request has unsupported metadata",
            notes=(f"unclassified keys: {', '.join(unknown)}",),
        )
    return InternalObservableRequest(
        node.name,
        tuple(node.wires),
        _convert_parameter(
            node.coefficient,
            slot=f"observable{index}.coefficient",
            binding_entries=binding_entries,
        ),
        FrozenAttributes(metadata),
    )


def import_circuit_ir(program: object) -> CircuitImportResult:
    """Import one validated CircuitIR without mutating it or changing public paths."""

    if not isinstance(program, CircuitIR):
        diagnostic = Diagnostic(
            DiagnosticCode.VALUE_TYPE_MISMATCH,
            "CircuitIR importer requires a validated CircuitIR 1.0 object",
        )
        return CircuitImportResult(
            ImportStatus.INVALID_INPUT, diagnostics=(diagnostic,)
        )
    try:
        program.validate()
    except (TypeError, ValueError) as exc:
        diagnostic = Diagnostic(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            f"invalid CircuitIR input: {exc}",
        )
        return CircuitImportResult(
            ImportStatus.INVALID_INPUT, diagnostics=(diagnostic,)
        )
    try:
        if program.dtype not in {"complex64", "complex128"}:
            raise _unsupported(
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                f"unsupported CircuitIR dtype {program.dtype!r}",
            )
        metadata = dict(program.metadata)
        constraint_values = {
            key: metadata[key] for key in _CIRCUIT_CONSTRAINTS if key in metadata
        }
        source_fingerprint = _source_fingerprint(program)
        cached = _cached_success(program, source_fingerprint)
        if cached is not None:
            return CircuitImportResult(ImportStatus.SUPPORTED_EXACT, cached)
        # Compute the canonical source identity before allocating the internal
        # module so serialization temporaries do not overlap its retained graph.
        source_content_hash = program.content_hash
        binding_entries: list[tuple[RuntimeBindingRef, Any]] = []
        instruction_semantics: list[InstructionSemantics] = []
        provenance = {key: metadata[key] for key in _CIRCUIT_PROVENANCE & set(metadata)}
        current_values = {
            wire: ValueRef(ValueId(wire), QUBIT) for wire in range(program.n_wires)
        }
        next_value = program.n_wires
        operations = []
        for index, instruction in enumerate(program.instructions):
            try:
                operation, next_value = _instruction_operation(
                    instruction,
                    index=index,
                    dtype=program.dtype,
                    current_values=current_values,
                    next_value=next_value,
                    binding_entries=binding_entries,
                    provenance=provenance,
                    instruction_semantics=instruction_semantics,
                )
            except _UnsupportedImportError as exc:
                if exc.diagnostic.location is not None:
                    raise
                raise _UnsupportedImportError(
                    Diagnostic(
                        exc.diagnostic.code,
                        exc.diagnostic.message,
                        _source_location(index),
                        exc.diagnostic.notes,
                    )
                ) from exc
            operations.append(operation)
        arguments = tuple(
            ValueRef(ValueId(wire), QUBIT) for wire in range(program.n_wires)
        )
        module = QuantumModule(Region((Block(arguments, tuple(operations)),)))
        verifier_result = verify_module(module, circuit_ir_v1_schema_registry())
        if not verifier_result.ok:
            return CircuitImportResult(
                ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
                diagnostics=verifier_result.diagnostics,
            )
        request = InternalExecutionRequest(
            tuple(
                _observable_request(
                    node,
                    index=index,
                    binding_entries=binding_entries,
                )
                for index, node in enumerate(program.observables)
            ),
            tuple(_measurement_request(node) for node in program.measurements),
            classical_width=(
                None
                if metadata.get("num_clbits") is None
                else int(metadata["num_clbits"])
            ),
        )
        constraints = ImportConstraints(
            program.dtype,
            tuple(program.shape),
            batch_size=(
                None
                if metadata.get("batch_size") is None
                else int(metadata["batch_size"])
            ),
            logical_state_shape=(
                None
                if constraint_values.get("logical_state_shape") is None
                else tuple(
                    int(size) for size in constraint_values["logical_state_shape"]
                )
            ),
            runtime_config=(
                None
                if metadata.get("runtime_config") is None
                else FrozenAttributes(metadata["runtime_config"])
            ),
        )
        imported = ImportedCircuitProgram(
            module,
            request,
            constraints,
            SourceIdentity(program.version, source_content_hash),
            SourceProvenance(FrozenAttributes(provenance)),
            tuple(instruction_semantics),
            BindingTable(tuple(binding_entries)),
        )
        _remember_success(program, source_fingerprint, imported)
        return CircuitImportResult(ImportStatus.SUPPORTED_EXACT, imported)
    except _UnsupportedImportError as exc:
        return CircuitImportResult(
            ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
            diagnostics=(exc.diagnostic,),
        )
    except (TypeError, ValueError) as exc:
        diagnostic = Diagnostic(
            DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
            f"CircuitIR cannot be represented exactly: {exc}",
        )
        return CircuitImportResult(
            ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
            diagnostics=(diagnostic,),
        )


__all__ = ["import_circuit_ir"]
