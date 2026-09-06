"""Explicit, removable differential bridge for Phase 1 evidence only."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from flagquantum.core.ir import CircuitIR

from ..diagnostics import Diagnostic
from ..exporters.circuit_ir import (
    SealedCircuitIRRoundTrip,
    export_imported_circuit_ir,
    export_transformed_circuit_ir,
    seal_circuit_ir_round_trip,
)
from ..ir.modules import QuantumModule


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


def lower_sealed_for_differential(
    artifact: SealedCircuitIRRoundTrip,
) -> DifferentialLoweringResult:
    """Lower verified QuantumIR operations onto a sealed legacy source envelope."""

    exported = export_imported_circuit_ir(artifact)
    if not exported.ok or exported.circuit_ir is None:
        return DifferentialLoweringResult(None, False, exported.diagnostics)
    lowered = exported.circuit_ir
    identity_preserved = True
    for reference in artifact.imported.bindings.references:
        value = artifact.imported.bindings[reference.slot]
        if reference.slot.startswith("op"):
            operation_index, parameter = reference.slot[2:].split(".", 1)
            identity_preserved &= (
                lowered.instructions[int(operation_index)].params[parameter] is value
            )
        elif reference.slot.startswith("observable"):
            observable_index = int(reference.slot[10:].split(".", 1)[0])
            identity_preserved &= (
                lowered.observables[observable_index].coefficient is value
            )
    return DifferentialLoweringResult(lowered, identity_preserved)


def lower_module_for_differential(
    artifact: SealedCircuitIRRoundTrip,
    module: QuantumModule,
) -> DifferentialLoweringResult:
    """Lower one verified Phase 2 module against its immutable source envelope."""

    exported = export_transformed_circuit_ir(artifact, module)
    if not exported.ok or exported.circuit_ir is None:
        return DifferentialLoweringResult(None, False, exported.diagnostics)
    return DifferentialLoweringResult(exported.circuit_ir, True)


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
