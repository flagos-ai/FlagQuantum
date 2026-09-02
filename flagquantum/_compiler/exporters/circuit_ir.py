"""Restricted exact CircuitIR round-trip exporter for Phase 1 evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from flagquantum.core.ir import CircuitIR, IRSerializationError, IRValidationError

from ..diagnostics import Diagnostic, DiagnosticCode
from ..import_models import ImportedCircuitProgram, ImportStatus
from ..importers.circuit_ir import import_circuit_ir


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


__all__ = [
    "CircuitExportResult",
    "ExportStatus",
    "RoundTripSealResult",
    "SealedCircuitIRRoundTrip",
    "export_circuit_ir",
    "seal_circuit_ir_round_trip",
]
