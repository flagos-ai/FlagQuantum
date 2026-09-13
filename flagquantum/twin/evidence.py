"""Empirical evidence reports for QPU digital-twin predictions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ..core.ir import CircuitIR
from .prediction import TwinPrediction

_ENVELOPE_SCHEMA = "flagquantum.twin_evidence_envelope.v1"
_REPORT_SCHEMA = "flagquantum.twin_evidence_report.v1"

TwinEvidenceStatus = Literal[
    "exact_circuit_verified",
    "within_evidence_envelope",
    "unverified",
    "out_of_scope",
]


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: str, name: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _radius(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return normalized


@dataclass(frozen=True)
class TwinEvidenceEnvelope:
    """Immutable empirical evidence boundary for one mapped Twin.

    The envelope distinguishes exact circuits verified against later hardware
    from unseen circuits that only remain inside a measured structural scope.
    It carries error bounds produced elsewhere; constructing an envelope does
    not create validation evidence.
    """

    snapshot_identity: str
    physical_qubits: tuple[int, ...]
    supported_operations: tuple[str, ...]
    maximum_instruction_count: int
    verified_circuit_identities: tuple[str, ...]
    evidence_identity: str
    verified_tv_error_bound: float | None = None
    estimated_tv_error_bound: float | None = None
    confidence_level: float | None = None
    schema: str = _ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ENVELOPE_SCHEMA:
            raise ValueError("unsupported Twin evidence-envelope schema")
        _digest(self.snapshot_identity, "snapshot_identity")
        _digest(self.evidence_identity, "evidence_identity")
        physical_qubits = tuple(int(qubit) for qubit in self.physical_qubits)
        if (
            not physical_qubits
            or len(physical_qubits) != len(set(physical_qubits))
            or any(qubit < 0 for qubit in physical_qubits)
        ):
            raise ValueError(
                "physical_qubits must be non-empty, unique, and non-negative"
            )
        operations = tuple(
            str(name).strip().lower() for name in self.supported_operations
        )
        if not operations or any(not name for name in operations):
            raise ValueError("supported_operations must contain non-empty names")
        if len(operations) != len(set(operations)):
            raise ValueError("supported_operations must be unique")
        if self.maximum_instruction_count < 0:
            raise ValueError("maximum_instruction_count must be non-negative")
        identities = tuple(
            _digest(value, "verified_circuit_identities")
            for value in self.verified_circuit_identities
        )
        if len(identities) != len(set(identities)):
            raise ValueError("verified_circuit_identities must be unique")
        verified_bound = _radius(
            self.verified_tv_error_bound, "verified_tv_error_bound"
        )
        estimated_bound = _radius(
            self.estimated_tv_error_bound, "estimated_tv_error_bound"
        )
        confidence = self.confidence_level
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
                raise ValueError("confidence_level must be finite and in (0, 1)")
        if (verified_bound is not None or estimated_bound is not None) and (
            confidence is None
        ):
            raise ValueError("an error radius requires confidence_level")
        object.__setattr__(self, "physical_qubits", physical_qubits)
        object.__setattr__(self, "supported_operations", operations)
        object.__setattr__(self, "verified_circuit_identities", identities)
        object.__setattr__(self, "verified_tv_error_bound", verified_bound)
        object.__setattr__(self, "estimated_tv_error_bound", estimated_bound)
        object.__setattr__(self, "confidence_level", confidence)

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    def supports_structure(self, circuit: CircuitIR) -> bool:
        """Return whether a circuit remains inside the structural boundary."""

        return (
            circuit.n_wires == len(self.physical_qubits)
            and len(circuit.instructions) <= self.maximum_instruction_count
            and all(
                instruction.name in self.supported_operations
                for instruction in circuit.instructions
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "snapshot_identity": self.snapshot_identity,
            "physical_qubits": list(self.physical_qubits),
            "supported_operations": list(self.supported_operations),
            "maximum_instruction_count": self.maximum_instruction_count,
            "verified_circuit_identities": list(self.verified_circuit_identities),
            "evidence_identity": self.evidence_identity,
            "verified_tv_error_bound": self.verified_tv_error_bound,
            "estimated_tv_error_bound": self.estimated_tv_error_bound,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class TwinEvidenceReport:
    """Empirical support facts for one frozen Twin prediction."""

    status: TwinEvidenceStatus
    prediction: TwinPrediction | None
    evidence_envelope_identity: str | None
    evidence_identity: str | None
    exact_circuit_verified: bool
    structurally_supported: bool
    tv_error_bound: float | None
    confidence_level: float | None
    reasons: tuple[str, ...]
    schema: str = _REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _REPORT_SCHEMA:
            raise ValueError("unsupported Twin evidence-report schema")
        if self.status not in {
            "exact_circuit_verified",
            "within_evidence_envelope",
            "unverified",
            "out_of_scope",
        }:
            raise ValueError("unsupported Twin evidence status")
        if self.evidence_envelope_identity is not None:
            _digest(self.evidence_envelope_identity, "evidence_envelope_identity")
        if self.evidence_identity is not None:
            _digest(self.evidence_identity, "evidence_identity")
        bound = _radius(self.tv_error_bound, "tv_error_bound")
        confidence = self.confidence_level
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
                raise ValueError("confidence_level must be finite and in (0, 1)")
        if (bound is None) != (confidence is None):
            raise ValueError("tv_error_bound and confidence_level must appear together")
        if self.status == "exact_circuit_verified" and (
            not self.exact_circuit_verified or bound is None
        ):
            raise ValueError(
                "exact_circuit_verified requires exact evidence and an error bound"
            )
        if self.status == "within_evidence_envelope" and (
            not self.structurally_supported or bound is None
        ):
            raise ValueError(
                "within_evidence_envelope requires structural support and a bound"
            )
        reasons = tuple(str(reason).strip() for reason in self.reasons)
        if any(not reason for reason in reasons):
            raise ValueError("evidence-report reasons cannot be empty")
        object.__setattr__(self, "tv_error_bound", bound)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "prediction": (
                None if self.prediction is None else self.prediction.to_dict()
            ),
            "evidence_envelope_identity": self.evidence_envelope_identity,
            "evidence_identity": self.evidence_identity,
            "exact_circuit_verified": self.exact_circuit_verified,
            "structurally_supported": self.structurally_supported,
            "tv_error_bound": self.tv_error_bound,
            "confidence_level": self.confidence_level,
            "reasons": list(self.reasons),
        }


def build_evidence_report(
    prediction: TwinPrediction,
    circuit: CircuitIR,
    *,
    physical_qubits: Sequence[int],
    evidence: TwinEvidenceEnvelope | None,
) -> TwinEvidenceReport:
    """Describe empirical support without making an application decision."""

    if evidence is None:
        return TwinEvidenceReport(
            status="unverified",
            prediction=prediction,
            evidence_envelope_identity=None,
            evidence_identity=None,
            exact_circuit_verified=False,
            structurally_supported=False,
            tv_error_bound=None,
            confidence_level=None,
            reasons=("evidence_envelope_missing",),
        )
    envelope_identity = evidence.identity
    reasons: list[str] = []
    if evidence.snapshot_identity != prediction.snapshot_identity:
        reasons.append("snapshot_identity_mismatch")
    if evidence.physical_qubits != tuple(int(value) for value in physical_qubits):
        reasons.append("physical_mapping_mismatch")
    if reasons:
        return TwinEvidenceReport(
            status="out_of_scope",
            prediction=prediction,
            evidence_envelope_identity=envelope_identity,
            evidence_identity=evidence.evidence_identity,
            exact_circuit_verified=False,
            structurally_supported=False,
            tv_error_bound=None,
            confidence_level=None,
            reasons=tuple(reasons),
        )

    structurally_supported = evidence.supports_structure(circuit)
    exact = circuit.content_hash in evidence.verified_circuit_identities
    if (
        exact
        and structurally_supported
        and evidence.verified_tv_error_bound is not None
    ):
        return TwinEvidenceReport(
            status="exact_circuit_verified",
            prediction=prediction,
            evidence_envelope_identity=envelope_identity,
            evidence_identity=evidence.evidence_identity,
            exact_circuit_verified=True,
            structurally_supported=True,
            tv_error_bound=evidence.verified_tv_error_bound,
            confidence_level=evidence.confidence_level,
            reasons=(),
        )
    if structurally_supported and evidence.estimated_tv_error_bound is not None:
        return TwinEvidenceReport(
            status="within_evidence_envelope",
            prediction=prediction,
            evidence_envelope_identity=envelope_identity,
            evidence_identity=evidence.evidence_identity,
            exact_circuit_verified=exact,
            structurally_supported=True,
            tv_error_bound=evidence.estimated_tv_error_bound,
            confidence_level=evidence.confidence_level,
            reasons=("exact_circuit_not_verified",) if not exact else (),
        )

    if not structurally_supported:
        reasons.append("outside_structural_support")
    if exact and evidence.verified_tv_error_bound is None:
        reasons.append("verified_error_bound_missing")
    elif not exact:
        reasons.append("exact_circuit_not_verified")
    if evidence.estimated_tv_error_bound is None:
        reasons.append("estimated_error_bound_missing")
    return TwinEvidenceReport(
        status="out_of_scope" if not structurally_supported else "unverified",
        prediction=prediction,
        evidence_envelope_identity=envelope_identity,
        evidence_identity=evidence.evidence_identity,
        exact_circuit_verified=exact,
        structurally_supported=structurally_supported,
        tv_error_bound=None,
        confidence_level=None,
        reasons=tuple(reasons),
    )


__all__ = (
    "build_evidence_report",
    "TwinEvidenceEnvelope",
    "TwinEvidenceReport",
    "TwinEvidenceStatus",
)
