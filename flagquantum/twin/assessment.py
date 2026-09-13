"""Evidence-bounded decisions for QPU digital-twin predictions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ..core.ir import CircuitIR
from .prediction import TwinPrediction

_SUPPORT_SCHEMA = "flagquantum.twin_support_envelope.v1"
_ASSESSMENT_SCHEMA = "flagquantum.twin_assessment.v1"

TwinDecision = Literal[
    "verified_prediction",
    "bounded_estimate",
    "physical_reference",
    "unsupported",
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
class TwinSupportEnvelope:
    """Immutable evidence boundary for one mapped Twin.

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
    verified_tv_error_radius: float | None = None
    estimated_tv_error_radius: float | None = None
    confidence_level: float | None = None
    schema: str = _SUPPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SUPPORT_SCHEMA:
            raise ValueError("unsupported Twin support-envelope schema")
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
        verified_radius = _radius(
            self.verified_tv_error_radius, "verified_tv_error_radius"
        )
        estimated_radius = _radius(
            self.estimated_tv_error_radius, "estimated_tv_error_radius"
        )
        confidence = self.confidence_level
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
                raise ValueError("confidence_level must be finite and in (0, 1)")
        if (verified_radius is not None or estimated_radius is not None) and (
            confidence is None
        ):
            raise ValueError("an error radius requires confidence_level")
        object.__setattr__(self, "physical_qubits", physical_qubits)
        object.__setattr__(self, "supported_operations", operations)
        object.__setattr__(self, "verified_circuit_identities", identities)
        object.__setattr__(self, "verified_tv_error_radius", verified_radius)
        object.__setattr__(self, "estimated_tv_error_radius", estimated_radius)
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
            "verified_tv_error_radius": self.verified_tv_error_radius,
            "estimated_tv_error_radius": self.estimated_tv_error_radius,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class TwinAssessment:
    """A prediction classified by its frozen empirical support."""

    decision: TwinDecision
    prediction: TwinPrediction | None
    support_envelope_identity: str | None
    evidence_identity: str | None
    exact_circuit_verified: bool
    structurally_supported: bool
    tv_error_radius: float | None
    confidence_level: float | None
    reasons: tuple[str, ...]
    schema: str = _ASSESSMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ASSESSMENT_SCHEMA:
            raise ValueError("unsupported Twin assessment schema")
        if self.decision not in {
            "verified_prediction",
            "bounded_estimate",
            "physical_reference",
            "unsupported",
        }:
            raise ValueError("unsupported Twin assessment decision")
        if self.support_envelope_identity is not None:
            _digest(self.support_envelope_identity, "support_envelope_identity")
        if self.evidence_identity is not None:
            _digest(self.evidence_identity, "evidence_identity")
        radius = _radius(self.tv_error_radius, "tv_error_radius")
        confidence = self.confidence_level
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
                raise ValueError("confidence_level must be finite and in (0, 1)")
        if (radius is None) != (confidence is None):
            raise ValueError(
                "tv_error_radius and confidence_level must appear together"
            )
        if self.decision == "verified_prediction" and (
            not self.exact_circuit_verified or radius is None
        ):
            raise ValueError(
                "verified_prediction requires exact-circuit evidence and an error bound"
            )
        if self.decision == "bounded_estimate" and (
            not self.structurally_supported or radius is None
        ):
            raise ValueError(
                "bounded_estimate requires structural support and an error bound"
            )
        reasons = tuple(str(reason).strip() for reason in self.reasons)
        if any(not reason for reason in reasons):
            raise ValueError("assessment reasons cannot be empty")
        object.__setattr__(self, "tv_error_radius", radius)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "reasons", reasons)

    @property
    def actionable(self) -> bool:
        """Whether the assessment carries an empirically bounded prediction."""

        return self.decision in {"verified_prediction", "bounded_estimate"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "decision": self.decision,
            "prediction": (
                None if self.prediction is None else self.prediction.to_dict()
            ),
            "support_envelope_identity": self.support_envelope_identity,
            "evidence_identity": self.evidence_identity,
            "exact_circuit_verified": self.exact_circuit_verified,
            "structurally_supported": self.structurally_supported,
            "tv_error_radius": self.tv_error_radius,
            "confidence_level": self.confidence_level,
            "reasons": list(self.reasons),
            "actionable": self.actionable,
        }


def assess_prediction(
    prediction: TwinPrediction,
    circuit: CircuitIR,
    *,
    physical_qubits: Sequence[int],
    support: TwinSupportEnvelope | None,
) -> TwinAssessment:
    """Classify one prediction without manufacturing missing evidence."""

    if support is None:
        return TwinAssessment(
            decision="physical_reference",
            prediction=prediction,
            support_envelope_identity=None,
            evidence_identity=None,
            exact_circuit_verified=False,
            structurally_supported=False,
            tv_error_radius=None,
            confidence_level=None,
            reasons=("support_envelope_missing",),
        )
    envelope_identity = support.identity
    reasons: list[str] = []
    if support.snapshot_identity != prediction.snapshot_identity:
        reasons.append("snapshot_identity_mismatch")
    if support.physical_qubits != tuple(int(value) for value in physical_qubits):
        reasons.append("physical_mapping_mismatch")
    if reasons:
        return TwinAssessment(
            decision="unsupported",
            prediction=prediction,
            support_envelope_identity=envelope_identity,
            evidence_identity=support.evidence_identity,
            exact_circuit_verified=False,
            structurally_supported=False,
            tv_error_radius=None,
            confidence_level=None,
            reasons=tuple(reasons),
        )

    structurally_supported = support.supports_structure(circuit)
    exact = circuit.content_hash in support.verified_circuit_identities
    if (
        exact
        and structurally_supported
        and support.verified_tv_error_radius is not None
    ):
        return TwinAssessment(
            decision="verified_prediction",
            prediction=prediction,
            support_envelope_identity=envelope_identity,
            evidence_identity=support.evidence_identity,
            exact_circuit_verified=True,
            structurally_supported=True,
            tv_error_radius=support.verified_tv_error_radius,
            confidence_level=support.confidence_level,
            reasons=(),
        )
    if structurally_supported and support.estimated_tv_error_radius is not None:
        return TwinAssessment(
            decision="bounded_estimate",
            prediction=prediction,
            support_envelope_identity=envelope_identity,
            evidence_identity=support.evidence_identity,
            exact_circuit_verified=exact,
            structurally_supported=True,
            tv_error_radius=support.estimated_tv_error_radius,
            confidence_level=support.confidence_level,
            reasons=("exact_circuit_not_verified",) if not exact else (),
        )

    if not structurally_supported:
        reasons.append("outside_structural_support")
    if exact and support.verified_tv_error_radius is None:
        reasons.append("verified_error_bound_missing")
    elif not exact:
        reasons.append("exact_circuit_not_verified")
    if support.estimated_tv_error_radius is None:
        reasons.append("estimated_error_bound_missing")
    return TwinAssessment(
        decision="unsupported" if not structurally_supported else "physical_reference",
        prediction=prediction,
        support_envelope_identity=envelope_identity,
        evidence_identity=support.evidence_identity,
        exact_circuit_verified=exact,
        structurally_supported=structurally_supported,
        tv_error_radius=None,
        confidence_level=None,
        reasons=tuple(reasons),
    )


__all__ = (
    "assess_prediction",
    "TwinAssessment",
    "TwinDecision",
    "TwinSupportEnvelope",
)
