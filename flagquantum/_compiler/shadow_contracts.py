"""Immutable contracts for private, privacy-safe shadow comparison."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

from .runtime_abi import ExecutionState

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ShadowResultKind(str, Enum):
    STATE = "state"
    PROBABILITIES = "probabilities"
    SAMPLES = "samples"
    COUNTS = "counts"
    SCALAR = "scalar"
    OPAQUE = "opaque"


class ShadowMismatch(str, Enum):
    MATCH = "match"
    STATUS_MISMATCH = "status_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    SHAPE_MISMATCH = "shape_mismatch"
    VALUE_MISMATCH = "value_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    CANDIDATE_FAILURE = "candidate_failure"
    LIMIT_BREACH = "limit_breach"


class ShadowSkipReason(str, Enum):
    POLICY_DISABLED = "policy_disabled"
    KILL_SWITCHED = "kill_switched"
    COMPARISON_LIMIT = "comparison_limit"
    INPUT_LIMIT = "input_limit"
    EVIDENCE_LIMIT = "evidence_limit"


class ShadowKillReason(str, Enum):
    OPERATOR = "operator"
    MISMATCH_LIMIT = "mismatch_limit"
    OVERHEAD_LIMIT = "overhead_limit"
    EVIDENCE_LIMIT = "evidence_limit"


@dataclass(frozen=True)
class ShadowPolicy:
    enabled: bool
    max_comparisons: int
    max_input_bytes: int
    max_evidence_bytes: int
    max_candidate_time_ns: int
    max_mismatches: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("shadow enabled flag must be boolean")
        for name in (
            "max_comparisons",
            "max_input_bytes",
            "max_evidence_bytes",
            "max_candidate_time_ns",
            "max_mismatches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"shadow {name} must be a positive integer")
        if self.max_mismatches > self.max_comparisons:
            raise ValueError("shadow mismatch limit cannot exceed comparison limit")


@dataclass(frozen=True)
class ShadowObservation:
    status: ExecutionState
    result_kind: ShadowResultKind
    shape: tuple[int, ...]
    payload: bytes
    result_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionState) or not self.status.terminal:
            raise ValueError("shadow observation requires a terminal execution state")
        if not isinstance(self.result_kind, ShadowResultKind):
            raise ValueError("shadow result kind must use the closed enum")
        shape = tuple(self.shape)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        ):
            raise ValueError("shadow result shape must contain nonnegative integers")
        if not isinstance(self.payload, bytes):
            raise ValueError("shadow result payload must be immutable bytes")
        if _SHA256.fullmatch(self.result_identity) is None:
            raise ValueError("shadow result identity must be lowercase SHA-256")
        object.__setattr__(self, "shape", shape)

    @property
    def payload_content_hash(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def classify_observations(
    legacy: ShadowObservation,
    candidate: ShadowObservation,
) -> ShadowMismatch:
    if legacy.status is not candidate.status:
        return ShadowMismatch.STATUS_MISMATCH
    if legacy.result_kind is not candidate.result_kind:
        return ShadowMismatch.TYPE_MISMATCH
    if legacy.shape != candidate.shape:
        return ShadowMismatch.SHAPE_MISMATCH
    if legacy.payload != candidate.payload:
        return ShadowMismatch.VALUE_MISMATCH
    if legacy.result_identity != candidate.result_identity:
        return ShadowMismatch.IDENTITY_MISMATCH
    return ShadowMismatch.MATCH


@dataclass(frozen=True)
class ShadowEvidence:
    classification: ShadowMismatch
    input_content_hash: str
    input_bytes: int
    legacy_content_hash: str
    legacy_bytes: int
    candidate_content_hash: str | None
    candidate_bytes: int | None
    candidate_time_ns: int
    candidate_time_exceeded: bool
    evidence_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.classification, ShadowMismatch):
            raise ValueError("shadow classification must use the closed enum")
        for name in ("input_content_hash", "legacy_content_hash"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"shadow {name} must be lowercase SHA-256")
        if (
            self.candidate_content_hash is not None
            and _SHA256.fullmatch(self.candidate_content_hash) is None
        ):
            raise ValueError("shadow candidate hash must be lowercase SHA-256")
        for name in ("input_bytes", "legacy_bytes", "candidate_time_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"shadow {name} must be a nonnegative integer")
        if self.candidate_bytes is not None and (
            isinstance(self.candidate_bytes, bool)
            or not isinstance(self.candidate_bytes, int)
            or self.candidate_bytes < 0
        ):
            raise ValueError("shadow candidate size must be a nonnegative integer")
        if not isinstance(self.candidate_time_exceeded, bool):
            raise ValueError("shadow timing limit flag must be boolean")
        if (self.candidate_content_hash is None) != (self.candidate_bytes is None):
            raise ValueError("shadow candidate hash and size must be present together")
        if self.classification is ShadowMismatch.CANDIDATE_FAILURE:
            if self.candidate_content_hash is not None:
                raise ValueError(
                    "failed shadow candidate must not expose result content"
                )
        elif self.candidate_content_hash is None:
            raise ValueError("successful shadow candidate must provide result content")
        if _SHA256.fullmatch(self.evidence_identity) is None:
            raise ValueError("shadow evidence identity must be lowercase SHA-256")
        if self.evidence_identity != _digest(self.identity_dict()):
            raise ValueError("shadow evidence identity does not match its content")

    def identity_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification.value,
            "input_content_hash": self.input_content_hash,
            "input_bytes": self.input_bytes,
            "legacy_content_hash": self.legacy_content_hash,
            "legacy_bytes": self.legacy_bytes,
            "candidate_content_hash": self.candidate_content_hash,
            "candidate_bytes": self.candidate_bytes,
            "candidate_time_ns": self.candidate_time_ns,
            "candidate_time_exceeded": self.candidate_time_exceeded,
        }

    @property
    def encoded_size(self) -> int:
        return len(
            json.dumps(
                {**self.identity_dict(), "evidence_identity": self.evidence_identity},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )


def create_shadow_evidence(
    classification: ShadowMismatch,
    input_payload: bytes,
    legacy: ShadowObservation,
    candidate: ShadowObservation | None,
    *,
    candidate_time_ns: int,
    candidate_time_exceeded: bool,
) -> ShadowEvidence:
    if not isinstance(input_payload, bytes):
        raise ValueError("shadow input must be immutable bytes")
    if not isinstance(legacy, ShadowObservation):
        raise ValueError("shadow legacy result must be typed")
    if candidate is not None and not isinstance(candidate, ShadowObservation):
        raise ValueError("shadow candidate result must be typed")
    values = {
        "classification": classification.value,
        "input_content_hash": hashlib.sha256(input_payload).hexdigest(),
        "input_bytes": len(input_payload),
        "legacy_content_hash": legacy.payload_content_hash,
        "legacy_bytes": len(legacy.payload),
        "candidate_content_hash": (
            None if candidate is None else candidate.payload_content_hash
        ),
        "candidate_bytes": None if candidate is None else len(candidate.payload),
        "candidate_time_ns": candidate_time_ns,
        "candidate_time_exceeded": candidate_time_exceeded,
    }
    return ShadowEvidence(
        classification,
        values["input_content_hash"],
        values["input_bytes"],
        values["legacy_content_hash"],
        values["legacy_bytes"],
        values["candidate_content_hash"],
        values["candidate_bytes"],
        candidate_time_ns,
        candidate_time_exceeded,
        _digest(values),
    )


@dataclass(frozen=True)
class ShadowOutcome:
    authoritative: ShadowObservation
    candidate_executed: bool
    classification: ShadowMismatch | None = None
    evidence: ShadowEvidence | None = None
    skip_reason: ShadowSkipReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.authoritative, ShadowObservation):
            raise ValueError("shadow authoritative result must be typed")
        if not isinstance(self.candidate_executed, bool):
            raise ValueError("shadow candidate-executed flag must be boolean")
        if self.classification is not None and not isinstance(
            self.classification, ShadowMismatch
        ):
            raise ValueError("shadow classification must be typed")
        if self.skip_reason is not None and not isinstance(
            self.skip_reason, ShadowSkipReason
        ):
            raise ValueError("shadow skip reason must be typed")
        compared = self.classification is not None
        if compared != (self.evidence is not None):
            raise ValueError("shadow comparison and evidence must be present together")
        if compared == (self.skip_reason is not None):
            raise ValueError("shadow outcome must be exactly compared or skipped")
        if (
            self.evidence is not None
            and self.evidence.classification is not self.classification
        ):
            raise ValueError("shadow outcome classification and evidence disagree")
        if compared and not self.candidate_executed:
            raise ValueError("shadow comparison requires candidate execution")
        if (
            self.candidate_executed
            and not compared
            and self.skip_reason is not ShadowSkipReason.EVIDENCE_LIMIT
        ):
            raise ValueError(
                "executed shadow candidate must be compared or evidence-limited"
            )


__all__ = [
    "ShadowEvidence",
    "ShadowKillReason",
    "ShadowMismatch",
    "ShadowObservation",
    "ShadowOutcome",
    "ShadowPolicy",
    "ShadowResultKind",
    "ShadowSkipReason",
    "classify_observations",
    "create_shadow_evidence",
]
