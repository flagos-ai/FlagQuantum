"""Runtime-owned candidate matching over the Core Target Capabilities seam.

This module is intentionally not imported by the default execution path.  It
owns candidate ordering, route provenance, fallback authorization and a private
decision record; Core owns requirement/snapshot validation and pure comparison.
No value is copied into or rewritten on either Core object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

import flagquantum.core.target_capabilities as _core
from flagquantum.core.target_capabilities import (
    CapabilityBlocker,
    CapabilityContractError,
    CapabilityMatchResult,
    CapabilityRequirement,
    CapabilityScope,
    ComparisonOperator,
    EvidenceLevel,
    FactExposure,
    FallbackAuthorizations,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    TargetCapabilitySnapshot,
    TargetIdentity,
)

DECISION_SCHEMA_VERSION = "flagquantum.runtime.target_capability_decision.v1"
SORTING_KEY_VERSION = "satisfied_preferences_then_rank_then_identity.v1"
ROUTE_INTENT_SCHEMA_VERSION = "flagquantum.runtime.route_intent.v1"
CANDIDATE_PROVENANCE_SCHEMA_VERSION = "flagquantum.runtime.candidate_provenance.v1"


class FallbackAxis(str, Enum):
    """Independent Runtime fallback authorization axes."""

    BACKEND = "backend"
    DEVICE = "device"
    CPU = "cpu"
    PRECISION = "precision"
    ALGORITHM = "algorithm"
    APPROXIMATION = "approximation"


class RuntimeDecisionBlockerCode(str, Enum):
    """Closed Runtime decision blockers; Core blockers stay nested unchanged."""

    CANDIDATE_CORE_MISMATCH = "candidate_core_mismatch"
    FALLBACK_AXIS_UNAUTHORIZED = "fallback_axis_unauthorized"
    CPU_CANDIDATE_REQUIRED = "cpu_candidate_required"
    CPU_IDENTITY_MISMATCH = "cpu_identity_mismatch"
    CPU_IDENTITY_UNVERIFIED = "cpu_identity_unverified"
    PRECISION_IDENTITY_UNVERIFIED = "precision_identity_unverified"
    ROUTE_INTENT_REQUIRED = "route_intent_required"
    CANDIDATE_PROVENANCE_REQUIRED = "candidate_provenance_required"
    PROVENANCE_IDENTITY_MISMATCH = "provenance_identity_mismatch"
    PROVENANCE_SNAPSHOT_MISMATCH = "provenance_snapshot_mismatch"
    DUPLICATE_CANDIDATE_ID = "duplicate_candidate_id"
    DUPLICATE_SNAPSHOT_IDENTITY = "duplicate_snapshot_identity"
    DUPLICATE_TARGET_IDENTITY = "duplicate_target_identity"
    NO_EXECUTABLE_CANDIDATE = "no_executable_candidate"


_ROUTE_FIELDS = (
    (FallbackAxis.BACKEND, "backend"),
    (FallbackAxis.DEVICE, "device_kind"),
    (FallbackAxis.CPU, "cpu"),
    (FallbackAxis.PRECISION, "effective_precision"),
    (FallbackAxis.ALGORITHM, "algorithm"),
    (FallbackAxis.APPROXIMATION, "approximation"),
)


def _route_value(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _stable_identity(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RouteIntent:
    """The original Runtime routing intent with a closed six-axis shape.

    ``effective_precision`` is the execution dtype expected from a candidate.
    This is an internal policy value.  It contains no provider handles or
    credentials; its identity is derived only from its normalized values.
    """

    backend: str
    device_kind: str
    cpu: bool
    effective_precision: str
    algorithm: str
    approximation: str
    intent_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "device_kind",
            "effective_precision",
            "algorithm",
            "approximation",
        ):
            object.__setattr__(
                self, name, _route_value(getattr(self, name), field_name=name)
            )
        if type(self.cpu) is not bool:
            raise TypeError("cpu must be a boolean")
        if self.cpu is not (self.device_kind == "cpu"):
            raise ValueError("cpu must agree with device_kind")
        if not isinstance(self.intent_id, str):
            raise TypeError("intent_id must be a string")
        if self.intent_id and not self.intent_id.strip():
            raise ValueError("intent_id must be empty or a non-empty string")
        object.__setattr__(self, "intent_id", self.intent_id.strip())
        if not self.intent_id:
            object.__setattr__(self, "intent_id", self.canonical_identity)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": ROUTE_INTENT_SCHEMA_VERSION,
            "backend": self.backend,
            "device_kind": self.device_kind,
            "cpu": self.cpu,
            "effective_precision": self.effective_precision,
            "algorithm": self.algorithm,
            "approximation": self.approximation,
        }

    @property
    def canonical_identity(self) -> str:
        return _stable_identity(self._identity_payload())

    @property
    def identity_valid(self) -> bool:
        return self.intent_id == self.canonical_identity

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "intent_id": self.intent_id}


@dataclass(frozen=True)
class CandidateProvenance:
    """Immutable route provenance bound to one intent and one snapshot.

    The binding and canonical identity detect accidental misattachment or
    mutation; they are not external source authentication.
    """

    intent_id: str
    snapshot_id: str
    backend: str
    device_kind: str
    cpu: bool
    effective_precision: str
    algorithm: str
    approximation: str
    provenance_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.intent_id, str) or not self.intent_id.strip():
            raise ValueError("intent_id must be a non-empty string")
        object.__setattr__(self, "intent_id", self.intent_id.strip())
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string")
        object.__setattr__(self, "snapshot_id", self.snapshot_id.strip())
        for name in (
            "backend",
            "device_kind",
            "effective_precision",
            "algorithm",
            "approximation",
        ):
            object.__setattr__(
                self, name, _route_value(getattr(self, name), field_name=name)
            )
        if type(self.cpu) is not bool:
            raise TypeError("cpu must be a boolean")
        if self.cpu is not (self.device_kind == "cpu"):
            raise ValueError("cpu must agree with device_kind")
        if not isinstance(self.provenance_id, str):
            raise TypeError("provenance_id must be a string")
        if self.provenance_id and not self.provenance_id.strip():
            raise ValueError("provenance_id must be empty or a non-empty string")
        object.__setattr__(self, "provenance_id", self.provenance_id.strip())
        if not self.provenance_id:
            object.__setattr__(self, "provenance_id", self.canonical_identity)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": CANDIDATE_PROVENANCE_SCHEMA_VERSION,
            "intent_id": self.intent_id,
            "snapshot_id": self.snapshot_id,
            "backend": self.backend,
            "device_kind": self.device_kind,
            "cpu": self.cpu,
            "effective_precision": self.effective_precision,
            "algorithm": self.algorithm,
            "approximation": self.approximation,
        }

    @property
    def canonical_identity(self) -> str:
        return _stable_identity(self._identity_payload())

    @property
    def identity_valid(self) -> bool:
        return self.provenance_id == self.canonical_identity

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "provenance_id": self.provenance_id}


@dataclass(frozen=True)
class TargetCapabilityCandidate:
    """One explicit target snapshot offered to Runtime policy.

    Runtime computes fallback axes from ``route_intent`` and ``provenance``;
    candidate metadata cannot assert, remove, or reinterpret an axis.
    """

    candidate_id: str
    snapshot: TargetCapabilitySnapshot
    preference_rank: int = 0
    provenance: CandidateProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.snapshot, TargetCapabilitySnapshot):
            raise TypeError("snapshot must be a Core TargetCapabilitySnapshot")
        if type(self.preference_rank) is not int or self.preference_rank < 0:
            raise ValueError("preference_rank must be a non-negative integer")
        if self.provenance is not None and not isinstance(
            self.provenance, CandidateProvenance
        ):
            raise TypeError("provenance must be a CandidateProvenance")


@dataclass(frozen=True)
class RuntimeDecisionBlocker:
    """A typed Runtime blocker with the exact Core blockers it explains."""

    code: RuntimeDecisionBlockerCode
    message: str
    candidate_id: str | None = None
    fallback_axes: tuple[FallbackAxis, ...] = ()
    core_blockers: tuple[CapabilityBlocker, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, RuntimeDecisionBlockerCode):
            object.__setattr__(self, "code", RuntimeDecisionBlockerCode(self.code))
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("decision blocker message must be non-empty")
        if self.candidate_id is not None and (
            not isinstance(self.candidate_id, str) or not self.candidate_id
        ):
            raise ValueError("decision blocker candidate_id must be non-empty")
        axes = tuple(
            sorted(
                {
                    axis if isinstance(axis, FallbackAxis) else FallbackAxis(axis)
                    for axis in self.fallback_axes
                },
                key=lambda item: item.value,
            )
        )
        object.__setattr__(self, "fallback_axes", axes)
        blockers = tuple(self.core_blockers)
        if any(not isinstance(item, CapabilityBlocker) for item in blockers):
            raise TypeError("core_blockers must contain Core CapabilityBlocker values")
        object.__setattr__(self, "core_blockers", blockers)


@dataclass(frozen=True)
class TargetCapabilityCandidateEvaluation:
    """One candidate's Core result plus Runtime authorization outcome."""

    candidate: TargetCapabilityCandidate
    core_match: CapabilityMatchResult
    blockers: tuple[RuntimeDecisionBlocker, ...]
    score: tuple[int, int, str, str, str]
    computed_fallback_axes: frozenset[FallbackAxis] = frozenset()

    @property
    def executable(self) -> bool:
        return self.core_match.executable and not self.blockers


@dataclass(frozen=True)
class FallbackDecisionRecord:
    """Private record for an accepted fallback candidate.

    This is deliberately separate from RuntimePlan/ExecutionResult.  A later
    attempt/evidence contract may consume it without changing those schemas.
    """

    requirement_set_id: str
    snapshot_id: str
    target_identity: TargetIdentity
    fallback_axes: tuple[FallbackAxis, ...]
    core_blockers: tuple[CapabilityBlocker, ...]
    score: tuple[int, int, str, str, str]
    decision_id: str
    candidate_snapshot_ids: tuple[str, ...]
    evaluated_at: str
    fallback_authorization_id: str
    sorting_key_version: str = SORTING_KEY_VERSION


@dataclass(frozen=True)
class TargetCapabilityDecision:
    """Deterministic candidate decision returned to a future Runtime caller."""

    requirement_set_id: str
    evaluations: tuple[TargetCapabilityCandidateEvaluation, ...]
    selected: TargetCapabilityCandidateEvaluation | None
    blockers: tuple[RuntimeDecisionBlocker, ...]
    fallback_record: FallbackDecisionRecord | None = None
    decision_id: str = ""
    evaluated_at: str = ""
    candidate_snapshot_ids: tuple[str, ...] = ()
    fallback_authorization_id: str = ""
    sorting_key_version: str = SORTING_KEY_VERSION

    @property
    def executable(self) -> bool:
        return self.selected is not None and self.selected.executable


def _axis_authorized(
    authorizations: FallbackAuthorizations, axis: FallbackAxis
) -> bool:
    return bool(getattr(authorizations, axis.value))


def _verified_string_fact(
    snapshot: TargetCapabilitySnapshot,
    *,
    capability_name: str,
    evaluated_at: datetime,
    expected_target_identity: TargetIdentity | None,
    required_scope: CapabilityScope | None,
    claim_minimum_evidence_level: EvidenceLevel,
    minimum_evidence_level: EvidenceLevel = EvidenceLevel.OBSERVABLE,
) -> tuple[str | None, tuple[CapabilityBlocker, ...]]:
    """Return one string fact only after Core validates its evidence.

    Missing, unsupported, unmeasured, unknown, or non-observed facts remain
    unresolved.  The policy requirement deliberately reuses Core's source,
    evidence, scope, freshness, and blocker semantics instead of duplicating an
    evidence matcher here.
    """

    fact = next((item for item in snapshot.facts if item.name == capability_name), None)
    if fact is None or not isinstance(fact.value, str):
        return None, ()
    identity_requirements = RequirementSet(
        requirements=(
            CapabilityRequirement(
                name=capability_name,
                operator=ComparisonOperator.EQUALS,
                value=fact.value,
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.RUNTIME_PROTOCOL,
                minimum_evidence_level=minimum_evidence_level,
                accepted_exposures=(FactExposure.OBSERVED,),
            ),
        )
    )
    identity_match = _core.match_target_capabilities(
        identity_requirements,
        snapshot,
        evaluated_at=evaluated_at,
        expected_target_identity=expected_target_identity,
        required_scope=required_scope,
        claim_minimum_evidence_level=claim_minimum_evidence_level,
    )
    if not identity_match.executable:
        return None, identity_match.blockers
    return fact.value, ()


def _cpu_identity(
    snapshot: TargetCapabilitySnapshot,
    *,
    evaluated_at: datetime,
    expected_target_identity: TargetIdentity | None,
    required_scope: CapabilityScope | None,
    claim_minimum_evidence_level: EvidenceLevel,
) -> tuple[bool | None, tuple[CapabilityBlocker, ...]]:
    device_kind, blockers = _verified_string_fact(
        snapshot,
        capability_name="device.kind",
        evaluated_at=evaluated_at,
        expected_target_identity=expected_target_identity,
        required_scope=required_scope,
        claim_minimum_evidence_level=claim_minimum_evidence_level,
    )
    return (None if device_kind is None else device_kind == "cpu"), blockers


def _effective_precision_identity(
    snapshot: TargetCapabilitySnapshot,
    *,
    evaluated_at: datetime,
    expected_target_identity: TargetIdentity | None,
    required_scope: CapabilityScope | None,
    claim_minimum_evidence_level: EvidenceLevel,
) -> tuple[str | None, tuple[CapabilityBlocker, ...]]:
    return _verified_string_fact(
        snapshot,
        capability_name="precision.effective_dtype",
        evaluated_at=evaluated_at,
        expected_target_identity=expected_target_identity,
        required_scope=required_scope,
        claim_minimum_evidence_level=claim_minimum_evidence_level,
    )


def _precision_path_blockers(
    candidate: TargetCapabilityCandidate,
    *,
    effective_precision: str | None,
    evaluated_at: datetime,
    expected_target_identity: TargetIdentity | None,
    required_scope: CapabilityScope | None,
    claim_minimum_evidence_level: EvidenceLevel,
) -> tuple[RuntimeDecisionBlocker, ...]:
    """Require a complete, certified account of software-expanded precision."""

    if effective_precision is None:
        return ()

    precision_path_names = (
        "precision.native_dtype",
        "precision.storage_dtype",
        "precision.software_mechanism",
    )
    facts = {fact.name: fact for fact in candidate.snapshot.facts}
    missing = tuple(
        name
        for name in precision_path_names
        if name not in facts or not isinstance(facts[name].value, str)
    )
    if missing:
        return (
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PRECISION_IDENTITY_UNVERIFIED,
                message=(
                    "candidate precision path is incomplete; missing verified "
                    + ", ".join(missing)
                ),
                candidate_id=candidate.candidate_id,
                fallback_axes=(FallbackAxis.PRECISION,),
            ),
        )

    path_requirements = RequirementSet(
        requirements=tuple(
            CapabilityRequirement(
                name=name,
                operator=ComparisonOperator.EQUALS,
                value=facts[name].value,
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.RUNTIME_PROTOCOL,
                minimum_evidence_level=EvidenceLevel.OBSERVABLE,
                accepted_exposures=(FactExposure.OBSERVED,),
            )
            for name in precision_path_names
        )
    )
    path_match = _core.match_target_capabilities(
        path_requirements,
        candidate.snapshot,
        evaluated_at=evaluated_at,
        expected_target_identity=expected_target_identity,
        required_scope=required_scope,
        claim_minimum_evidence_level=claim_minimum_evidence_level,
    )
    if not path_match.executable:
        return (
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PRECISION_IDENTITY_UNVERIFIED,
                message="candidate precision path is not backed by observed evidence",
                candidate_id=candidate.candidate_id,
                fallback_axes=(FallbackAxis.PRECISION,),
                core_blockers=path_match.blockers,
            ),
        )

    native = facts["precision.native_dtype"].value
    storage = facts["precision.storage_dtype"].value
    mechanism = facts["precision.software_mechanism"].value
    software_expanded = (
        mechanism != "none"
        or native != effective_precision
        or storage != effective_precision
    )
    if not software_expanded:
        return ()
    if mechanism == "none":
        return (
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PRECISION_IDENTITY_UNVERIFIED,
                message=(
                    "candidate precision dtypes imply software expansion but "
                    "precision.software_mechanism is none"
                ),
                candidate_id=candidate.candidate_id,
                fallback_axes=(FallbackAxis.PRECISION,),
            ),
        )

    certified_effective, certification_blockers = _verified_string_fact(
        candidate.snapshot,
        capability_name="precision.effective_dtype",
        evaluated_at=evaluated_at,
        expected_target_identity=expected_target_identity,
        required_scope=required_scope,
        claim_minimum_evidence_level=claim_minimum_evidence_level,
        minimum_evidence_level=EvidenceLevel.CERTIFICATION,
    )
    if (
        certified_effective != effective_precision
        or candidate.snapshot.scope.dtype != effective_precision
    ):
        return (
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PRECISION_IDENTITY_UNVERIFIED,
                message=(
                    "software-expanded precision requires certification-level "
                    "effective_dtype evidence scoped to that dtype"
                ),
                candidate_id=candidate.candidate_id,
                fallback_axes=(FallbackAxis.PRECISION,),
                core_blockers=certification_blockers,
            ),
        )
    return ()


def _requires_cpu(requirements: RequirementSet) -> bool:
    """Whether the original mandatory request explicitly requires a CPU."""

    return any(
        requirement.strength is RequirementStrength.MANDATORY
        and requirement.name == "device.kind"
        and requirement.operator is ComparisonOperator.EQUALS
        and requirement.value == "cpu"
        for requirement in requirements.requirements
    )


def _computed_fallback_axes(
    route_intent: RouteIntent, provenance: CandidateProvenance
) -> frozenset[FallbackAxis]:
    """Compute all changed route axes; caller assertions are never consulted."""

    return frozenset(
        axis
        for axis, field_name in _ROUTE_FIELDS
        if getattr(route_intent, field_name) != getattr(provenance, field_name)
    )


def _provenance_blockers(
    candidate: TargetCapabilityCandidate,
    *,
    route_intent: RouteIntent | None,
) -> tuple[RuntimeDecisionBlocker, ...]:
    blockers: list[RuntimeDecisionBlocker] = []
    provenance = candidate.provenance
    if route_intent is None:
        blockers.append(
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.ROUTE_INTENT_REQUIRED,
                message="candidate selection requires an immutable original route intent",
                candidate_id=candidate.candidate_id,
            )
        )
        return tuple(blockers)
    if not route_intent.identity_valid or provenance is None:
        blockers.append(
            RuntimeDecisionBlocker(
                code=(
                    RuntimeDecisionBlockerCode.CANDIDATE_PROVENANCE_REQUIRED
                    if provenance is None
                    else RuntimeDecisionBlockerCode.PROVENANCE_IDENTITY_MISMATCH
                ),
                message=(
                    "candidate provenance is required and must bind the route intent"
                    if provenance is None
                    else "route intent identity is not canonical"
                ),
                candidate_id=candidate.candidate_id,
            )
        )
        return tuple(blockers)
    if not provenance.identity_valid or provenance.intent_id != route_intent.intent_id:
        blockers.append(
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PROVENANCE_IDENTITY_MISMATCH,
                message="candidate provenance identity does not bind the route intent",
                candidate_id=candidate.candidate_id,
            )
        )
    return tuple(blockers)


def _provenance_snapshot_blockers(
    candidate: TargetCapabilityCandidate,
    *,
    provenance: CandidateProvenance | None,
    cpu_identity: bool | None,
    effective_precision: str | None,
    effective_precision_core_blockers: tuple[CapabilityBlocker, ...],
) -> tuple[RuntimeDecisionBlocker, ...]:
    """Cross-check provenance claims against Core-validated snapshot facts."""

    if provenance is None:
        return ()
    blockers: list[RuntimeDecisionBlocker] = []
    if provenance.snapshot_id != candidate.snapshot.snapshot_id:
        blockers.append(
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PROVENANCE_SNAPSHOT_MISMATCH,
                message="candidate provenance is bound to a different snapshot identity",
                candidate_id=candidate.candidate_id,
            )
        )
    if effective_precision is None:
        blockers.append(
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PRECISION_IDENTITY_UNVERIFIED,
                message=(
                    "candidate effective precision is not verifiable by Core; "
                    "Runtime cannot assume the precision"
                ),
                candidate_id=candidate.candidate_id,
                fallback_axes=(FallbackAxis.PRECISION,),
                core_blockers=effective_precision_core_blockers,
            )
        )
    elif provenance.effective_precision != effective_precision:
        blockers.append(
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PROVENANCE_SNAPSHOT_MISMATCH,
                message="provenance effective precision conflicts with the Core fact",
                candidate_id=candidate.candidate_id,
                fallback_axes=(FallbackAxis.PRECISION,),
            )
        )
    if cpu_identity is not None and provenance.cpu is not cpu_identity:
        blockers.append(
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PROVENANCE_SNAPSHOT_MISMATCH,
                message="provenance CPU axis conflicts with verified device.kind fact",
                candidate_id=candidate.candidate_id,
                fallback_axes=(FallbackAxis.CPU,),
            )
        )
    device_kind_fact = next(
        (item for item in candidate.snapshot.facts if item.name == "device.kind"),
        None,
    )
    if cpu_identity is not None and (
        device_kind_fact is None
        or not isinstance(device_kind_fact.value, str)
        or provenance.device_kind != device_kind_fact.value
    ):
        blockers.append(
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.PROVENANCE_SNAPSHOT_MISMATCH,
                message="provenance device identity conflicts with the verified device.kind fact",
                candidate_id=candidate.candidate_id,
                fallback_axes=(FallbackAxis.DEVICE,),
            )
        )
    return tuple(blockers)


def _cpu_identity_blockers(
    candidate: TargetCapabilityCandidate,
    *,
    cpu_identity: bool | None,
    cpu_identity_core_blockers: tuple[CapabilityBlocker, ...],
    request_requires_cpu: bool,
    computed_fallback_axes: frozenset[FallbackAxis],
) -> tuple[RuntimeDecisionBlocker, ...]:
    blockers: list[RuntimeDecisionBlocker] = []
    declares_cpu = FallbackAxis.CPU in computed_fallback_axes
    if cpu_identity is None:
        blockers.append(
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.CPU_IDENTITY_UNVERIFIED,
                message=(
                    "candidate device.kind identity is not verifiable by Core; "
                    "Runtime cannot assume it is non-CPU"
                ),
                candidate_id=candidate.candidate_id,
                fallback_axes=((FallbackAxis.CPU,) if declares_cpu else ()),
                core_blockers=cpu_identity_core_blockers,
            )
        )
        return tuple(blockers)
    if cpu_identity is False:
        if declares_cpu:
            blockers.append(
                RuntimeDecisionBlocker(
                    code=RuntimeDecisionBlockerCode.CPU_IDENTITY_MISMATCH,
                    message="candidate CPU metadata conflicts with device.kind fact",
                    candidate_id=candidate.candidate_id,
                    fallback_axes=((FallbackAxis.CPU,) if declares_cpu else ()),
                )
            )
        return tuple(blockers)
    if not request_requires_cpu and not declares_cpu:
        blockers.append(
            RuntimeDecisionBlocker(
                code=RuntimeDecisionBlockerCode.CPU_CANDIDATE_REQUIRED,
                message=(
                    "a CPU snapshot is a fallback unless the original request "
                    "explicitly requires CPU; declare the cpu fallback axis"
                ),
                candidate_id=candidate.candidate_id,
                fallback_axes=(FallbackAxis.CPU,),
            )
        )
    return tuple(blockers)


def _canonical_evaluation_time(value: datetime | None) -> tuple[datetime, str]:
    selected = value or datetime.now(timezone.utc)
    if selected.tzinfo is None:
        # Keep Core's contract error family for invalid evaluation context;
        # Runtime must not reinterpret a fail-closed validation failure.
        raise CapabilityContractError("evaluated_at must include a timezone")
    selected = selected.astimezone(timezone.utc)
    # ISO-8601 is the only time representation included in the decision
    # identity.  The Core matcher receives the same timezone-aware instant.
    return selected, selected.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _authorization_identity(authorizations: FallbackAuthorizations) -> str:
    payload = authorizations.to_dict()
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _target_identity_key(identity: TargetIdentity) -> tuple[str, ...]:
    return (
        identity.target_id,
        identity.target_class,
        identity.provider,
        identity.provider_version,
        identity.target_revision,
        identity.environment_id,
    )


def _runtime_blocker_payload(blocker: RuntimeDecisionBlocker) -> dict[str, object]:
    return {
        "code": blocker.code.value,
        "message": blocker.message,
        "candidate_id": blocker.candidate_id,
        "fallback_axes": [item.value for item in blocker.fallback_axes],
        "core_blockers": [item.to_dict() for item in blocker.core_blockers],
    }


def _decision_identity(
    requirements: RequirementSet,
    evaluations: tuple[TargetCapabilityCandidateEvaluation, ...],
    selected: TargetCapabilityCandidateEvaluation | None,
    blockers: tuple[RuntimeDecisionBlocker, ...],
    *,
    evaluated_at: str,
    fallback_authorization_id: str,
    route_intent: RouteIntent | None,
) -> str:
    payload = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "requirement_set_id": requirements.requirement_set_id,
        "candidate_snapshot_ids": [
            item.candidate.snapshot.snapshot_id for item in evaluations
        ],
        "selected_snapshot_id": (
            selected.candidate.snapshot.snapshot_id if selected is not None else None
        ),
        "selected_target_identity": (
            selected.candidate.snapshot.target_identity.to_dict()
            if selected is not None
            else None
        ),
        "evaluated_at": evaluated_at,
        "fallback_authorization_id": fallback_authorization_id,
        "route_intent": (route_intent.to_dict() if route_intent is not None else None),
        "sorting_key_version": SORTING_KEY_VERSION,
        "evaluations": [
            {
                "candidate_id": item.candidate.candidate_id,
                "snapshot_id": item.candidate.snapshot.snapshot_id,
                "preference_rank": item.candidate.preference_rank,
                "fallback_axes": [
                    axis.value
                    for axis in sorted(
                        item.computed_fallback_axes, key=lambda axis: axis.value
                    )
                ],
                "provenance": (
                    item.candidate.provenance.to_dict()
                    if item.candidate.provenance is not None
                    else None
                ),
                "core_executable": item.core_match.executable,
                "core_blockers": [
                    blocker.to_dict() for blocker in item.core_match.blockers
                ],
                "runtime_blockers": [
                    _runtime_blocker_payload(blocker) for blocker in item.blockers
                ],
                "score": list(item.score),
            }
            for item in evaluations
        ],
        "blockers": [_runtime_blocker_payload(item) for item in blockers],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _candidate_sort_key(
    candidate: TargetCapabilityCandidate,
) -> tuple[str, str, str]:
    identity = candidate.snapshot.target_identity
    return (identity.target_id, candidate.snapshot.snapshot_id, candidate.candidate_id)


def _score(
    candidate: TargetCapabilityCandidate,
    core_match: CapabilityMatchResult,
) -> tuple[int, int, str, str, str]:
    identity = candidate.snapshot.target_identity
    # More satisfied preferences win; explicit rank and identities are stable
    # tie-breaks and never make an ineligible candidate executable.
    return (
        -core_match.satisfied_preferences,
        candidate.preference_rank,
        identity.target_id,
        candidate.snapshot.snapshot_id,
        candidate.candidate_id,
    )


def _core_mismatch_blocker(
    candidate: TargetCapabilityCandidate, result: CapabilityMatchResult
) -> RuntimeDecisionBlocker:
    return RuntimeDecisionBlocker(
        code=RuntimeDecisionBlockerCode.CANDIDATE_CORE_MISMATCH,
        message="Core matcher rejected the candidate; Core blockers are preserved",
        candidate_id=candidate.candidate_id,
        core_blockers=result.blockers,
    )


def _unauthorized_axes_blocker(
    candidate: TargetCapabilityCandidate,
    authorizations: FallbackAuthorizations,
    computed_fallback_axes: frozenset[FallbackAxis],
) -> RuntimeDecisionBlocker | None:
    unauthorized = tuple(
        sorted(
            (
                axis
                for axis in computed_fallback_axes
                if not _axis_authorized(authorizations, axis)
            ),
            key=lambda item: item.value,
        )
    )
    if not unauthorized:
        return None
    return RuntimeDecisionBlocker(
        code=RuntimeDecisionBlockerCode.FALLBACK_AXIS_UNAUTHORIZED,
        message="computed candidate fallback axes are not authorized",
        candidate_id=candidate.candidate_id,
        fallback_axes=unauthorized,
    )


def match_target_capability_candidates(
    requirements: RequirementSet,
    candidates: Iterable[TargetCapabilityCandidate],
    *,
    evaluated_at: datetime | None = None,
    expected_target_identity: TargetIdentity | None = None,
    required_scope: CapabilityScope | None = None,
    claim_minimum_evidence_level: EvidenceLevel = EvidenceLevel.BASIC,
    route_intent: RouteIntent | None = None,
) -> TargetCapabilityDecision:
    """Match and rank explicit snapshots without touching the execution path.

    Every candidate is passed to Core's pure matcher with the same
    ``RequirementSet``.  Runtime only filters on Core's executable result,
    checks independent fallback authorization, and ranks eligible candidates.
    In particular, it never rewrites a device/precision requirement to make a
    CPU or other fallback candidate pass.
    """

    if not isinstance(requirements, RequirementSet):
        raise TypeError("requirements must be a Core RequirementSet")
    evaluation_instant, evaluation_time = _canonical_evaluation_time(evaluated_at)
    fallback_authorization_id = _authorization_identity(
        requirements.fallback_authorizations
    )
    candidate_values = tuple(candidates)
    if not candidate_values:
        blocker = RuntimeDecisionBlocker(
            code=RuntimeDecisionBlockerCode.NO_EXECUTABLE_CANDIDATE,
            message="Runtime candidate set is empty",
        )
        decision_id = _decision_identity(
            requirements,
            (),
            None,
            (blocker,),
            evaluated_at=evaluation_time,
            fallback_authorization_id=fallback_authorization_id,
            route_intent=route_intent,
        )
        return TargetCapabilityDecision(
            requirement_set_id=requirements.requirement_set_id,
            evaluations=(),
            selected=None,
            blockers=(blocker,),
            decision_id=decision_id,
            evaluated_at=evaluation_time,
            fallback_authorization_id=fallback_authorization_id,
        )
    if any(
        not isinstance(item, TargetCapabilityCandidate) for item in candidate_values
    ):
        raise TypeError("candidates must contain TargetCapabilityCandidate values")

    ordered = tuple(sorted(candidate_values, key=_candidate_sort_key))
    target_identity_groups: dict[tuple[str, ...], set[str]] = {}
    snapshot_identity_groups: dict[str, set[str]] = {}
    for item in ordered:
        target_identity_groups.setdefault(
            _target_identity_key(item.snapshot.target_identity), set()
        ).add(item.snapshot.snapshot_id)
        snapshot_identity_groups.setdefault(item.snapshot.snapshot_id, set()).add(
            item.candidate_id
        )
    conflicting_target_identities = {
        key
        for key, snapshot_ids in target_identity_groups.items()
        if len(snapshot_ids) > 1
    }
    duplicate_snapshot_ids = {
        snapshot_id
        for snapshot_id, candidate_ids in snapshot_identity_groups.items()
        if len(candidate_ids) > 1
    }
    candidate_id_groups: dict[str, int] = {}
    for item in ordered:
        candidate_id_groups[item.candidate_id] = (
            candidate_id_groups.get(item.candidate_id, 0) + 1
        )
    duplicate_candidate_ids = {
        candidate_id for candidate_id, count in candidate_id_groups.items() if count > 1
    }
    request_requires_cpu = _requires_cpu(requirements)
    evaluations: list[TargetCapabilityCandidateEvaluation] = []
    for candidate in ordered:
        # Core matching is deliberately called before Runtime authorization;
        # unauthorized alternatives cannot hide stale/scope/identity blockers.
        core_match = _core.match_target_capabilities(
            requirements,
            candidate.snapshot,
            evaluated_at=evaluation_instant,
            expected_target_identity=expected_target_identity,
            required_scope=required_scope,
            claim_minimum_evidence_level=claim_minimum_evidence_level,
        )
        candidate_blockers: list[RuntimeDecisionBlocker] = []
        provenance = candidate.provenance
        if (
            route_intent is not None
            and provenance is not None
            and route_intent.identity_valid
            and provenance.identity_valid
            and provenance.intent_id == route_intent.intent_id
        ):
            computed_axes = _computed_fallback_axes(route_intent, provenance)
        else:
            computed_axes = frozenset()
        candidate_blockers.extend(
            _provenance_blockers(candidate, route_intent=route_intent)
        )
        if candidate.candidate_id in duplicate_candidate_ids:
            candidate_blockers.append(
                RuntimeDecisionBlocker(
                    code=RuntimeDecisionBlockerCode.DUPLICATE_CANDIDATE_ID,
                    message=(
                        "candidate_id is shared by multiple candidates; all "
                        "conflicting members are rejected"
                    ),
                    candidate_id=candidate.candidate_id,
                )
            )
        if (
            _target_identity_key(candidate.snapshot.target_identity)
            in conflicting_target_identities
        ):
            candidate_blockers.append(
                RuntimeDecisionBlocker(
                    code=RuntimeDecisionBlockerCode.DUPLICATE_TARGET_IDENTITY,
                    message=(
                        "one target identity was offered with different snapshot "
                        "semantics"
                    ),
                    candidate_id=candidate.candidate_id,
                )
            )
        if candidate.snapshot.snapshot_id in duplicate_snapshot_ids:
            candidate_blockers.append(
                RuntimeDecisionBlocker(
                    code=RuntimeDecisionBlockerCode.DUPLICATE_SNAPSHOT_IDENTITY,
                    message="one snapshot identity was offered more than once",
                    candidate_id=candidate.candidate_id,
                )
            )
        cpu_identity, cpu_identity_core_blockers = _cpu_identity(
            candidate.snapshot,
            evaluated_at=evaluation_instant,
            expected_target_identity=expected_target_identity,
            required_scope=required_scope,
            claim_minimum_evidence_level=claim_minimum_evidence_level,
        )
        effective_precision, effective_precision_core_blockers = (
            _effective_precision_identity(
                candidate.snapshot,
                evaluated_at=evaluation_instant,
                expected_target_identity=expected_target_identity,
                required_scope=required_scope,
                claim_minimum_evidence_level=claim_minimum_evidence_level,
            )
        )
        candidate_blockers.extend(
            _cpu_identity_blockers(
                candidate,
                cpu_identity=cpu_identity,
                cpu_identity_core_blockers=cpu_identity_core_blockers,
                request_requires_cpu=request_requires_cpu,
                computed_fallback_axes=computed_axes,
            )
        )
        candidate_blockers.extend(
            _provenance_snapshot_blockers(
                candidate,
                provenance=provenance,
                cpu_identity=cpu_identity,
                effective_precision=effective_precision,
                effective_precision_core_blockers=effective_precision_core_blockers,
            )
        )
        candidate_blockers.extend(
            _precision_path_blockers(
                candidate,
                effective_precision=effective_precision,
                evaluated_at=evaluation_instant,
                expected_target_identity=expected_target_identity,
                required_scope=required_scope,
                claim_minimum_evidence_level=claim_minimum_evidence_level,
            )
        )
        unauthorized = _unauthorized_axes_blocker(
            candidate, requirements.fallback_authorizations, computed_axes
        )
        if unauthorized is not None:
            candidate_blockers.append(unauthorized)
        if not core_match.executable:
            candidate_blockers.append(_core_mismatch_blocker(candidate, core_match))
        evaluations.append(
            TargetCapabilityCandidateEvaluation(
                candidate=candidate,
                core_match=core_match,
                blockers=tuple(candidate_blockers),
                score=_score(candidate, core_match),
                computed_fallback_axes=computed_axes,
            )
        )

    eligible = tuple(item for item in evaluations if item.executable)
    selected = min(eligible, key=lambda item: item.score) if eligible else None
    if selected is None:
        blockers = tuple(blocker for item in evaluations for blocker in item.blockers)
        if not blockers:
            blockers = (
                RuntimeDecisionBlocker(
                    code=RuntimeDecisionBlockerCode.NO_EXECUTABLE_CANDIDATE,
                    message="no candidate satisfied all mandatory requirements",
                ),
            )
        decision_id = _decision_identity(
            requirements,
            tuple(evaluations),
            None,
            blockers,
            evaluated_at=evaluation_time,
            fallback_authorization_id=fallback_authorization_id,
            route_intent=route_intent,
        )
        return TargetCapabilityDecision(
            requirement_set_id=requirements.requirement_set_id,
            evaluations=evaluations,
            selected=None,
            blockers=blockers,
            decision_id=decision_id,
            evaluated_at=evaluation_time,
            candidate_snapshot_ids=tuple(
                item.candidate.snapshot.snapshot_id for item in evaluations
            ),
            fallback_authorization_id=fallback_authorization_id,
        )

    decision_id = _decision_identity(
        requirements,
        tuple(evaluations),
        selected,
        (),
        evaluated_at=evaluation_time,
        fallback_authorization_id=fallback_authorization_id,
        route_intent=route_intent,
    )
    fallback_record = None
    if selected.computed_fallback_axes:
        fallback_record = FallbackDecisionRecord(
            requirement_set_id=requirements.requirement_set_id,
            snapshot_id=selected.candidate.snapshot.snapshot_id,
            target_identity=selected.candidate.snapshot.target_identity,
            fallback_axes=tuple(
                sorted(selected.computed_fallback_axes, key=lambda item: item.value)
            ),
            core_blockers=selected.core_match.blockers,
            score=selected.score,
            decision_id=decision_id,
            candidate_snapshot_ids=tuple(
                item.candidate.snapshot.snapshot_id for item in evaluations
            ),
            evaluated_at=evaluation_time,
            fallback_authorization_id=fallback_authorization_id,
        )
    return TargetCapabilityDecision(
        requirement_set_id=requirements.requirement_set_id,
        evaluations=evaluations,
        selected=selected,
        blockers=(),
        fallback_record=fallback_record,
        decision_id=decision_id,
        evaluated_at=evaluation_time,
        candidate_snapshot_ids=tuple(
            item.candidate.snapshot.snapshot_id for item in evaluations
        ),
        fallback_authorization_id=fallback_authorization_id,
    )


# A concise alias is useful to private Runtime callers while keeping the
# longer name discoverable in code review.
select_target_capability_candidate = match_target_capability_candidates


__all__ = [
    "CANDIDATE_PROVENANCE_SCHEMA_VERSION",
    "DECISION_SCHEMA_VERSION",
    "CandidateProvenance",
    "FallbackAxis",
    "FallbackDecisionRecord",
    "RuntimeDecisionBlocker",
    "RuntimeDecisionBlockerCode",
    "ROUTE_INTENT_SCHEMA_VERSION",
    "RouteIntent",
    "SORTING_KEY_VERSION",
    "TargetCapabilityCandidate",
    "TargetCapabilityCandidateEvaluation",
    "TargetCapabilityDecision",
    "match_target_capability_candidates",
    "select_target_capability_candidate",
]
