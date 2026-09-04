"""Runtime-owned candidate matching over the Core Target Capabilities seam.

This module is intentionally not imported by the default execution path.  It
owns candidate ordering, fallback authorization and a private decision record;
Core owns requirement/snapshot validation and pure comparison.  No value is
copied into or rewritten on either Core object.
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
    CapabilityScope,
    EvidenceLevel,
    FallbackAuthorizations,
    RequirementSet,
    TargetCapabilitySnapshot,
    TargetIdentity,
)

DECISION_SCHEMA_VERSION = "flagquantum.runtime.target_capability_decision.v1"
SORTING_KEY_VERSION = "satisfied_preferences_then_rank_then_identity.v1"


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
    DUPLICATE_CANDIDATE_ID = "duplicate_candidate_id"
    DUPLICATE_SNAPSHOT_IDENTITY = "duplicate_snapshot_identity"
    DUPLICATE_TARGET_IDENTITY = "duplicate_target_identity"
    NO_EXECUTABLE_CANDIDATE = "no_executable_candidate"


@dataclass(frozen=True)
class TargetCapabilityCandidate:
    """One explicit target snapshot offered to Runtime policy.

    ``fallback_axes`` describes what changed relative to the original request;
    it does not modify that request or the snapshot.  A CPU candidate is
    explicitly marked so device resolution cannot silently manufacture one.
    """

    candidate_id: str
    snapshot: TargetCapabilitySnapshot
    preference_rank: int = 0
    fallback_axes: frozenset[FallbackAxis] = frozenset()
    is_cpu_candidate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.snapshot, TargetCapabilitySnapshot):
            raise TypeError("snapshot must be a Core TargetCapabilitySnapshot")
        if type(self.preference_rank) is not int or self.preference_rank < 0:
            raise ValueError("preference_rank must be a non-negative integer")
        axes: set[FallbackAxis] = set()
        for axis in self.fallback_axes:
            try:
                axes.add(axis if isinstance(axis, FallbackAxis) else FallbackAxis(axis))
            except (TypeError, ValueError) as error:
                allowed = ", ".join(item.value for item in FallbackAxis)
                raise ValueError(f"fallback_axes must use only: {allowed}") from error
        object.__setattr__(self, "fallback_axes", frozenset(axes))
        if type(self.is_cpu_candidate) is not bool:
            raise TypeError("is_cpu_candidate must be a boolean")


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
        "sorting_key_version": SORTING_KEY_VERSION,
        "evaluations": [
            {
                "candidate_id": item.candidate.candidate_id,
                "snapshot_id": item.candidate.snapshot.snapshot_id,
                "preference_rank": item.candidate.preference_rank,
                "fallback_axes": [
                    axis.value
                    for axis in sorted(
                        item.candidate.fallback_axes, key=lambda axis: axis.value
                    )
                ],
                "is_cpu_candidate": item.candidate.is_cpu_candidate,
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
) -> RuntimeDecisionBlocker | None:
    unauthorized = tuple(
        sorted(
            (
                axis
                for axis in candidate.fallback_axes
                if not _axis_authorized(authorizations, axis)
            ),
            key=lambda item: item.value,
        )
    )
    if not unauthorized:
        return None
    return RuntimeDecisionBlocker(
        code=RuntimeDecisionBlockerCode.FALLBACK_AXIS_UNAUTHORIZED,
        message="candidate declares fallback axes that are not authorized",
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
    seen_ids: set[str] = set()
    evaluations: list[TargetCapabilityCandidateEvaluation] = []
    for candidate in ordered:
        duplicate = candidate.candidate_id in seen_ids
        seen_ids.add(candidate.candidate_id)
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
        if duplicate:
            candidate_blockers.append(
                RuntimeDecisionBlocker(
                    code=RuntimeDecisionBlockerCode.DUPLICATE_CANDIDATE_ID,
                    message="candidate_id must be unique for deterministic selection",
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
        if (
            FallbackAxis.CPU in candidate.fallback_axes
            and not candidate.is_cpu_candidate
        ):
            candidate_blockers.append(
                RuntimeDecisionBlocker(
                    code=RuntimeDecisionBlockerCode.CPU_CANDIDATE_REQUIRED,
                    message="the cpu fallback axis requires an explicit CPU candidate",
                    candidate_id=candidate.candidate_id,
                    fallback_axes=(FallbackAxis.CPU,),
                )
            )
        unauthorized = _unauthorized_axes_blocker(
            candidate, requirements.fallback_authorizations
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
    )
    fallback_record = None
    if selected.candidate.fallback_axes:
        fallback_record = FallbackDecisionRecord(
            requirement_set_id=requirements.requirement_set_id,
            snapshot_id=selected.candidate.snapshot.snapshot_id,
            target_identity=selected.candidate.snapshot.target_identity,
            fallback_axes=tuple(
                sorted(selected.candidate.fallback_axes, key=lambda item: item.value)
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
    "DECISION_SCHEMA_VERSION",
    "FallbackAxis",
    "FallbackDecisionRecord",
    "RuntimeDecisionBlocker",
    "RuntimeDecisionBlockerCode",
    "SORTING_KEY_VERSION",
    "TargetCapabilityCandidate",
    "TargetCapabilityCandidateEvaluation",
    "TargetCapabilityDecision",
    "match_target_capability_candidates",
    "select_target_capability_candidate",
]
