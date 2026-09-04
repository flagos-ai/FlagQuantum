"""Runtime-owned candidate matching over the Core Target Capabilities seam.

This module is intentionally not imported by the default execution path.  It
owns candidate ordering, fallback authorization and a private decision record;
Core owns requirement/snapshot validation and pure comparison.  No value is
copied into or rewritten on either Core object.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable

import flagquantum.core.target_capabilities as _core
from flagquantum.core.target_capabilities import (
    CapabilityBlocker,
    CapabilityMatchResult,
    CapabilityScope,
    EvidenceLevel,
    FallbackAuthorizations,
    RequirementSet,
    TargetCapabilitySnapshot,
    TargetIdentity,
)


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


@dataclass(frozen=True)
class TargetCapabilityDecision:
    """Deterministic candidate decision returned to a future Runtime caller."""

    requirement_set_id: str
    evaluations: tuple[TargetCapabilityCandidateEvaluation, ...]
    selected: TargetCapabilityCandidateEvaluation | None
    blockers: tuple[RuntimeDecisionBlocker, ...]
    fallback_record: FallbackDecisionRecord | None = None

    @property
    def executable(self) -> bool:
        return self.selected is not None and self.selected.executable


def _axis_authorized(
    authorizations: FallbackAuthorizations, axis: FallbackAxis
) -> bool:
    return bool(getattr(authorizations, axis.value))


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
    candidate_values = tuple(candidates)
    if not candidate_values:
        blocker = RuntimeDecisionBlocker(
            code=RuntimeDecisionBlockerCode.NO_EXECUTABLE_CANDIDATE,
            message="Runtime candidate set is empty",
        )
        return TargetCapabilityDecision(
            requirement_set_id=requirements.requirement_set_id,
            evaluations=(),
            selected=None,
            blockers=(blocker,),
        )
    if any(
        not isinstance(item, TargetCapabilityCandidate) for item in candidate_values
    ):
        raise TypeError("candidates must contain TargetCapabilityCandidate values")

    ordered = tuple(sorted(candidate_values, key=_candidate_sort_key))
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
            evaluated_at=evaluated_at,
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
        return TargetCapabilityDecision(
            requirement_set_id=requirements.requirement_set_id,
            evaluations=evaluations,
            selected=None,
            blockers=blockers,
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
        )
    return TargetCapabilityDecision(
        requirement_set_id=requirements.requirement_set_id,
        evaluations=evaluations,
        selected=selected,
        blockers=(),
        fallback_record=fallback_record,
    )


# A concise alias is useful to private Runtime callers while keeping the
# longer name discoverable in code review.
select_target_capability_candidate = match_target_capability_candidates


__all__ = [
    "FallbackAxis",
    "FallbackDecisionRecord",
    "RuntimeDecisionBlocker",
    "RuntimeDecisionBlockerCode",
    "TargetCapabilityCandidate",
    "TargetCapabilityCandidateEvaluation",
    "TargetCapabilityDecision",
    "match_target_capability_candidates",
    "select_target_capability_candidate",
]
