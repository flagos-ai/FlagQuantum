"""Pure matching for the internal Target Capabilities v1 value objects."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from .target_capabilities import (
    _EVIDENCE_RANK,
    AUTHORITATIVE_STATIC_DECLARATION_ALLOWED,
    OBSERVED_REQUIRED,
    CapabilityBlocker,
    CapabilityContractError,
    CapabilityFact,
    CapabilityMatchResult,
    CapabilityRequirement,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    MatchBlockerCode,
    RequirementSet,
    RequirementStrength,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
    _blocker_key,
    _compare_values,
    _enum,
    _parse_timestamp,
)


def match_target_capabilities(
    requirements: RequirementSet,
    snapshot: TargetCapabilitySnapshot,
    *,
    evaluated_at: datetime | None = None,
    expected_target_identity: TargetIdentity | None = None,
    required_scope: CapabilityScope | None = None,
    claim_minimum_evidence_level: EvidenceLevel = EvidenceLevel.BASIC,
) -> CapabilityMatchResult:
    """Compare one requirement set and snapshot without policy or fallback."""

    if not isinstance(requirements, RequirementSet) or not isinstance(
        snapshot, TargetCapabilitySnapshot
    ):
        raise TypeError(
            "requirements and snapshot must be Core Target Capabilities v1 objects"
        )
    claim_level = _enum(
        EvidenceLevel,
        claim_minimum_evidence_level,
        field_name="claim_minimum_evidence_level",
    )
    now = evaluated_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise CapabilityContractError("evaluated_at must include a timezone")
    now = now.astimezone(timezone.utc)
    blockers: list[CapabilityBlocker] = []
    if (
        expected_target_identity is not None
        and snapshot.target_identity != expected_target_identity
    ):
        blockers.append(
            _match_blocker(
                MatchBlockerCode.TARGET_IDENTITY_MISMATCH,
                "snapshot target identity does not match the requested target",
            )
        )
    if required_scope is not None and not _scope_covers(snapshot.scope, required_scope):
        blockers.append(
            _match_blocker(
                MatchBlockerCode.SCOPE_MISMATCH,
                "snapshot scope does not match every requested scope field",
            )
        )
    captured = _parse_timestamp(snapshot.captured_at, field_name="captured_at")
    valid_until = _parse_timestamp(snapshot.valid_until, field_name="valid_until")
    if now < captured:
        blockers.append(
            _match_blocker(
                MatchBlockerCode.SNAPSHOT_NOT_YET_VALID,
                "snapshot was captured after the evaluation time",
            )
        )
    if now >= valid_until:
        blockers.append(
            _match_blocker(
                MatchBlockerCode.SNAPSHOT_STALE,
                "snapshot has reached or passed valid_until",
            )
        )
    if requirements.extensions:
        blockers.append(
            _match_blocker(
                MatchBlockerCode.UNKNOWN_EXTENSION_HANDLER,
                "RequirementSet contains extensions without a registered v1 matcher",
            )
        )

    facts = {fact.name: fact for fact in snapshot.facts}
    evidence = {item.evidence_id: item for item in snapshot.evidence_refs}
    satisfied_preferences = 0
    total_preferences = sum(
        item.strength is RequirementStrength.PREFERENCE
        for item in requirements.requirements
    )
    mandatory_thresholds = [
        requirement.minimum_evidence_level
        for requirement in requirements.requirements
        if requirement.strength is RequirementStrength.MANDATORY
    ]
    required_level = max(
        (claim_level, *mandatory_thresholds), key=_EVIDENCE_RANK.__getitem__
    )

    for requirement in requirements.requirements:
        failure = _match_requirement(
            requirement,
            facts.get(requirement.name),
            evidence,
            snapshot.scope,
            required_level,
        )
        if failure is None:
            if requirement.strength is RequirementStrength.PREFERENCE:
                satisfied_preferences += 1
        elif requirement.strength is RequirementStrength.MANDATORY:
            blockers.append(failure)

    blockers = sorted(blockers, key=_blocker_key)
    return CapabilityMatchResult(
        executable=not blockers,
        blockers=tuple(blockers),
        satisfied_preferences=satisfied_preferences,
        total_preferences=total_preferences,
    )


def _match_requirement(
    requirement: CapabilityRequirement,
    fact: CapabilityFact | None,
    evidence: Mapping[str, EvidenceReference],
    snapshot_scope: CapabilityScope,
    required_level: EvidenceLevel,
) -> CapabilityBlocker | None:
    name = requirement.name
    if fact is None:
        return _match_blocker(
            MatchBlockerCode.MISSING_FACT,
            "snapshot does not contain the required fact",
            name,
        )
    status_codes = {
        SupportStatus.UNKNOWN: MatchBlockerCode.FACT_UNKNOWN,
        SupportStatus.UNMEASURED: MatchBlockerCode.FACT_UNMEASURED,
        SupportStatus.UNSUPPORTED: MatchBlockerCode.FACT_UNSUPPORTED,
    }
    if fact.support_status is not SupportStatus.VERIFIED:
        return _match_blocker(
            status_codes[fact.support_status],
            f"fact support status is {fact.support_status.value}",
            name,
        )
    if fact.fact_exposure not in requirement.accepted_exposures:
        return _match_blocker(
            MatchBlockerCode.EXPOSURE_NOT_ACCEPTED,
            f"fact exposure {fact.fact_exposure.value} is not accepted",
            name,
        )
    if fact.fact_exposure is FactExposure.NOT_APPLICABLE:
        return _match_blocker(
            MatchBlockerCode.NOT_APPLICABLE,
            "v1 generic matching does not accept not_applicable facts",
            name,
        )
    if name in OBSERVED_REQUIRED and fact.fact_exposure is not FactExposure.OBSERVED:
        return _match_blocker(
            MatchBlockerCode.EXPOSURE_REQUIRES_OBSERVATION,
            "capability validation profile requires observed evidence",
            name,
        )
    if (
        fact.fact_exposure is FactExposure.DECLARED
        and name not in AUTHORITATIVE_STATIC_DECLARATION_ALLOWED
    ):
        return _match_blocker(
            MatchBlockerCode.DECLARATION_NOT_AUTHORITATIVE,
            "declared exposure is not authoritative for this capability",
            name,
        )
    evidence_ref = evidence.get(fact.source.ref)
    if evidence_ref is None:
        return _match_blocker(
            MatchBlockerCode.UNRESOLVED_EVIDENCE_REFERENCE,
            "fact source does not resolve to an evidence reference",
            name,
        )
    if not _scope_covers(evidence_ref.scope, snapshot_scope):
        return _match_blocker(
            MatchBlockerCode.SCOPE_MISMATCH,
            "evidence scope does not cover the fact scope",
            name,
        )
    threshold = max(
        required_level,
        requirement.minimum_evidence_level,
        key=_EVIDENCE_RANK.__getitem__,
    )
    if _EVIDENCE_RANK[evidence_ref.level] < _EVIDENCE_RANK[threshold]:
        return _match_blocker(
            MatchBlockerCode.INSUFFICIENT_EVIDENCE,
            f"evidence level {evidence_ref.level.value} is below required {threshold.value}",
            name,
        )
    if not _compare_values(requirement.operator, fact.value, requirement.value):
        return _match_blocker(
            MatchBlockerCode.VALUE_MISMATCH,
            f"fact does not satisfy {requirement.operator.value}",
            name,
        )
    return None


def _scope_covers(available: CapabilityScope, required: CapabilityScope) -> bool:
    for name in (
        "device_ids",
        "dtype",
        "kernel",
        "workload_id",
        "world_size",
        "node_count",
    ):
        required_value = getattr(required, name)
        if required_value is not None and getattr(available, name) != required_value:
            return False
    return True


def _match_blocker(
    code: MatchBlockerCode, message: str, capability_name: str | None = None
) -> CapabilityBlocker:
    return CapabilityBlocker(
        code=code.value, message=message, capability_name=capability_name
    )


__all__ = ["CapabilityMatchResult", "match_target_capabilities"]
