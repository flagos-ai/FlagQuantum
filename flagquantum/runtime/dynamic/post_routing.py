"""Pure, internal post-routing capability rematch for dynamic programs.

The module deliberately owns no compiler, provider, credential, package, or
transport boundary.  It checks one preserved source requirement set and one
exact target snapshot, then combines that result with route observations from
the boundary that owns routing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from ...core.target_capabilities import (
    CapabilityBlocker,
    RequirementSet,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from .remote_capabilities import (
    DYNAMIC_REMOTE_EXTENSION,
    _active_requirement,
    _value_satisfies,
    match_dynamic_remote_capabilities,
)

_ROUTE_FACT_STATES = frozenset(
    {"supported", "unsupported", "unknown", "unmeasured", "not_exposed", "missing"}
)
_PROVIDER_LEGALITY_STATES = frozenset({"accepted", "unavailable", "rejected"})
_MATCH_IDENTITY_CODES = frozenset(
    {
        "dynamic_capability_snapshot_stale",
        "dynamic_target_identity_mismatch",
    }
)


@dataclass(frozen=True)
class _DynamicPostRoutingRematchDecision:
    source_requirement_set_id: str
    snapshot_id: str
    target_identity: TargetIdentity
    blocker_codes: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.blocker_codes


def _required_features(requirements: RequirementSet) -> dict[str, object]:
    extension = requirements.to_dict()["extensions"].get(DYNAMIC_REMOTE_EXTENSION)
    if not isinstance(extension, dict):
        raise ValueError("requirements do not contain the dynamic remote extension")
    required = extension.get("required")
    if not isinstance(required, dict):
        raise ValueError("dynamic remote extension does not contain requirements")
    return {
        name: value
        for name, value in required.items()
        if _active_requirement(name, value)
    }


def _preserved_features(
    source_requirements: RequirementSet,
    routed_requirements: RequirementSet,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source = _required_features(source_requirements)
    routed = _required_features(routed_requirements)
    source_names = tuple(sorted(source))
    preserved = tuple(
        name
        for name in source_names
        if name in routed and _value_satisfies(name, routed[name], source[name])
    )
    return source_names, preserved


def _blocker_code(blocker: CapabilityBlocker) -> str:
    if blocker.capability_name is None:
        return blocker.code
    return f"{blocker.code}:{blocker.capability_name}"


def _evaluate_post_routing_policy(
    *,
    source_requirement_set_id: str,
    presented_source_requirement_set_id: str | None,
    source_requirements: Sequence[str],
    preserved_source_requirements: Sequence[str],
    requested_target_id: str,
    postrouting_target_id: str,
    exact_target_identity_matches: bool | None = None,
    preflight_snapshot_id: str,
    postrouting_snapshot_id: str,
    snapshot_fresh: bool,
    source_match_blockers: Sequence[str] = (),
    route_requirements: Sequence[str] = (),
    route_facts: Mapping[str, str] | None = None,
    provider_legality: str = "unavailable",
) -> tuple[str, ...]:
    """Apply the frozen API Change 089 decision order without side effects."""

    if provider_legality not in _PROVIDER_LEGALITY_STATES:
        raise ValueError("provider_legality must be accepted, unavailable, or rejected")
    facts = dict(route_facts or {})
    invalid_states = sorted(set(facts.values()) - _ROUTE_FACT_STATES)
    if invalid_states:
        raise ValueError(
            "invalid route capability fact state(s): " + ", ".join(invalid_states)
        )

    blockers: list[str] = []
    if not presented_source_requirement_set_id:
        blockers.append("dynamic_source_requirement_identity_missing")
    elif presented_source_requirement_set_id != source_requirement_set_id:
        blockers.append("dynamic_source_requirement_identity_changed")

    preserved = set(preserved_source_requirements)
    blockers.extend(
        f"dynamic_source_semantic_lost:{name}"
        for name in sorted(set(source_requirements) - preserved)
    )

    target_matches = (
        postrouting_target_id == requested_target_id
        if exact_target_identity_matches is None
        else exact_target_identity_matches
    )
    if not target_matches:
        blockers.append("dynamic_postrouting_target_identity_mismatch")
    if postrouting_snapshot_id != preflight_snapshot_id:
        blockers.append("dynamic_postrouting_snapshot_identity_mismatch")
    if not snapshot_fresh:
        blockers.append("dynamic_postrouting_snapshot_stale")

    blockers.extend(source_match_blockers)
    for requirement in sorted(set(route_requirements)):
        state = facts.get(requirement, "missing")
        if state != "supported":
            blockers.append(f"dynamic_route_capability_fact_{state}:{requirement}")

    if provider_legality != "accepted":
        blockers.append(f"dynamic_provider_legality_{provider_legality}")
    return tuple(dict.fromkeys(blockers))


def evaluate_dynamic_post_routing_rematch(
    source_requirements: RequirementSet,
    routed_requirements: RequirementSet,
    *,
    presented_source_requirement_set_id: str | None,
    requested_target_identity: TargetIdentity,
    preflight_snapshot_id: str,
    postrouting_snapshot: TargetCapabilitySnapshot,
    evaluated_at: datetime,
    route_requirements: Sequence[str] = (),
    route_facts: Mapping[str, str] | None = None,
    provider_legality: str = "unavailable",
) -> _DynamicPostRoutingRematchDecision:
    """Rematch a routed dynamic program against the exact preflight snapshot."""

    if not isinstance(source_requirements, RequirementSet) or not isinstance(
        routed_requirements, RequirementSet
    ):
        raise TypeError("source and routed requirements must be RequirementSet values")
    if not isinstance(requested_target_identity, TargetIdentity):
        raise TypeError("requested_target_identity must be a TargetIdentity")
    if not isinstance(postrouting_snapshot, TargetCapabilitySnapshot):
        raise TypeError("postrouting_snapshot must be a TargetCapabilitySnapshot")
    if not isinstance(evaluated_at, datetime) or evaluated_at.tzinfo is None:
        raise ValueError("evaluated_at must be a timezone-aware datetime")
    if not isinstance(preflight_snapshot_id, str) or not preflight_snapshot_id:
        raise ValueError("preflight_snapshot_id must be a non-empty string")

    source_names, preserved = _preserved_features(
        source_requirements, routed_requirements
    )
    source_match = match_dynamic_remote_capabilities(
        source_requirements,
        postrouting_snapshot,
        evaluated_at=evaluated_at,
        expected_target_identity=requested_target_identity,
    )
    source_match_blockers = tuple(
        _blocker_code(blocker)
        for blocker in source_match.blockers
        if blocker.code not in _MATCH_IDENTITY_CODES
    )
    snapshot_fresh = all(
        blocker.code != "dynamic_capability_snapshot_stale"
        for blocker in source_match.blockers
    )
    blockers = _evaluate_post_routing_policy(
        source_requirement_set_id=source_requirements.requirement_set_id,
        presented_source_requirement_set_id=presented_source_requirement_set_id,
        source_requirements=source_names,
        preserved_source_requirements=preserved,
        requested_target_id=requested_target_identity.target_id,
        postrouting_target_id=postrouting_snapshot.target_identity.target_id,
        exact_target_identity_matches=(
            postrouting_snapshot.target_identity == requested_target_identity
        ),
        preflight_snapshot_id=preflight_snapshot_id,
        postrouting_snapshot_id=postrouting_snapshot.snapshot_id,
        snapshot_fresh=snapshot_fresh,
        source_match_blockers=source_match_blockers,
        route_requirements=route_requirements,
        route_facts=route_facts,
        provider_legality=provider_legality,
    )
    return _DynamicPostRoutingRematchDecision(
        source_requirement_set_id=source_requirements.requirement_set_id,
        snapshot_id=postrouting_snapshot.snapshot_id,
        target_identity=postrouting_snapshot.target_identity,
        blocker_codes=blockers,
    )


__all__: tuple[str, ...] = ()
