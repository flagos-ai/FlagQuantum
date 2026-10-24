"""Fail-closed admission for stable remote dynamic execution.

This module is internal. It derives requirements from the program and matches
them against one exact, device-scoped capability snapshot before any compiler,
credential, provider, package, network, or task boundary is entered.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from ...core.target_capabilities import (
    CapabilityMatchResult,
    CapabilityScope,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from ...errors import CapabilityError
from .circuit import DynamicCircuit
from .remote_capabilities import (
    DYNAMIC_REMOTE_EXTENSION,
    DYNAMIC_REMOTE_EXTENSION_VERSION,
    dynamic_remote_requirements,
    match_dynamic_remote_capabilities,
)


@dataclass(frozen=True)
class _DynamicRemotePreflightDecision:
    target: str
    target_identity: TargetIdentity
    requirement_set_id: str
    snapshot_id: str
    blocker_codes: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.blocker_codes


def _explicit_target_identity(target: str) -> tuple[str, TargetIdentity]:
    if not isinstance(target, str):
        raise CapabilityError(
            "dynamic_remote_target_required: remote DynamicCircuit execution "
            "requires target='provider:backend'"
        )
    candidate = target.strip()
    provider, separator, backend = candidate.partition(":")
    if (
        candidate != target
        or
        separator != ":"
        or not provider
        or not backend
        or provider != provider.strip()
        or backend != backend.strip()
    ):
        raise CapabilityError(
            "dynamic_remote_target_invalid: remote DynamicCircuit execution "
            "requires target='provider:backend'"
        )
    canonical = f"{provider.lower()}:{backend}"
    return candidate, TargetIdentity(
        target_id=canonical,
        target_class="remote_qpu",
        provider=provider.lower(),
        provider_version="unresolved",
        target_revision="unresolved",
        environment_id="unresolved",
    )


def _missing_snapshot(
    identity: TargetIdentity,
    *,
    evaluated_at: datetime,
) -> TargetCapabilitySnapshot:
    """Represent the absence of provider facts without inventing capacity."""

    captured_at = evaluated_at.astimezone(timezone.utc)
    valid_until = captured_at + timedelta(hours=1)
    scope = CapabilityScope(device_ids=(identity.target_id,))
    return TargetCapabilitySnapshot(
        target_identity=identity,
        scope=scope,
        captured_at=captured_at.isoformat().replace("+00:00", "Z"),
        valid_until=valid_until.isoformat().replace("+00:00", "Z"),
        facts=(),
        evidence_refs=(),
        extensions={
            DYNAMIC_REMOTE_EXTENSION: {
                "schema_version": DYNAMIC_REMOTE_EXTENSION_VERSION,
                "feature_set": None,
                "facts": {},
            }
        },
    )


def _blocker_codes(result: CapabilityMatchResult) -> tuple[str, ...]:
    return tuple(
        blocker.code
        if blocker.capability_name is None
        else f"{blocker.code}:{blocker.capability_name}"
        for blocker in result.blockers
    )


def evaluate_dynamic_remote_preflight(
    circuit: DynamicCircuit,
    *,
    target: str,
    snapshot: TargetCapabilitySnapshot | None = None,
    evaluated_at: datetime | None = None,
    require_mid_circuit_measurements: bool = False,
) -> _DynamicRemotePreflightDecision:
    """Return one pure admission decision for an explicit remote target."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("circuit must be a DynamicCircuit")
    now = evaluated_at or datetime.now(timezone.utc)
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("evaluated_at must be a timezone-aware datetime")
    now = now.astimezone(timezone.utc)
    explicit_target, requested_identity = _explicit_target_identity(target)
    requirements = dynamic_remote_requirements(
        circuit,
        require_mid_circuit_measurements=require_mid_circuit_measurements,
    )
    candidate = snapshot or _missing_snapshot(requested_identity, evaluated_at=now)
    expected_identity = (
        candidate.target_identity
        if candidate.target_identity.target_id == requested_identity.target_id
        and candidate.target_identity.provider == requested_identity.provider
        else replace(
            candidate.target_identity,
            target_id=requested_identity.target_id,
            provider=requested_identity.provider,
        )
    )
    result = match_dynamic_remote_capabilities(
        requirements,
        candidate,
        evaluated_at=now,
        expected_target_identity=expected_identity,
    )
    return _DynamicRemotePreflightDecision(
        target=explicit_target,
        target_identity=expected_identity,
        requirement_set_id=requirements.requirement_set_id,
        snapshot_id=candidate.snapshot_id,
        blocker_codes=_blocker_codes(result),
    )


def require_dynamic_remote_preflight(
    circuit: DynamicCircuit,
    *,
    target: str | None,
) -> _DynamicRemotePreflightDecision:
    """Refuse remote dynamic execution until detailed target facts are supplied."""

    if target is None:
        raise CapabilityError(
            "dynamic_remote_target_required: remote DynamicCircuit execution "
            "requires target='provider:backend'"
        )
    decision = evaluate_dynamic_remote_preflight(circuit, target=target)
    if not decision.compatible:
        raise CapabilityError(
            "dynamic_remote_preflight_failed: "
            f"target={decision.target}; blockers="
            + ", ".join(decision.blocker_codes)
        )
    return decision


__all__: tuple[str, ...] = ()
