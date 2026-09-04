from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from flagquantum.core.target_capabilities import (
    CapabilityBlocker,
    CapabilityContractError,
    CapabilityFact,
    CapabilityRequirement,
    CapabilityScope,
    ComparisonOperator,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
    match_target_capabilities,
)

pytestmark = pytest.mark.unit

_SCOPE = CapabilityScope(
    device_ids=("cpu:0",),
    dtype="complex128",
    kernel="statevector",
    workload_id="nullable-fact",
    world_size=1,
    node_count=1,
)
_SOURCE = FactSource("probe", "probe-1")
_BLOCKER = CapabilityBlocker("not_verified", "fact is not verified", "device.count")


def _fact(
    *,
    value: object,
    status: SupportStatus,
    exposure: FactExposure,
    blockers: tuple[CapabilityBlocker, ...] = (_BLOCKER,),
) -> CapabilityFact:
    return CapabilityFact(
        name="device.count",
        value=value,
        support_status=status,
        fact_exposure=exposure,
        source=_SOURCE,
        blockers=blockers,
    )


def _snapshot(fact: CapabilityFact) -> TargetCapabilitySnapshot:
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="cpu-local",
            target_class="local_runtime",
            provider="flagquantum",
            provider_version="test",
            target_revision="fixture",
            environment_id="test-environment",
        ),
        scope=_SCOPE,
        captured_at="2026-09-03T00:00:00Z",
        valid_until="2026-09-05T00:00:00Z",
        facts=(fact,),
        evidence_refs=(
            EvidenceReference(
                evidence_id="probe-1",
                sha256="a" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _requirements(*accepted: FactExposure) -> RequirementSet:
    return RequirementSet(
        requirements=(
            CapabilityRequirement(
                name="device.count",
                operator=ComparisonOperator.AT_LEAST,
                value=1,
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.USER,
                minimum_evidence_level=EvidenceLevel.BASIC,
                accepted_exposures=accepted,
            ),
        )
    )


@pytest.mark.parametrize(
    ("status", "exposure"),
    [
        (SupportStatus.UNKNOWN, FactExposure.OBSERVED),
        (SupportStatus.UNMEASURED, FactExposure.OBSERVED),
        (SupportStatus.UNSUPPORTED, FactExposure.OBSERVED),
        (SupportStatus.UNKNOWN, FactExposure.NOT_EXPOSED),
        (SupportStatus.UNKNOWN, FactExposure.UNKNOWN),
        (SupportStatus.UNKNOWN, FactExposure.NOT_APPLICABLE),
    ],
)
def test_non_verified_null_and_unexposed_matrix_is_constructible(
    status: SupportStatus, exposure: FactExposure
) -> None:
    fact = _fact(value=None, status=status, exposure=exposure)

    assert fact.value is None
    assert fact.support_status is status
    assert fact.fact_exposure is exposure
    assert fact.blockers == (_BLOCKER,)


def test_observed_unsupported_can_retain_a_negative_probe_value() -> None:
    fact = _fact(
        value=0,
        status=SupportStatus.UNSUPPORTED,
        exposure=FactExposure.OBSERVED,
    )

    assert fact.value == 0
    assert fact.blockers == (_BLOCKER,)


def test_verified_null_is_rejected_but_verified_value_remains_non_nullable() -> None:
    with pytest.raises(CapabilityContractError, match="must be non-null"):
        _fact(
            value=None,
            status=SupportStatus.VERIFIED,
            exposure=FactExposure.OBSERVED,
            blockers=(),
        )


@pytest.mark.parametrize(
    ("status", "exposure"),
    [
        (SupportStatus.UNKNOWN, FactExposure.OBSERVED),
        (SupportStatus.UNMEASURED, FactExposure.OBSERVED),
        (SupportStatus.UNSUPPORTED, FactExposure.OBSERVED),
        (SupportStatus.UNKNOWN, FactExposure.NOT_EXPOSED),
        (SupportStatus.UNKNOWN, FactExposure.UNKNOWN),
        (SupportStatus.UNKNOWN, FactExposure.NOT_APPLICABLE),
    ],
)
def test_every_nullable_fact_variant_fails_matcher_closed(
    status: SupportStatus, exposure: FactExposure
) -> None:
    accepted = (
        FactExposure.OBSERVED,
        FactExposure.NOT_EXPOSED,
        FactExposure.UNKNOWN,
        FactExposure.NOT_APPLICABLE,
    )
    result = match_target_capabilities(
        _requirements(*accepted),
        _snapshot(_fact(value=None, status=status, exposure=exposure)),
        evaluated_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
    )

    assert result.executable is False
    assert result.blockers


def test_nullable_fact_round_trip_and_snapshot_identity_are_deterministic() -> None:
    snapshot = _snapshot(
        _fact(
            value=None,
            status=SupportStatus.UNMEASURED,
            exposure=FactExposure.NOT_EXPOSED,
        )
    )
    payload = snapshot.to_json()
    restored = TargetCapabilitySnapshot.from_json(payload)

    assert restored == snapshot
    assert restored.snapshot_id == snapshot.snapshot_id
    assert restored.to_json() == payload
    assert json.loads(payload)["facts"][0]["value"] is None


@pytest.mark.parametrize(
    "exposure",
    [FactExposure.NOT_EXPOSED, FactExposure.UNKNOWN, FactExposure.NOT_APPLICABLE],
)
def test_unavailable_exposure_null_uses_non_verified_blocker_rule(
    exposure: FactExposure,
) -> None:
    with pytest.raises(CapabilityContractError, match="non-verified facts require"):
        _fact(
            value=None,
            status=SupportStatus.UNKNOWN,
            exposure=exposure,
            blockers=(),
        )
