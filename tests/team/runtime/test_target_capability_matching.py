"""Runtime candidate matching seam tests against the Core pure matcher."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import flagquantum.core.target_capabilities as core_capabilities
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityRequirement,
    CapabilityScope,
    ComparisonOperator,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    FallbackAuthorizations,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.runtime import target_capability_matching as seam
from flagquantum.runtime.target_capability_matching import (
    FallbackAxis,
    RuntimeDecisionBlockerCode,
    TargetCapabilityCandidate,
    match_target_capability_candidates,
)

pytestmark = pytest.mark.unit

_CAPTURED_AT = datetime(2026, 9, 4, 8, 0, tzinfo=timezone.utc)
_EVALUATED_AT = _CAPTURED_AT + timedelta(minutes=1)


def _require(
    name: str,
    value: object,
    *,
    operator: ComparisonOperator = ComparisonOperator.EQUALS,
    strength: RequirementStrength = RequirementStrength.MANDATORY,
    accepted_exposures: tuple[FactExposure, ...] = (FactExposure.OBSERVED,),
) -> CapabilityRequirement:
    return CapabilityRequirement(
        name=name,
        operator=operator,
        value=value,
        strength=strength,
        source=RequirementSource.RUNTIME_PROTOCOL,
        minimum_evidence_level=EvidenceLevel.OBSERVABLE,
        accepted_exposures=accepted_exposures,
    )


def _requirements(
    *items: CapabilityRequirement,
    authorizations: FallbackAuthorizations = FallbackAuthorizations(),
) -> RequirementSet:
    return RequirementSet(
        requirements=items,
        fallback_authorizations=authorizations,
    )


def _snapshot(
    candidate_id: str,
    *,
    kind: str = "cuda",
    memory: int = 8 * 1024**3,
    device_count: int = 1,
    captured_at: datetime = _CAPTURED_AT,
    valid_for: timedelta = timedelta(hours=1),
    include_device_kind: bool = False,
    memory_status: SupportStatus = SupportStatus.VERIFIED,
    memory_exposure: FactExposure = FactExposure.OBSERVED,
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVABLE,
) -> TargetCapabilitySnapshot:
    device_id = f"{kind}:0"
    evidence_id = f"{candidate_id}-probe"
    scope = CapabilityScope(device_ids=(device_id,))
    evidence = EvidenceReference(
        evidence_id=evidence_id,
        sha256=(candidate_id.encode("utf-8").hex() + "0" * 64)[:64],
        level=evidence_level,
        scope=scope,
    )
    source = FactSource(kind="test_probe", ref=evidence_id)
    blockers = ()
    if memory_status is not SupportStatus.VERIFIED:
        from flagquantum.core.target_capabilities import CapabilityBlocker

        blockers = (
            CapabilityBlocker(
                code=f"{memory_status.value}_memory",
                message=f"memory is {memory_status.value}",
                capability_name="memory.available_bytes",
            ),
        )
    facts: list[CapabilityFact] = [
        CapabilityFact(
            name="memory.available_bytes",
            value=memory if memory_status is SupportStatus.VERIFIED else None,
            support_status=memory_status,
            fact_exposure=memory_exposure,
            source=source,
            blockers=blockers,
        ),
        CapabilityFact(
            name="device.count",
            value=device_count,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.OBSERVED,
            source=source,
        ),
    ]
    if include_device_kind:
        facts.append(
            CapabilityFact(
                name="device.kind",
                value=kind,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.OBSERVED,
                source=source,
            )
        )
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=f"target-{candidate_id}",
            target_class="local_runtime",
            provider="runtime-test",
            provider_version="1",
            target_revision="r1",
            environment_id="env-test",
        ),
        scope=scope,
        captured_at=captured_at.isoformat().replace("+00:00", "Z"),
        valid_until=(captured_at + valid_for).isoformat().replace("+00:00", "Z"),
        facts=tuple(facts),
        evidence_refs=(evidence,),
    )


def _candidate(
    candidate_id: str,
    *,
    kind: str = "cuda",
    memory: int = 8 * 1024**3,
    preference_rank: int = 0,
    fallback_axes: frozenset[FallbackAxis] = frozenset(),
    is_cpu_candidate: bool = False,
    **snapshot_kwargs: object,
) -> TargetCapabilityCandidate:
    return TargetCapabilityCandidate(
        candidate_id=candidate_id,
        snapshot=_snapshot(
            candidate_id,
            kind=kind,
            memory=memory,
            **snapshot_kwargs,
        ),
        preference_rank=preference_rank,
        fallback_axes=fallback_axes,
        is_cpu_candidate=is_cpu_candidate,
    )


def test_every_candidate_uses_the_same_immutable_requirement_set_and_core_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirements = _requirements(
        _require(
            "memory.available_bytes",
            1024,
            operator=ComparisonOperator.AT_LEAST,
        )
    )
    candidates = (_candidate("b"), _candidate("a"))
    calls: list[tuple[object, str]] = []
    original = core_capabilities.match_target_capabilities

    def spy(request: object, snapshot: TargetCapabilitySnapshot, **kwargs: object):
        calls.append((request, snapshot.snapshot_id))
        return original(request, snapshot, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(seam._core, "match_target_capabilities", spy)
    decision = match_target_capability_candidates(
        requirements, candidates, evaluated_at=_EVALUATED_AT
    )

    assert len(calls) == 2
    assert all(request is requirements for request, _ in calls)
    assert {snapshot_id for _, snapshot_id in calls} == {
        item.snapshot.snapshot_id for item in candidates
    }
    assert decision.executable is True


def test_cpu_is_an_explicit_candidate_and_is_fully_rematched() -> None:
    requirements = _requirements(
        _require(
            "memory.available_bytes",
            4 * 1024**3,
            operator=ComparisonOperator.AT_LEAST,
        ),
        _require("device.count", 1, operator=ComparisonOperator.AT_LEAST),
        authorizations=FallbackAuthorizations(cpu=True),
    )
    primary = _candidate("primary", memory=1 * 1024**3)
    cpu = _candidate(
        "cpu",
        kind="cpu",
        memory=8 * 1024**3,
        fallback_axes=frozenset({FallbackAxis.CPU}),
        is_cpu_candidate=True,
    )

    decision = match_target_capability_candidates(
        requirements, (primary, cpu), evaluated_at=_EVALUATED_AT
    )

    assert decision.executable is True
    assert decision.selected is not None
    assert decision.selected.candidate.candidate_id == "cpu"
    assert decision.fallback_record is not None
    assert (
        decision.fallback_record.requirement_set_id == requirements.requirement_set_id
    )
    assert decision.fallback_record.snapshot_id == cpu.snapshot.snapshot_id
    assert decision.fallback_record.target_identity == cpu.snapshot.target_identity
    assert decision.fallback_record.fallback_axes == (FallbackAxis.CPU,)
    assert decision.fallback_record.core_blockers == ()


def test_no_cpu_candidate_means_no_silent_cpu_fallback() -> None:
    requirements = _requirements(
        _require(
            "memory.available_bytes",
            4 * 1024**3,
            operator=ComparisonOperator.AT_LEAST,
        ),
        authorizations=FallbackAuthorizations(cpu=True),
    )
    decision = match_target_capability_candidates(
        requirements,
        (_candidate("primary", memory=1 * 1024**3),),
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.executable is False
    assert decision.selected is None
    assert decision.fallback_record is None
    assert all(not item.candidate.is_cpu_candidate for item in decision.evaluations)


def test_cpu_candidate_does_not_rewrite_a_mandatory_device_requirement() -> None:
    requirements = _requirements(
        _require("device.kind", "cuda"),
        authorizations=FallbackAuthorizations(cpu=True),
    )
    cpu = _candidate(
        "cpu",
        kind="cpu",
        fallback_axes=frozenset({FallbackAxis.CPU}),
        is_cpu_candidate=True,
        include_device_kind=True,
    )

    decision = match_target_capability_candidates(
        requirements, (cpu,), evaluated_at=_EVALUATED_AT
    )

    assert decision.selected is None
    assert decision.evaluations[0].core_match.executable is False
    assert any(
        blocker.code == "value_mismatch" and blocker.capability_name == "device.kind"
        for blocker in decision.evaluations[0].core_match.blockers
    )


@pytest.mark.parametrize("axis", tuple(FallbackAxis))
def test_fallback_axes_are_independent_and_authorized_per_axis(
    axis: FallbackAxis,
) -> None:
    authorizations = FallbackAuthorizations(**{axis.value: True})
    candidate = _candidate(
        f"fallback-{axis.value}",
        kind="cpu" if axis is FallbackAxis.CPU else "cuda",
        fallback_axes=frozenset({axis}),
        is_cpu_candidate=axis is FallbackAxis.CPU,
    )

    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
            authorizations=authorizations,
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.executable is True
    assert decision.fallback_record is not None
    assert decision.fallback_record.fallback_axes == (axis,)


def test_backend_authorization_does_not_imply_cpu_or_precision() -> None:
    candidate = _candidate(
        "cpu",
        kind="cpu",
        fallback_axes=frozenset({FallbackAxis.CPU}),
        is_cpu_candidate=True,
    )
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
            authorizations=FallbackAuthorizations(backend=True),
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.selected is None
    blocker = next(
        item
        for item in decision.blockers
        if item.code is RuntimeDecisionBlockerCode.FALLBACK_AXIS_UNAUTHORIZED
    )
    assert blocker.fallback_axes == (FallbackAxis.CPU,)


def test_multiple_changed_axes_require_all_authorizations() -> None:
    candidate = _candidate(
        "multi",
        fallback_axes=frozenset({FallbackAxis.BACKEND, FallbackAxis.PRECISION}),
    )
    requirements = _requirements(
        _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
        authorizations=FallbackAuthorizations(backend=True),
    )

    denied = match_target_capability_candidates(
        requirements, (candidate,), evaluated_at=_EVALUATED_AT
    )
    assert denied.selected is None
    unauthorized = next(
        item
        for item in denied.blockers
        if item.code is RuntimeDecisionBlockerCode.FALLBACK_AXIS_UNAUTHORIZED
    )
    assert unauthorized.fallback_axes == (FallbackAxis.PRECISION,)

    allowed = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
            authorizations=FallbackAuthorizations(backend=True, precision=True),
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )
    assert allowed.executable is True


def test_preference_cannot_rescue_an_ineligible_candidate() -> None:
    requirements = _requirements(
        _require(
            "memory.available_bytes", 4 * 1024**3, operator=ComparisonOperator.AT_LEAST
        )
    )
    preferred_but_ineligible = _candidate("preferred", memory=1, preference_rank=0)
    executable = _candidate("fallback", memory=8 * 1024**3, preference_rank=100)

    decision = match_target_capability_candidates(
        requirements,
        (executable, preferred_but_ineligible),
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.selected is not None
    assert decision.selected.candidate.candidate_id == "fallback"
    assert decision.evaluations[1].core_match.executable is False


def test_candidate_order_does_not_change_deterministic_preference_selection() -> None:
    requirements = _requirements(
        _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
    )
    candidates = (
        _candidate("z", preference_rank=3),
        _candidate("a", preference_rank=3),
        _candidate("m", preference_rank=1),
    )
    first = match_target_capability_candidates(
        requirements, candidates, evaluated_at=_EVALUATED_AT
    )
    second = match_target_capability_candidates(
        requirements, tuple(reversed(candidates)), evaluated_at=_EVALUATED_AT
    )

    assert first.selected is not None and second.selected is not None
    assert first.selected.candidate.candidate_id == "m"
    assert second.selected.candidate.candidate_id == "m"
    assert [item.candidate.candidate_id for item in first.evaluations] == [
        item.candidate.candidate_id for item in second.evaluations
    ]


def test_unauthorized_fallback_returns_typed_blocker_without_selection() -> None:
    requirements = _requirements(
        _require(
            "memory.available_bytes", 4 * 1024**3, operator=ComparisonOperator.AT_LEAST
        )
    )
    primary = _candidate("primary", memory=1)
    alternative = _candidate(
        "alternative",
        memory=8 * 1024**3,
        fallback_axes=frozenset({FallbackAxis.BACKEND}),
    )

    decision = match_target_capability_candidates(
        requirements, (primary, alternative), evaluated_at=_EVALUATED_AT
    )

    assert decision.selected is None
    assert decision.fallback_record is None
    blocker = next(
        item
        for item in decision.blockers
        if item.code is RuntimeDecisionBlockerCode.FALLBACK_AXIS_UNAUTHORIZED
    )
    assert blocker.candidate_id == "alternative"
    assert blocker.fallback_axes == (FallbackAxis.BACKEND,)


def test_core_stale_and_not_exposed_blockers_are_not_reinterpreted() -> None:
    stale = _candidate(
        "stale",
        captured_at=_CAPTURED_AT - timedelta(hours=2),
        valid_for=timedelta(minutes=1),
    )
    not_exposed = _candidate(
        "not-exposed",
        memory_status=SupportStatus.UNKNOWN,
        memory_exposure=FactExposure.NOT_EXPOSED,
    )
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (stale, not_exposed),
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.selected is None
    codes = {
        blocker.code
        for evaluation in decision.evaluations
        for blocker in evaluation.core_match.blockers
    }
    assert "snapshot_stale" in codes
    assert "fact_unknown" in codes


def test_core_unmeasured_and_weak_evidence_blockers_are_not_reinterpreted() -> None:
    unmeasured = _candidate(
        "unmeasured",
        memory_status=SupportStatus.UNMEASURED,
        memory_exposure=FactExposure.DECLARED,
    )
    weak_evidence = _candidate("weak", evidence_level=EvidenceLevel.BASIC)
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (unmeasured, weak_evidence),
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.selected is None
    codes = {
        blocker.code
        for evaluation in decision.evaluations
        for blocker in evaluation.core_match.blockers
    }
    assert "fact_unmeasured" in codes
    assert "insufficient_evidence" in codes


def test_default_runtime_plan_path_is_not_imported_or_changed() -> None:
    import flagquantum as fq

    plan = fq.plan(
        fq.Circuit(1).h(0),
        options=fq.ExecutionOptions(mode="statevector", device="cpu"),
    )

    assert plan.to_dict()["decision"]["device"] == "cpu"
    assert not hasattr(fq.runtime, "match_target_capability_candidates")
