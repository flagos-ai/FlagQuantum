"""Runtime candidate matching seam tests against the Core pure matcher."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pytest

import flagquantum.core.target_capabilities as core_capabilities
from flagquantum.core.target_capabilities import (
    CapabilityBlocker,
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
    CandidateProvenance,
    FallbackAxis,
    RouteIntent,
    RuntimeDecisionBlockerCode,
    TargetCapabilityCandidate,
)
from flagquantum.runtime.target_capability_matching import (
    match_target_capability_candidates as _raw_match_target_capability_candidates,
)

pytestmark = pytest.mark.unit

_CAPTURED_AT = datetime(2026, 9, 4, 8, 0, tzinfo=timezone.utc)
_EVALUATED_AT = _CAPTURED_AT + timedelta(minutes=1)
_ROUTE_INTENT = RouteIntent(
    backend="test-backend",
    device_kind="cuda",
    cpu=False,
    effective_precision="complex128",
    algorithm="exact",
    approximation="exact",
)
_CPU_ROUTE_INTENT = RouteIntent(
    backend="test-backend",
    device_kind="cpu",
    cpu=True,
    effective_precision="complex128",
    algorithm="exact",
    approximation="exact",
)
_UNSET = object()


def match_target_capability_candidates(requirements, candidates, **kwargs):
    """Keep legacy test calls explicit about the shared original intent."""

    if "route_intent" not in kwargs:
        explicit_cpu = any(
            item.name == "device.kind"
            and item.value == "cpu"
            and item.strength is RequirementStrength.MANDATORY
            for item in requirements.requirements
        )
        kwargs["route_intent"] = _CPU_ROUTE_INTENT if explicit_cpu else _ROUTE_INTENT
    return _raw_match_target_capability_candidates(requirements, candidates, **kwargs)


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
    include_device_kind: bool = True,
    effective_precision: str | None = "complex128",
    native_precision: str | None = None,
    storage_precision: str | None = None,
    software_mechanism: str | None = "none",
    precision_scope_dtype: str | None = None,
    effective_precision_evidence_level: EvidenceLevel | None = None,
    memory_status: SupportStatus = SupportStatus.VERIFIED,
    memory_exposure: FactExposure = FactExposure.OBSERVED,
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVABLE,
) -> TargetCapabilitySnapshot:
    device_id = f"{kind}:0"
    evidence_id = f"{candidate_id}-probe"
    scope = CapabilityScope(device_ids=(device_id,), dtype=precision_scope_dtype)
    evidence = EvidenceReference(
        evidence_id=evidence_id,
        sha256=(candidate_id.encode("utf-8").hex() + "0" * 64)[:64],
        level=evidence_level,
        scope=scope,
    )
    source = FactSource(kind="test_probe", ref=evidence_id)
    evidence_refs = [evidence]
    effective_source = source
    if effective_precision_evidence_level is not None:
        effective_evidence_id = f"{candidate_id}-precision"
        evidence_refs.append(
            EvidenceReference(
                evidence_id=effective_evidence_id,
                sha256=(effective_evidence_id.encode("utf-8").hex() + "0" * 64)[:64],
                level=effective_precision_evidence_level,
                scope=scope,
            )
        )
        effective_source = FactSource(kind="test_probe", ref=effective_evidence_id)
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
    if effective_precision is not None:
        facts.append(
            CapabilityFact(
                name="precision.effective_dtype",
                value=effective_precision,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.OBSERVED,
                source=effective_source,
            )
        )
        for name, value in (
            (
                "precision.native_dtype",
                native_precision
                or ("float64" if effective_precision == "complex128" else "float32"),
            ),
            (
                "precision.storage_dtype",
                storage_precision
                or ("float64" if effective_precision == "complex128" else "float32"),
            ),
            ("precision.software_mechanism", software_mechanism),
        ):
            if value is not None:
                facts.append(
                    CapabilityFact(
                        name=name,
                        value=value,
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
        evidence_refs=tuple(evidence_refs),
    )


def _candidate(
    candidate_id: str,
    *,
    kind: str = "cuda",
    memory: int = 8 * 1024**3,
    preference_rank: int = 0,
    provenance_axes: frozenset[FallbackAxis] = frozenset(),
    provenance: CandidateProvenance | None = None,
    route_intent: RouteIntent = _ROUTE_INTENT,
    effective_precision: str | None | object = _UNSET,
    **snapshot_kwargs: object,
) -> TargetCapabilityCandidate:
    effective_kind = (
        "rocm" if FallbackAxis.DEVICE in provenance_axes and kind == "cuda" else kind
    )
    snapshot_kwargs.setdefault(
        "effective_precision",
        (
            ("complex64" if FallbackAxis.PRECISION in provenance_axes else "complex128")
            if effective_precision is _UNSET
            else effective_precision
        ),
    )
    snapshot = _snapshot(
        candidate_id,
        kind=effective_kind,
        memory=memory,
        **snapshot_kwargs,
    )
    if provenance is None:
        axes = set(provenance_axes)
        if effective_kind == "cpu" and FallbackAxis.CPU in axes:
            axes.add(FallbackAxis.DEVICE)
        provenance = CandidateProvenance(
            intent_id=route_intent.intent_id,
            snapshot_id=snapshot.snapshot_id,
            backend=(
                "fallback-backend"
                if FallbackAxis.BACKEND in axes
                else route_intent.backend
            ),
            device_kind=(
                effective_kind
                if FallbackAxis.DEVICE in axes or effective_kind == "cpu"
                else route_intent.device_kind
            ),
            cpu=(effective_kind == "cpu"),
            effective_precision=(
                "complex64"
                if FallbackAxis.PRECISION in axes
                else route_intent.effective_precision
            ),
            algorithm=(
                "approximate"
                if FallbackAxis.ALGORITHM in axes
                else route_intent.algorithm
            ),
            approximation=(
                "approximate"
                if FallbackAxis.APPROXIMATION in axes
                else route_intent.approximation
            ),
        )
    return TargetCapabilityCandidate(
        candidate_id=candidate_id,
        snapshot=snapshot,
        preference_rank=preference_rank,
        provenance=provenance,
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

    candidate_calls = [call for call in calls if call[0] is requirements]
    assert len(candidate_calls) == 2
    assert all(request is requirements for request, _ in candidate_calls)
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
        authorizations=FallbackAuthorizations(cpu=True, device=True),
    )
    primary = _candidate("primary", memory=1 * 1024**3)
    cpu = _candidate(
        "cpu",
        kind="cpu",
        memory=8 * 1024**3,
        provenance_axes=frozenset({FallbackAxis.CPU}),
        include_device_kind=True,
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
    assert decision.fallback_record.fallback_axes == (
        FallbackAxis.CPU,
        FallbackAxis.DEVICE,
    )
    assert decision.fallback_record.core_blockers == ()
    assert decision.fallback_record.decision_id == decision.decision_id
    assert (
        decision.fallback_record.candidate_snapshot_ids
        == decision.candidate_snapshot_ids
    )
    assert decision.fallback_record.evaluated_at == decision.evaluated_at
    assert (
        decision.fallback_record.fallback_authorization_id
        == decision.fallback_authorization_id
    )
    assert decision.fallback_record.sorting_key_version == seam.SORTING_KEY_VERSION


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
    assert all(item.candidate.provenance is not None for item in decision.evaluations)


def test_verified_cpu_snapshot_requires_declared_cpu_fallback_axis() -> None:
    cpu = _candidate("cpu", kind="cpu", include_device_kind=True)
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (cpu,),
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.FALLBACK_AXIS_UNAUTHORIZED
        and blocker.fallback_axes == (FallbackAxis.CPU, FallbackAxis.DEVICE)
        for blocker in decision.blockers
    )


def test_explicit_cpu_requirement_allows_normal_cpu_without_fallback_authorization() -> (
    None
):
    cpu = _candidate(
        "cpu",
        kind="cpu",
        include_device_kind=True,
        route_intent=_CPU_ROUTE_INTENT,
    )
    decision = match_target_capability_candidates(
        _requirements(
            _require("device.kind", "cpu"),
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
        ),
        (cpu,),
        evaluated_at=_EVALUATED_AT,
        route_intent=_CPU_ROUTE_INTENT,
    )

    assert decision.executable is True
    assert decision.selected is not None
    assert decision.selected.candidate.provenance is not None
    assert decision.fallback_record is None


def test_cpu_marker_cannot_replace_missing_or_unknown_device_kind_fact() -> None:
    missing = _candidate("missing-cpu", kind="cpu", include_device_kind=False)
    unknown_base = _candidate("unknown-cpu", kind="cpu", include_device_kind=False)
    unknown_snapshot = unknown_base.snapshot
    unknown_fact = CapabilityFact(
        name="device.kind",
        value=None,
        support_status=SupportStatus.UNKNOWN,
        fact_exposure=FactExposure.NOT_EXPOSED,
        source=FactSource(
            kind="test_probe", ref=unknown_snapshot.evidence_refs[0].evidence_id
        ),
        blockers=(
            CapabilityBlocker(
                code="device_kind_not_exposed",
                message="device kind probe did not expose a value",
                capability_name="device.kind",
            ),
        ),
    )
    unknown = TargetCapabilityCandidate(
        "unknown-cpu",
        TargetCapabilitySnapshot(
            target_identity=unknown_snapshot.target_identity,
            scope=unknown_snapshot.scope,
            captured_at=unknown_snapshot.captured_at,
            valid_until=unknown_snapshot.valid_until,
            facts=(*unknown_snapshot.facts, unknown_fact),
            evidence_refs=unknown_snapshot.evidence_refs,
        ),
    )

    for candidate in (missing, unknown):
        decision = match_target_capability_candidates(
            _requirements(
                _require(
                    "memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST
                )
            ),
            (candidate,),
            evaluated_at=_EVALUATED_AT,
        )

        assert decision.selected is None
        assert any(
            blocker.code is RuntimeDecisionBlockerCode.CPU_IDENTITY_UNVERIFIED
            for blocker in decision.blockers
        )


def test_cpu_provenance_cannot_override_non_cpu_device_kind_fact() -> None:
    base = _candidate("cuda", kind="cuda", include_device_kind=True)
    claimed_cpu = replace(
        base,
        provenance=replace(
            base.provenance,
            device_kind="cpu",
            cpu=True,
            provenance_id="",
        ),
    )
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
            authorizations=FallbackAuthorizations(cpu=True),
        ),
        (claimed_cpu,),
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.PROVENANCE_SNAPSHOT_MISMATCH
        for blocker in decision.blockers
    )


def test_device_kind_identity_reuses_core_evidence_validation() -> None:
    unresolved_base = _candidate("unresolved-kind", kind="cuda")
    unresolved_snapshot = unresolved_base.snapshot
    unresolved_kind = CapabilityFact(
        name="device.kind",
        value="cuda",
        support_status=SupportStatus.VERIFIED,
        fact_exposure=FactExposure.OBSERVED,
        source=FactSource(kind="test_probe", ref="missing-device-kind-evidence"),
    )
    unresolved = TargetCapabilityCandidate(
        "unresolved-kind",
        TargetCapabilitySnapshot(
            target_identity=unresolved_snapshot.target_identity,
            scope=unresolved_snapshot.scope,
            captured_at=unresolved_snapshot.captured_at,
            valid_until=unresolved_snapshot.valid_until,
            facts=tuple(
                unresolved_kind if fact.name == "device.kind" else fact
                for fact in unresolved_snapshot.facts
            ),
            evidence_refs=unresolved_snapshot.evidence_refs,
        ),
    )
    weak = _candidate(
        "weak-kind",
        kind="cuda",
        evidence_level=EvidenceLevel.BASIC,
    )

    for candidate, expected_code in (
        (unresolved, "unresolved_evidence_reference"),
        (weak, "insufficient_evidence"),
    ):
        decision = match_target_capability_candidates(
            _requirements(), (candidate,), evaluated_at=_EVALUATED_AT
        )

        assert decision.selected is None
        identity_blocker = next(
            blocker
            for blocker in decision.blockers
            if blocker.code is RuntimeDecisionBlockerCode.CPU_IDENTITY_UNVERIFIED
        )
        assert any(
            blocker.code == expected_code and blocker.capability_name == "device.kind"
            for blocker in identity_blocker.core_blockers
        )


def test_cpu_candidate_does_not_rewrite_a_mandatory_device_requirement() -> None:
    requirements = _requirements(
        _require("device.kind", "cuda"),
        authorizations=FallbackAuthorizations(cpu=True),
    )
    cpu = _candidate(
        "cpu",
        kind="cpu",
        provenance_axes=frozenset({FallbackAxis.CPU}),
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
    authorization_values = {axis.value: True}
    if axis is FallbackAxis.CPU:
        authorization_values[FallbackAxis.DEVICE.value] = True
    authorizations = FallbackAuthorizations(**authorization_values)
    candidate = _candidate(
        f"fallback-{axis.value}",
        kind="cpu" if axis is FallbackAxis.CPU else "cuda",
        provenance_axes=frozenset({axis}),
        include_device_kind=True,
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
    expected_axes = (
        (FallbackAxis.CPU, FallbackAxis.DEVICE) if axis is FallbackAxis.CPU else (axis,)
    )
    assert decision.fallback_record.fallback_axes == expected_axes


def test_backend_authorization_does_not_imply_cpu_or_precision() -> None:
    candidate = _candidate(
        "cpu",
        kind="cpu",
        provenance_axes=frozenset({FallbackAxis.CPU}),
        include_device_kind=True,
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
    assert blocker.fallback_axes == (FallbackAxis.CPU, FallbackAxis.DEVICE)


def test_multiple_changed_axes_require_all_authorizations() -> None:
    candidate = _candidate(
        "multi",
        provenance_axes=frozenset({FallbackAxis.BACKEND, FallbackAxis.PRECISION}),
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


def test_failed_preferred_candidate_cannot_select_unauthorized_cpu_fallback() -> None:
    requirements = _requirements(
        _require(
            "memory.available_bytes", 4 * 1024**3, operator=ComparisonOperator.AT_LEAST
        )
    )
    preferred = _candidate("preferred", memory=1, preference_rank=0)
    cpu = _candidate(
        "cpu",
        kind="cpu",
        memory=8 * 1024**3,
        include_device_kind=True,
        provenance_axes=frozenset({FallbackAxis.CPU}),
        preference_rank=1,
    )

    decision = match_target_capability_candidates(
        requirements, (preferred, cpu), evaluated_at=_EVALUATED_AT
    )

    assert decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.FALLBACK_AXIS_UNAUTHORIZED
        and blocker.fallback_axes == (FallbackAxis.CPU, FallbackAxis.DEVICE)
        for blocker in decision.blockers
    )
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.CANDIDATE_CORE_MISMATCH
        and blocker.candidate_id == "preferred"
        for blocker in decision.blockers
    )


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
    assert first.decision_id == second.decision_id
    assert first.candidate_snapshot_ids == tuple(
        item.candidate.snapshot.snapshot_id for item in first.evaluations
    )


def test_decision_identity_binds_time_and_fallback_authorization() -> None:
    requirements = _requirements(
        _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
    )
    candidate = _candidate("candidate")
    first = match_target_capability_candidates(
        requirements, (candidate,), evaluated_at=_EVALUATED_AT
    )
    same = match_target_capability_candidates(
        requirements, (candidate,), evaluated_at=_EVALUATED_AT
    )
    later = match_target_capability_candidates(
        requirements,
        (candidate,),
        evaluated_at=_EVALUATED_AT + timedelta(seconds=1),
    )
    authorized = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
            authorizations=FallbackAuthorizations(backend=True),
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )
    axis_requirements = _requirements(
        _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
        authorizations=FallbackAuthorizations(backend=True, precision=True),
    )
    backend_axis = match_target_capability_candidates(
        axis_requirements,
        (_candidate("candidate", provenance_axes=frozenset({FallbackAxis.BACKEND})),),
        evaluated_at=_EVALUATED_AT,
    )
    precision_axis = match_target_capability_candidates(
        axis_requirements,
        (_candidate("candidate", provenance_axes=frozenset({FallbackAxis.PRECISION})),),
        evaluated_at=_EVALUATED_AT,
    )

    assert first.decision_id == same.decision_id
    assert first.decision_id != later.decision_id
    assert first.decision_id != authorized.decision_id
    assert backend_axis.decision_id != precision_axis.decision_id
    assert len(first.decision_id) == 64
    assert first.evaluated_at == "2026-09-04T08:01:00.000000Z"
    assert len(first.fallback_authorization_id) == 64


def test_duplicate_target_identity_with_different_snapshot_semantics_fails_closed() -> (
    None
):
    first_snapshot = _snapshot("first")
    second_raw = _snapshot("second", memory=16 * 1024**3)
    second_snapshot = TargetCapabilitySnapshot(
        target_identity=first_snapshot.target_identity,
        scope=second_raw.scope,
        captured_at=second_raw.captured_at,
        valid_until=second_raw.valid_until,
        facts=second_raw.facts,
        evidence_refs=second_raw.evidence_refs,
    )
    candidates = (
        TargetCapabilityCandidate("first", first_snapshot),
        TargetCapabilityCandidate("second", second_snapshot),
    )

    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        candidates,
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.selected is None
    assert all(
        any(
            blocker.code is RuntimeDecisionBlockerCode.DUPLICATE_TARGET_IDENTITY
            for blocker in evaluation.blockers
        )
        for evaluation in decision.evaluations
    )


def test_duplicate_snapshot_identity_is_not_last_write_wins() -> None:
    snapshot = _snapshot("duplicate")
    candidates = (
        TargetCapabilityCandidate("first", snapshot),
        TargetCapabilityCandidate("second", snapshot),
    )

    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        candidates,
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.selected is None
    assert all(
        any(
            blocker.code is RuntimeDecisionBlockerCode.DUPLICATE_SNAPSHOT_IDENTITY
            for blocker in evaluation.blockers
        )
        for evaluation in decision.evaluations
    )


def test_duplicate_candidate_id_rejects_all_members_under_input_permutations() -> None:
    requirements = _requirements(
        _require(
            "memory.available_bytes", 4 * 1024**3, operator=ComparisonOperator.AT_LEAST
        )
    )
    executable = _candidate("shared", memory=8 * 1024**3)
    ineligible = _candidate("shared", memory=1 * 1024**3)

    for candidates in ((executable, ineligible), (ineligible, executable)):
        decision = match_target_capability_candidates(
            requirements, candidates, evaluated_at=_EVALUATED_AT
        )

        assert decision.selected is None
        assert len(decision.evaluations) == 2
        assert all(
            any(
                blocker.code is RuntimeDecisionBlockerCode.DUPLICATE_CANDIDATE_ID
                for blocker in evaluation.blockers
            )
            for evaluation in decision.evaluations
        )


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
        provenance_axes=frozenset({FallbackAxis.BACKEND}),
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


def test_fallback_axes_and_cpu_identity_are_not_candidate_self_reports() -> None:
    assert not hasattr(TargetCapabilityCandidate, "fallback_axes")
    assert not hasattr(TargetCapabilityCandidate, "is_cpu_candidate")
    assert not hasattr(seam.TargetCapabilityCandidateEvaluation, "fallback_axes")
    candidate = _candidate(
        "computed-backend", provenance_axes=frozenset({FallbackAxis.BACKEND})
    )
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
            authorizations=FallbackAuthorizations(backend=True),
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )
    assert decision.selected is not None
    assert decision.fallback_record is not None
    assert decision.fallback_record.fallback_axes == (FallbackAxis.BACKEND,)


def test_non_cpu_axis_pairs_require_each_independent_authorization() -> None:
    axes = tuple(axis for axis in FallbackAxis if axis is not FallbackAxis.CPU)
    for first_axis, second_axis in combinations(axes, 2):
        candidate = _candidate(
            f"pair-{first_axis.value}-{second_axis.value}",
            provenance_axes=frozenset({first_axis, second_axis}),
        )
        decision = match_target_capability_candidates(
            _requirements(
                _require(
                    "memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST
                ),
                authorizations=FallbackAuthorizations(**{first_axis.value: True}),
            ),
            (candidate,),
            evaluated_at=_EVALUATED_AT,
        )
        assert decision.selected is None
        unauthorized = next(
            blocker
            for blocker in decision.blockers
            if blocker.code is RuntimeDecisionBlockerCode.FALLBACK_AXIS_UNAUTHORIZED
        )
        assert unauthorized.fallback_axes == (second_axis,)


def test_missing_or_unbound_provenance_fails_closed() -> None:
    candidate = _candidate("missing-provenance")
    missing = replace(candidate, provenance=None)
    tampered = replace(
        candidate,
        provenance=replace(candidate.provenance, provenance_id="tampered"),
    )
    for item, expected in (
        (missing, RuntimeDecisionBlockerCode.CANDIDATE_PROVENANCE_REQUIRED),
        (tampered, RuntimeDecisionBlockerCode.PROVENANCE_IDENTITY_MISMATCH),
    ):
        decision = match_target_capability_candidates(
            _requirements(
                _require(
                    "memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST
                )
            ),
            (item,),
            evaluated_at=_EVALUATED_AT,
        )
        assert decision.selected is None
        assert any(blocker.code is expected for blocker in decision.blockers)


def test_missing_route_intent_fails_closed_even_with_candidate_provenance() -> None:
    decision = _raw_match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (_candidate("missing-intent"),),
        evaluated_at=_EVALUATED_AT,
    )
    assert decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.ROUTE_INTENT_REQUIRED
        for blocker in decision.blockers
    )


def test_snapshot_device_declaration_must_match_provenance() -> None:
    base = _candidate("device-conflict")
    candidate = replace(
        base,
        provenance=replace(
            base.provenance,
            device_kind="other-device-kind",
            provenance_id="",
        ),
    )
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )
    assert decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.PROVENANCE_SNAPSHOT_MISMATCH
        for blocker in decision.blockers
    )


def test_provenance_must_bind_the_exact_snapshot_identity() -> None:
    original = _candidate("original")
    replacement = _candidate("replacement")
    swapped = replace(original, snapshot=replacement.snapshot)
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (swapped,),
        evaluated_at=_EVALUATED_AT,
    )
    assert decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.PROVENANCE_SNAPSHOT_MISMATCH
        for blocker in decision.blockers
    )


@pytest.mark.parametrize(
    ("effective_precision", "evidence_level", "expected"),
    (
        (
            None,
            EvidenceLevel.OBSERVABLE,
            RuntimeDecisionBlockerCode.PRECISION_IDENTITY_UNVERIFIED,
        ),
        (
            "complex128",
            EvidenceLevel.BASIC,
            RuntimeDecisionBlockerCode.PRECISION_IDENTITY_UNVERIFIED,
        ),
    ),
)
def test_effective_precision_requires_observed_core_evidence(
    effective_precision: str | None,
    evidence_level: EvidenceLevel,
    expected: RuntimeDecisionBlockerCode,
) -> None:
    candidate = _candidate(
        "precision-unverified",
        effective_precision=effective_precision,
        evidence_level=evidence_level,
    )
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )
    assert decision.selected is None
    assert any(blocker.code is expected for blocker in decision.blockers)


def test_effective_precision_provenance_cannot_disagree_with_snapshot() -> None:
    base = _candidate("precision-mismatch")
    candidate = replace(
        base,
        provenance=replace(
            base.provenance,
            effective_precision="complex64",
            provenance_id="",
        ),
    )
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST),
            authorizations=FallbackAuthorizations(precision=True),
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )
    assert decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.PROVENANCE_SNAPSHOT_MISMATCH
        and blocker.fallback_axes == (FallbackAxis.PRECISION,)
        for blocker in decision.blockers
    )


def test_software_expanded_precision_requires_certified_dtype_scope() -> None:
    incomplete = _candidate(
        "software-precision-observed",
        native_precision="float32",
        storage_precision="float32",
        software_mechanism="double-single",
    )
    unscoped = _candidate(
        "software-precision-unscoped",
        native_precision="float32",
        storage_precision="float32",
        software_mechanism="double-single",
        effective_precision_evidence_level=EvidenceLevel.CERTIFICATION,
    )
    certified = _candidate(
        "software-precision-certified",
        native_precision="float32",
        storage_precision="float32",
        software_mechanism="double-single",
        precision_scope_dtype="complex128",
        effective_precision_evidence_level=EvidenceLevel.CERTIFICATION,
    )

    for candidate in (incomplete, unscoped):
        decision = match_target_capability_candidates(
            _requirements(
                _require(
                    "memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST
                )
            ),
            (candidate,),
            evaluated_at=_EVALUATED_AT,
        )
        assert decision.selected is None
        assert any(
            blocker.code is RuntimeDecisionBlockerCode.PRECISION_IDENTITY_UNVERIFIED
            and "certification-level" in blocker.message
            for blocker in decision.blockers
        )

    accepted = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (certified,),
        evaluated_at=_EVALUATED_AT,
    )
    assert accepted.executable is True
    assert accepted.selected is not None
    assert accepted.selected.candidate.candidate_id == "software-precision-certified"


def test_candidate_requires_a_complete_observed_precision_path() -> None:
    candidate = _candidate("precision-path-incomplete", software_mechanism=None)
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )

    assert decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.PRECISION_IDENTITY_UNVERIFIED
        and "precision.software_mechanism" in blocker.message
        for blocker in decision.blockers
    )


def test_route_values_reject_cpu_device_kind_contradictions() -> None:
    with pytest.raises(ValueError, match="cpu must agree with device_kind"):
        RouteIntent(
            backend="test-backend",
            device_kind="cpu",
            cpu=False,
            effective_precision="complex128",
            algorithm="exact",
            approximation="exact",
        )
    with pytest.raises(ValueError, match="cpu must agree with device_kind"):
        CandidateProvenance(
            intent_id=_ROUTE_INTENT.intent_id,
            snapshot_id="snapshot",
            backend=_ROUTE_INTENT.backend,
            device_kind="cpu",
            cpu=False,
            effective_precision=_ROUTE_INTENT.effective_precision,
            algorithm=_ROUTE_INTENT.algorithm,
            approximation=_ROUTE_INTENT.approximation,
        )


def test_route_values_reject_scalar_effective_precision() -> None:
    with pytest.raises(ValueError, match="complex64 or complex128"):
        RouteIntent(
            backend="test-backend",
            device_kind="cuda",
            cpu=False,
            effective_precision="float64",
            algorithm="exact",
            approximation="exact",
        )
    with pytest.raises(ValueError, match="complex64 or complex128"):
        CandidateProvenance(
            intent_id=_ROUTE_INTENT.intent_id,
            snapshot_id="snapshot",
            backend=_ROUTE_INTENT.backend,
            device_kind=_ROUTE_INTENT.device_kind,
            cpu=False,
            effective_precision="float64",
            algorithm=_ROUTE_INTENT.algorithm,
            approximation=_ROUTE_INTENT.approximation,
        )


def test_equivalent_provenance_has_no_fallback_record() -> None:
    candidate = _candidate("equivalent")
    decision = match_target_capability_candidates(
        _requirements(
            _require("memory.available_bytes", 1, operator=ComparisonOperator.AT_LEAST)
        ),
        (candidate,),
        evaluated_at=_EVALUATED_AT,
    )
    assert decision.executable is True
    assert decision.selected is not None
    assert decision.selected.computed_fallback_axes == frozenset()
    assert decision.fallback_record is None


def test_decision_identity_is_independent_of_python_hash_seed() -> None:
    root = Path(__file__).resolve().parents[3]
    script = (
        "from tests.team.runtime.test_target_capability_matching import "
        "_candidate, _requirements, _require, _EVALUATED_AT, "
        "ComparisonOperator, match_target_capability_candidates; "
        "print(match_target_capability_candidates(_requirements(_require('memory.available_bytes', 1, "
        "operator=ComparisonOperator.AT_LEAST)), (_candidate('seed'),), "
        "evaluated_at=_EVALUATED_AT).decision_id)"
    )
    values = []
    for seed in ("1", "987"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(root)
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        values.append(result.stdout.strip())
    assert values[0] == values[1]


def test_default_runtime_plan_path_is_not_imported_or_changed() -> None:
    import flagquantum as fq

    plan = fq.plan(
        fq.Circuit(1).h(0),
        options=fq.ExecutionOptions(mode="statevector", device="cpu"),
    )

    assert plan.to_dict()["decision"]["device"] == "cpu"
    assert not hasattr(fq.runtime, "match_target_capability_candidates")
