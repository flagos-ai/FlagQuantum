"""Replacement conformance for CPU and the synthetic remote-style producer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from math import nan

import pytest

from flagquantum.core.target_capabilities import (
    CAPABILITY_NAMES,
    CapabilityRequirement,
    CapabilityScope,
    ComparisonOperator,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FallbackAuthorizations,
    MatchBlockerCode,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    SupportStatus,
)
from flagquantum.deployment.synthetic_remote_target_capabilities import (
    SyntheticRemoteCapabilityFixture,
    synthetic_remote_target_capability_snapshot,
)
from flagquantum.runtime.platforms.cpu_target_capabilities import (
    CPUCapabilityObservation,
    CPUPrecisionObservation,
    cpu_platform_to_target_capability_snapshot,
)
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
_MEMORY_REQUIREMENT = 4 * 1024**3
_ROUTE_INTENT = RouteIntent(
    backend="runtime-test",
    device_kind="synthetic_qpu",
    cpu=False,
    effective_precision="float64",
    algorithm="exact",
    approximation="exact",
)


def match_target_capability_candidates(requirements, candidates, **kwargs):
    kwargs.setdefault("route_intent", _ROUTE_INTENT)
    return _raw_match_target_capability_candidates(requirements, candidates, **kwargs)


@dataclass
class _CPUProbe:
    observation: CPUCapabilityObservation
    target_id: str = "cpu-target"
    provider_version: str = "torch-test"
    target_revision: str = "cpu-revision"
    environment_id: str = "cpu-environment"
    source_ref: str = "cpu-observation"
    target_class_source_ref: str = "cpu-declaration"

    def observe(self) -> CPUCapabilityObservation:
        return self.observation


def _cpu_snapshot(*, memory: int = 8 * 1024**3):
    scope = CapabilityScope(device_ids=("cpu:0",))
    evidence = (
        EvidenceReference(
            evidence_id="cpu-observation",
            sha256="a" * 64,
            level=EvidenceLevel.OBSERVABLE,
            scope=scope,
        ),
        EvidenceReference(
            evidence_id="cpu-declaration",
            sha256="b" * 64,
            level=EvidenceLevel.BASIC,
            scope=scope,
        ),
    )
    probe = _CPUProbe(
        CPUCapabilityObservation(
            available=True,
            device_count=1,
            device_ids=("cpu:0",),
            memory_available_bytes=memory,
            precision=(
                CPUPrecisionObservation(
                    "precision.effective_dtype",
                    "float64",
                ),
            ),
        )
    )
    return cpu_platform_to_target_capability_snapshot(
        probe=probe,
        captured_at=_CAPTURED_AT,
        ttl=timedelta(hours=1),
        evidence_refs=evidence,
    )


def _remote_fixture(
    *,
    target_id: str = "synthetic-remote-target",
    source_prefix: str = "remote",
    memory: int | None = 8 * 1024**3,
    device_kind: str = "synthetic_qpu",
    extra_observed_facts: dict[str, object] | None = None,
    declared_facts: dict[str, object] | None = None,
):
    scope = CapabilityScope(device_ids=(f"{target_id}:0",))
    evidence = (
        EvidenceReference(
            evidence_id=f"{source_prefix}-observation",
            sha256=("c" + source_prefix.encode().hex() + "0" * 64)[:64],
            level=EvidenceLevel.OBSERVABLE,
            scope=scope,
        ),
        EvidenceReference(
            evidence_id=f"{source_prefix}-declaration",
            sha256=("d" + source_prefix.encode().hex() + "0" * 64)[:64],
            level=EvidenceLevel.BASIC,
            scope=scope,
        ),
    )
    observed_facts = {
        "device.count": 1,
        "precision.effective_dtype": "float64",
    }
    if memory is not None:
        observed_facts["memory.available_bytes"] = memory
    if extra_observed_facts is not None:
        observed_facts.update(extra_observed_facts)
    fixture = SyntheticRemoteCapabilityFixture(
        target_id=target_id,
        target_revision="fixture-revision",
        environment_id="fixture-environment",
        device_kind=device_kind,
        device_ids=scope.device_ids or (),
        evidence_refs=evidence,
        observed_facts=observed_facts,
        declared_facts=declared_facts or {},
        observed_source_ref=f"{source_prefix}-observation",
        declared_source_ref=f"{source_prefix}-declaration",
    )
    return fixture


def _remote_snapshot(
    *,
    target_id: str = "synthetic-remote-target",
    source_prefix: str = "remote",
    memory: int | None = 8 * 1024**3,
    valid_for: timedelta = timedelta(hours=1),
    device_kind: str = "synthetic_qpu",
    observed_facts: dict[str, object] | None = None,
    declared_facts: dict[str, object] | None = None,
):
    fixture = _remote_fixture(
        target_id=target_id,
        source_prefix=source_prefix,
        memory=memory,
        device_kind=device_kind,
        extra_observed_facts=observed_facts,
        declared_facts=declared_facts,
    )
    return synthetic_remote_target_capability_snapshot(
        fixture=fixture,
        captured_at=_CAPTURED_AT,
        ttl=valid_for,
    )


def _requirements(
    *,
    authorizations: FallbackAuthorizations = FallbackAuthorizations(),
) -> RequirementSet:
    return RequirementSet(
        requirements=(
            CapabilityRequirement(
                name="memory.available_bytes",
                operator=ComparisonOperator.AT_LEAST,
                value=_MEMORY_REQUIREMENT,
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.RUNTIME_PROTOCOL,
                minimum_evidence_level=EvidenceLevel.OBSERVABLE,
                accepted_exposures=(FactExposure.OBSERVED,),
            ),
        ),
        fallback_authorizations=authorizations,
    )


def _candidate(
    candidate_id: str,
    snapshot,
    *,
    preference_rank: int = 0,
    provenance_axes: frozenset[FallbackAxis] = frozenset(),
) -> TargetCapabilityCandidate:
    kind = next(fact.value for fact in snapshot.facts if fact.name == "device.kind")
    declared_axes = set(provenance_axes)
    if kind == "cpu" and FallbackAxis.CPU in declared_axes:
        declared_axes.add(FallbackAxis.DEVICE)
    provenance = CandidateProvenance(
        intent_id=_ROUTE_INTENT.intent_id,
        snapshot_id=snapshot.snapshot_id,
        backend=(
            "fallback-backend"
            if FallbackAxis.BACKEND in declared_axes
            else _ROUTE_INTENT.backend
        ),
        device_kind=(
            str(kind)
            if kind == "cpu" or FallbackAxis.DEVICE in declared_axes
            else _ROUTE_INTENT.device_kind
        ),
        cpu=(kind == "cpu"),
        effective_precision=(
            "float32"
            if FallbackAxis.PRECISION in declared_axes
            else _ROUTE_INTENT.effective_precision
        ),
        algorithm=(
            "approximate"
            if FallbackAxis.ALGORITHM in declared_axes
            else _ROUTE_INTENT.algorithm
        ),
        approximation=(
            "approximate"
            if FallbackAxis.APPROXIMATION in declared_axes
            else _ROUTE_INTENT.approximation
        ),
    )
    return TargetCapabilityCandidate(
        candidate_id=candidate_id,
        snapshot=snapshot,
        preference_rank=preference_rank,
        provenance=provenance,
    )


def test_synthetic_producer_emits_complete_core_snapshot_without_inference() -> None:
    snapshot = _remote_snapshot()
    facts = {fact.name: fact for fact in snapshot.facts}

    assert set(facts) == set(CAPABILITY_NAMES)
    assert snapshot.target_identity.target_id == "synthetic-remote-target"
    assert snapshot.target_identity.provider == "flagquantum.synthetic_remote"
    assert snapshot.scope == CapabilityScope(device_ids=("synthetic-remote-target:0",))
    assert snapshot.valid_until == "2026-09-04T09:00:00Z"
    assert facts["device.kind"].value == "synthetic_qpu"
    assert facts["device.kind"].support_status is SupportStatus.VERIFIED
    assert facts["device.kind"].fact_exposure is FactExposure.OBSERVED
    assert facts["memory.available_bytes"].value == 8 * 1024**3
    assert facts["target.class"].value == "synthetic_remote_service"
    assert facts["target.class"].fact_exposure is FactExposure.DECLARED
    omitted = facts["precision.native_dtype"]
    assert omitted.value is None
    assert omitted.support_status is SupportStatus.UNKNOWN
    assert omitted.fact_exposure is FactExposure.NOT_EXPOSED
    assert omitted.blockers[0].code == "synthetic_precision_native_dtype_not_exposed"
    assert {item.evidence_id for item in snapshot.evidence_refs} == {
        "remote-observation",
        "remote-declaration",
    }


def test_same_runtime_consumer_replaces_cpu_with_synthetic_remote_snapshot() -> None:
    requirements = _requirements(
        authorizations=FallbackAuthorizations(cpu=True, device=True)
    )
    cpu = _candidate(
        "cpu",
        _cpu_snapshot(),
        provenance_axes=frozenset({FallbackAxis.CPU}),
    )
    remote = _candidate("remote", _remote_snapshot())

    cpu_decision = match_target_capability_candidates(
        requirements, (cpu,), evaluated_at=_EVALUATED_AT
    )
    remote_decision = match_target_capability_candidates(
        requirements, (remote,), evaluated_at=_EVALUATED_AT
    )

    assert cpu_decision.executable is True
    assert cpu_decision.selected is not None
    assert cpu_decision.selected.candidate.snapshot.target_identity.provider == (
        "pytorch_cpu"
    )
    assert remote_decision.executable is True
    assert remote_decision.selected is not None
    assert remote_decision.selected.candidate.snapshot.target_identity.provider == (
        "flagquantum.synthetic_remote"
    )
    assert cpu_decision.requirement_set_id == remote_decision.requirement_set_id
    assert cpu_decision.selected.candidate.snapshot.snapshot_id != (
        remote_decision.selected.candidate.snapshot.snapshot_id
    )


def test_replacement_order_is_deterministic_and_does_not_merge_target_identity() -> (
    None
):
    requirements = _requirements()
    first = _candidate(
        "first", _remote_snapshot(target_id="remote-a", source_prefix="a")
    )
    second = _candidate(
        "second",
        _remote_snapshot(target_id="remote-b", source_prefix="b"),
        preference_rank=1,
    )
    forward = match_target_capability_candidates(
        requirements, (second, first), evaluated_at=_EVALUATED_AT
    )
    reverse = match_target_capability_candidates(
        requirements, (first, second), evaluated_at=_EVALUATED_AT
    )

    assert forward.selected is not None
    assert reverse.selected is not None
    assert forward.selected.candidate.candidate_id == "first"
    assert reverse.selected.candidate.candidate_id == "first"
    assert [item.candidate.candidate_id for item in forward.evaluations] == [
        "first",
        "second",
    ]
    assert forward.decision_id == reverse.decision_id

    conflicting = _remote_snapshot(target_id="same-target", source_prefix="same-a")
    conflicting_other = _remote_snapshot(
        target_id="same-target", source_prefix="same-b", memory=16 * 1024**3
    )
    identity_decision = match_target_capability_candidates(
        requirements,
        (_candidate("one", conflicting), _candidate("two", conflicting_other)),
        evaluated_at=_EVALUATED_AT,
    )
    assert identity_decision.selected is None
    assert all(
        any(
            blocker.code is RuntimeDecisionBlockerCode.DUPLICATE_TARGET_IDENTITY
            for blocker in evaluation.blockers
        )
        for evaluation in identity_decision.evaluations
    )


def test_replacement_fails_closed_for_stale_scope_and_evidence_reference() -> None:
    requirements = _requirements()
    stale = _candidate(
        "stale",
        _remote_snapshot(valid_for=timedelta(seconds=10)),
    )
    stale_decision = match_target_capability_candidates(
        requirements, (stale,), evaluated_at=_CAPTURED_AT + timedelta(seconds=11)
    )
    assert stale_decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.CANDIDATE_CORE_MISMATCH
        and any(
            item.code == MatchBlockerCode.SNAPSHOT_STALE.value
            for item in blocker.core_blockers
        )
        for blocker in stale_decision.blockers
    )

    scoped = _candidate("scoped", _remote_snapshot())
    scope_decision = match_target_capability_candidates(
        requirements,
        (scoped,),
        evaluated_at=_EVALUATED_AT,
        required_scope=CapabilityScope(device_ids=("other:0",)),
    )
    assert scope_decision.selected is None
    assert any(
        any(
            item.code == MatchBlockerCode.SCOPE_MISMATCH.value
            for item in blocker.core_blockers
        )
        for blocker in scope_decision.blockers
        if blocker.code is RuntimeDecisionBlockerCode.CANDIDATE_CORE_MISMATCH
    )

    valid = _remote_snapshot()
    broken_evidence = replace(
        valid,
        snapshot_id="",
        evidence_refs=(valid.evidence_refs[0],),
    )
    evidence_decision = match_target_capability_candidates(
        requirements,
        (_candidate("broken-evidence", broken_evidence),),
        evaluated_at=_EVALUATED_AT,
    )
    assert evidence_decision.selected is None
    assert any(
        any(
            item.code == MatchBlockerCode.UNRESOLVED_EVIDENCE_REFERENCE.value
            for item in blocker.core_blockers
        )
        for blocker in evidence_decision.blockers
        if blocker.code is RuntimeDecisionBlockerCode.CANDIDATE_CORE_MISMATCH
    )


def test_missing_fact_is_nullable_unknown_and_not_exposed() -> None:
    snapshot = _remote_snapshot(memory=None)
    requirement = _requirements()
    fact = next(
        item for item in snapshot.facts if item.name == "memory.available_bytes"
    )

    assert fact.value is None
    assert fact.support_status is SupportStatus.UNKNOWN
    assert fact.fact_exposure is FactExposure.NOT_EXPOSED
    decision = match_target_capability_candidates(
        requirement,
        (_candidate("missing-memory", snapshot),),
        evaluated_at=_EVALUATED_AT,
    )
    assert decision.selected is None
    assert any(
        any(
            item.code == MatchBlockerCode.FACT_UNKNOWN.value
            for item in blocker.core_blockers
        )
        for blocker in decision.blockers
        if blocker.code is RuntimeDecisionBlockerCode.CANDIDATE_CORE_MISMATCH
    )


def test_cpu_fallback_axis_is_not_inferred_or_authorized_by_other_axes() -> None:
    cpu = _candidate(
        "cpu",
        _cpu_snapshot(),
        provenance_axes=frozenset({FallbackAxis.CPU}),
    )
    requirements = _requirements(authorizations=FallbackAuthorizations(backend=True))
    decision = match_target_capability_candidates(
        requirements, (cpu,), evaluated_at=_EVALUATED_AT
    )

    assert decision.selected is None
    assert any(
        blocker.code is RuntimeDecisionBlockerCode.FALLBACK_AXIS_UNAUTHORIZED
        and blocker.fallback_axes == (FallbackAxis.CPU, FallbackAxis.DEVICE)
        for blocker in decision.blockers
    )


def test_synthetic_fixture_deep_freezes_inputs_and_keeps_identity_stable() -> None:
    raw_observed = {
        "device.count": 1,
        "gates.native": [{"name": "h", "parameters": ["theta"]}],
    }
    raw_declared = {"artifacts.profiles": ["openqasm3"]}
    fixture = _remote_fixture(
        extra_observed_facts=raw_observed,
        declared_facts=raw_declared,
    )
    raw_observed["device.count"] = 2
    raw_observed["gates.native"][0]["parameters"].append("phi")
    raw_declared["artifacts.profiles"].append("qcis")

    assert fixture.observed_facts["device.count"] == 1
    assert fixture.observed_facts["gates.native"][0]["parameters"] == ("theta",)
    assert fixture.declared_facts["artifacts.profiles"] == ("openqasm3",)
    with pytest.raises(TypeError):
        fixture.observed_facts["device.count"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        fixture.observed_facts["gates.native"][0]["name"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        fixture.declared_facts["artifacts.profiles"] = ("qcis",)  # type: ignore[index]

    first = synthetic_remote_target_capability_snapshot(
        fixture=fixture,
        captured_at=_CAPTURED_AT,
        ttl=timedelta(hours=1),
    )
    second = synthetic_remote_target_capability_snapshot(
        fixture=fixture,
        captured_at=_CAPTURED_AT,
        ttl=timedelta(hours=1),
    )
    assert first.snapshot_id == second.snapshot_id
    assert first.to_json() == second.to_json()


@pytest.mark.parametrize(
    ("fact_name", "value", "error_match"),
    (
        ("gates.native", "h", "JSON array"),
        ("memory.available_bytes", nan, "finite"),
        ("memory.available_bytes", object(), "JSON-safe"),
        ("memory.available_bytes", -1, "non-negative integer"),
    ),
)
def test_synthetic_fixture_rejects_invalid_fact_values_early(
    fact_name: str, value: object, error_match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        _remote_fixture(extra_observed_facts={fact_name: value})
