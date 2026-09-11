"""CPU Platform adapter conformance against the Core v1 value objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pytest

from flagquantum.compute import get_platform_runtime, list_platform_status
from flagquantum.compute.cpu_target_capabilities import (
    CPUCapabilityObservation,
    CPUPrecisionObservation,
    cpu_platform_to_target_capability_snapshot,
    probe_local_cpu_target_capabilities,
)
from flagquantum.core.target_capabilities import (
    CapabilityBlocker,
    CapabilityRequirement,
    CapabilityScope,
    ComparisonOperator,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    SupportStatus,
    TargetCapabilitySnapshot,
    match_target_capabilities,
)

pytestmark = pytest.mark.unit

_CAPTURED_AT = datetime(2026, 9, 4, 8, 0, tzinfo=timezone.utc)
_ALL_PRECISION = (
    CPUPrecisionObservation("precision.native_dtype", "float64"),
    CPUPrecisionObservation("precision.effective_dtype", "complex128"),
    CPUPrecisionObservation("precision.storage_dtype", "float64"),
    CPUPrecisionObservation("precision.parameter_dtype", "float32"),
    CPUPrecisionObservation("precision.accumulator_dtype", "float64"),
    CPUPrecisionObservation("precision.software_mechanism", "none"),
)


@dataclass
class _FakeCPUProbe:
    observation: CPUCapabilityObservation
    target_id: str = "cpu-target-test"
    provider_version: str = "torch-test"
    target_revision: str = "probe-revision"
    environment_id: str = "environment-test"
    source_ref: str = "cpu-probe-evidence"
    target_class_source_ref: str = "cpu-static-evidence"
    calls: int = 0

    def observe(self) -> CPUCapabilityObservation:
        self.calls += 1
        return self.observation


def _evidence() -> tuple[EvidenceReference, ...]:
    return (
        EvidenceReference(
            evidence_id="cpu-probe-evidence",
            sha256="a" * 64,
            level=EvidenceLevel.OBSERVABLE,
            scope=CapabilityScope(device_ids=("cpu:0",)),
        ),
        EvidenceReference(
            evidence_id="cpu-static-evidence",
            sha256="b" * 64,
            level=EvidenceLevel.BASIC,
            scope=CapabilityScope(device_ids=("cpu:0",)),
        ),
    )


def _available_probe(**changes: object) -> _FakeCPUProbe:
    observation = CPUCapabilityObservation(
        available=True,
        device_count=1,
        device_ids=("cpu:0",),
        memory_available_bytes=16 * 1024**3,
        precision=_ALL_PRECISION,
    )
    if changes:
        observation = replace(observation, **changes)
    return _FakeCPUProbe(observation)


def test_cpu_adapter_observes_only_injected_device_memory_and_precision() -> None:
    probe = _available_probe()
    snapshot = cpu_platform_to_target_capability_snapshot(
        probe=probe,
        captured_at=_CAPTURED_AT,
        ttl=timedelta(minutes=30),
        evidence_refs=_evidence(),
    )

    assert probe.calls == 1
    assert snapshot.target_identity.target_class == "local_runtime"
    assert snapshot.target_identity.provider == "pytorch_cpu"
    assert snapshot.target_identity.target_id == "cpu-target-test"
    assert snapshot.target_identity.environment_id == "environment-test"
    assert snapshot.scope == CapabilityScope(device_ids=("cpu:0",))
    facts = {item.name: item for item in snapshot.facts}
    assert facts["device.kind"].value == "cpu"
    assert facts["device.kind"].support_status is SupportStatus.VERIFIED
    assert facts["device.count"].value == 1
    assert facts["memory.available_bytes"].value == 16 * 1024**3
    assert facts["precision.native_dtype"].value == "float64"
    assert facts["precision.native_dtype"].fact_exposure is FactExposure.OBSERVED
    assert facts["target.class"].value == "local_runtime"
    assert facts["target.class"].fact_exposure is FactExposure.DECLARED
    assert facts["target.class"].source.ref == "cpu-static-evidence"
    assert facts["device.count"].source.ref == "cpu-probe-evidence"
    evidence_ids = {evidence.evidence_id for evidence in snapshot.evidence_refs}
    assert {fact.source.ref for fact in snapshot.facts} <= evidence_ids
    assert all(item.support_status is SupportStatus.VERIFIED for item in facts.values())
    assert snapshot.blockers == ()

    requirements = RequirementSet(
        requirements=(
            CapabilityRequirement(
                name="target.class",
                operator=ComparisonOperator.EQUALS,
                value="local_runtime",
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.COMPILER,
                minimum_evidence_level=EvidenceLevel.BASIC,
                accepted_exposures=(FactExposure.DECLARED,),
            ),
            CapabilityRequirement(
                name="device.kind",
                operator=ComparisonOperator.EQUALS,
                value="cpu",
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.RUNTIME_PROTOCOL,
                minimum_evidence_level=EvidenceLevel.BASIC,
                accepted_exposures=(FactExposure.OBSERVED,),
            ),
            CapabilityRequirement(
                name="device.count",
                operator=ComparisonOperator.AT_LEAST,
                value=1,
                strength=RequirementStrength.MANDATORY,
                source=RequirementSource.RUNTIME_PROTOCOL,
                minimum_evidence_level=EvidenceLevel.BASIC,
                accepted_exposures=(FactExposure.OBSERVED,),
            ),
        )
    )
    result = match_target_capabilities(
        requirements, snapshot, evaluated_at=_CAPTURED_AT + timedelta(minutes=1)
    )
    assert result.executable is True
    assert result.blockers == ()


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_local_cpu_probe_observes_requested_precision(precision: str) -> None:
    snapshot = probe_local_cpu_target_capabilities(
        precision,
        captured_at=_CAPTURED_AT,
        ttl=timedelta(minutes=1),
    )
    facts = {item.name: item for item in snapshot.facts}

    assert facts["device.kind"].value == "cpu"
    assert facts["device.count"].value == 1
    assert facts["precision.effective_dtype"].value == precision
    assert facts["precision.software_mechanism"].value == "none"
    assert snapshot.evidence_refs[0].level is EvidenceLevel.OBSERVABLE


def test_cpu_adapter_source_and_evidence_round_trip_with_ttl() -> None:
    snapshot = cpu_platform_to_target_capability_snapshot(
        probe=_available_probe(),
        captured_at=_CAPTURED_AT,
        ttl=timedelta(hours=2),
        evidence_refs=_evidence(),
    )

    assert snapshot.captured_at == "2026-09-04T08:00:00Z"
    assert snapshot.valid_until == "2026-09-04T10:00:00Z"
    assert snapshot.facts[0].source.ref == "cpu-probe-evidence"
    assert TargetCapabilitySnapshot.from_json(snapshot.to_json()) == snapshot
    assert json.loads(snapshot.to_json())["evidence_refs"][0]["sha256"] == "a" * 64


def test_cpu_adapter_rejects_basic_evidence_for_observed_facts() -> None:
    basic_probe = EvidenceReference(
        evidence_id="cpu-probe-evidence",
        sha256="a" * 64,
        level=EvidenceLevel.BASIC,
        scope=CapabilityScope(device_ids=("cpu:0",)),
    )
    static = EvidenceReference(
        evidence_id="cpu-static-evidence",
        sha256="b" * 64,
        level=EvidenceLevel.BASIC,
        scope=CapabilityScope(device_ids=("cpu:0",)),
    )

    with pytest.raises(ValueError, match="observable or certification"):
        cpu_platform_to_target_capability_snapshot(
            probe=_available_probe(),
            captured_at=_CAPTURED_AT,
            ttl=timedelta(minutes=1),
            evidence_refs=(basic_probe, static),
        )


def test_verified_cpu_precision_must_be_observed() -> None:
    with pytest.raises(ValueError, match="must be observed"):
        CPUPrecisionObservation(
            "precision.native_dtype",
            "float64",
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
        )


def test_cpu_observation_rejects_string_device_ids() -> None:
    with pytest.raises(ValueError, match="device_ids must be an array"):
        CPUCapabilityObservation(
            available=True,
            device_count=1,
            device_ids="cpu:0",  # type: ignore[arg-type]
        )


def test_cpu_precision_blockers_and_observation_collections_are_immutable() -> None:
    blocker = CapabilityBlocker("probe_failed", "precision probe failed")
    raw_blockers = [blocker]
    precision = CPUPrecisionObservation(
        "precision.native_dtype",
        "float64",
        support_status=SupportStatus.UNSUPPORTED,
        fact_exposure=FactExposure.OBSERVED,
        blockers=raw_blockers,  # type: ignore[arg-type]
    )
    raw_blockers.append(CapabilityBlocker("later", "must not leak"))

    assert precision.blockers == (blocker,)
    assert isinstance(precision.blockers, tuple)
    observation = CPUCapabilityObservation(
        available=True,
        device_count=1,
        device_ids=["cpu:0"],  # type: ignore[arg-type]
        precision=[precision],  # type: ignore[arg-type]
    )
    assert observation.device_ids == ("cpu:0",)
    assert observation.precision == (precision,)
    assert isinstance(observation.device_ids, tuple)
    assert isinstance(observation.precision, tuple)


def test_cpu_adapter_rejects_unresolvable_probe_source() -> None:
    probe = _FakeCPUProbe(_available_probe().observation, source_ref="missing-evidence")

    with pytest.raises(ValueError, match="source_ref must resolve"):
        cpu_platform_to_target_capability_snapshot(
            probe=probe,
            captured_at=_CAPTURED_AT,
            ttl=timedelta(minutes=1),
            evidence_refs=_evidence(),
        )


def test_cpu_missing_memory_and_precision_are_nullable_facts_with_blockers() -> None:
    probe = _available_probe(memory_available_bytes=None, precision=())
    snapshot = cpu_platform_to_target_capability_snapshot(
        probe=probe,
        captured_at=_CAPTURED_AT,
        ttl=timedelta(minutes=5),
        evidence_refs=_evidence(),
    )
    facts = {item.name: item for item in snapshot.facts}

    assert facts["memory.available_bytes"].value is None
    assert facts["memory.available_bytes"].support_status is SupportStatus.UNKNOWN
    assert facts["memory.available_bytes"].fact_exposure is FactExposure.NOT_EXPOSED
    precision_names = {
        "precision.native_dtype",
        "precision.effective_dtype",
        "precision.storage_dtype",
        "precision.parameter_dtype",
        "precision.accumulator_dtype",
        "precision.software_mechanism",
    }
    assert precision_names <= {name for name in facts if name.startswith("precision.")}
    for name in precision_names:
        assert facts[name].value is None
        assert facts[name].support_status is SupportStatus.UNKNOWN
        assert facts[name].fact_exposure is FactExposure.NOT_EXPOSED
        assert facts[name].blockers
    assert snapshot.blockers == ()

    requirement = CapabilityRequirement(
        name="memory.available_bytes",
        operator=ComparisonOperator.AT_LEAST,
        value=1,
        strength=RequirementStrength.MANDATORY,
        source=RequirementSource.RUNTIME_PROTOCOL,
        minimum_evidence_level=EvidenceLevel.BASIC,
        accepted_exposures=(FactExposure.OBSERVED,),
    )
    result = match_target_capabilities(
        requirements=RequirementSet(requirements=(requirement,)),
        snapshot=snapshot,
        evaluated_at=_CAPTURED_AT,
    )
    assert result.executable is False
    assert any(item.code == "fact_unknown" for item in result.blockers)


def test_cpu_unavailable_probe_is_explicit_and_never_replaced() -> None:
    probe = _FakeCPUProbe(
        CPUCapabilityObservation(
            available=False,
            device_count=0,
            device_ids=None,
            memory_available_bytes=None,
            precision=(),
        )
    )
    snapshot = cpu_platform_to_target_capability_snapshot(
        probe=probe,
        captured_at=_CAPTURED_AT,
        ttl=timedelta(minutes=1),
        evidence_refs=_evidence(),
    )
    facts = {item.name: item for item in snapshot.facts}

    assert facts["device.kind"].value == "cpu"
    assert facts["device.kind"].support_status is SupportStatus.UNKNOWN
    assert facts["device.kind"].fact_exposure is FactExposure.NOT_EXPOSED
    assert facts["device.count"].value == 0
    assert facts["device.count"].support_status is SupportStatus.UNSUPPORTED
    assert facts["device.count"].fact_exposure is FactExposure.OBSERVED
    assert any(item.code == "cpu_unavailable" for item in snapshot.blockers)
    assert snapshot.target_identity.target_class == "local_runtime"


def test_cpu_adapter_preserves_explicit_negative_precision_without_promotion() -> None:
    negative = CPUPrecisionObservation(
        "precision.native_dtype",
        "float64",
        support_status=SupportStatus.UNSUPPORTED,
        fact_exposure=FactExposure.OBSERVED,
        blockers=(
            # The adapter accepts provider-owned negative evidence but never
            # changes it into a verified capability.
            CapabilityBlocker(
                "cpu_fp64_probe_failed", "CPU precision probe did not pass"
            ),
        ),
    )
    probe = _available_probe(precision=(negative,))
    snapshot = cpu_platform_to_target_capability_snapshot(
        probe=probe,
        captured_at=_CAPTURED_AT,
        ttl=timedelta(minutes=1),
        evidence_refs=_evidence(),
    )
    fact = next(
        item for item in snapshot.facts if item.name == "precision.native_dtype"
    )

    assert fact.support_status is SupportStatus.UNSUPPORTED
    assert fact.fact_exposure is FactExposure.OBSERVED
    assert fact.blockers


def test_cpu_adapter_rejects_non_deterministic_inputs_and_keeps_platform_defaults() -> (
    None
):
    with pytest.raises(ValueError, match="ttl must be positive"):
        cpu_platform_to_target_capability_snapshot(
            probe=_available_probe(),
            captured_at=_CAPTURED_AT,
            ttl=timedelta(0),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        cpu_platform_to_target_capability_snapshot(
            probe=_available_probe(),
            captured_at=datetime(2026, 9, 4),
            ttl=timedelta(minutes=1),
        )

    status = {item["device_type"]: item for item in list_platform_status()}
    assert get_platform_runtime("cpu").device_type == "cpu"
    assert status["cpu"]["available"] is True
