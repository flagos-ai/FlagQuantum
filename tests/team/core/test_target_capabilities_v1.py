from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from flagquantum.core.target_capabilities import (
    CapabilityBlocker,
    CapabilityContractError,
    CapabilityFact,
    CapabilityIdentityError,
    CapabilityRequirement,
    CapabilityScope,
    CapabilityVersionError,
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
    UnknownCapabilityFieldError,
    match_target_capabilities,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
_HASH = "a" * 64
_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "target_capabilities_v1_minimal.json"
)


def _requirement(
    name: str = "device.count",
    value: object = 2,
    *,
    operator: ComparisonOperator = ComparisonOperator.AT_LEAST,
    strength: RequirementStrength = RequirementStrength.MANDATORY,
    minimum_evidence_level: EvidenceLevel = EvidenceLevel.OBSERVABLE,
    accepted_exposures: tuple[FactExposure, ...] = (FactExposure.OBSERVED,),
) -> CapabilityRequirement:
    return CapabilityRequirement(
        name=name,
        operator=operator,
        value=value,
        strength=strength,
        source=RequirementSource.USER,
        minimum_evidence_level=minimum_evidence_level,
        accepted_exposures=accepted_exposures,
    )


def _fact(
    name: str = "device.count",
    value: object = 4,
    *,
    status: SupportStatus = SupportStatus.VERIFIED,
    exposure: FactExposure = FactExposure.OBSERVED,
    evidence_id: str = "probe-1",
) -> CapabilityFact:
    blockers = ()
    if status is not SupportStatus.VERIFIED:
        blockers = (CapabilityBlocker("not_verified", "provider did not verify", name),)
    return CapabilityFact(
        name=name,
        value=value,
        support_status=status,
        fact_exposure=exposure,
        source=FactSource("probe", evidence_id),
        blockers=blockers,
    )


def _identity(target_id: str = "cpu-local") -> TargetIdentity:
    return TargetIdentity(
        target_id=target_id,
        target_class="local_runtime",
        provider="flagquantum",
        provider_version="0.0",
        target_revision="test",
        environment_id="test-environment",
    )


def _scope(**changes: object) -> CapabilityScope:
    values: dict[str, object] = {
        "device_ids": ("cpu:0",),
        "dtype": "complex128",
        "kernel": "statevector",
        "workload_id": "fixture",
        "world_size": 1,
        "node_count": 1,
    }
    values.update(changes)
    return CapabilityScope(**values)  # type: ignore[arg-type]


def _evidence(
    evidence_id: str = "probe-1",
    *,
    level: EvidenceLevel = EvidenceLevel.OBSERVABLE,
    scope: CapabilityScope | None = None,
) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=evidence_id,
        sha256=_HASH,
        level=level,
        scope=scope or _scope(),
    )


def _snapshot(
    *facts: CapabilityFact,
    evidence_refs: tuple[EvidenceReference, ...] | None = None,
    scope: CapabilityScope | None = None,
    identity: TargetIdentity | None = None,
) -> TargetCapabilitySnapshot:
    selected_scope = scope or _scope()
    return TargetCapabilitySnapshot(
        target_identity=identity or _identity(),
        scope=selected_scope,
        captured_at="2026-09-03T00:00:00Z",
        valid_until="2026-09-04T00:00:00Z",
        facts=facts or (_fact(),),
        evidence_refs=evidence_refs or (_evidence(scope=selected_scope),),
    )


class FakeCapabilityProducer:
    """Replaceable producer fake proving the Core consumer contract."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id

    def discover(self) -> TargetCapabilitySnapshot:
        return _snapshot(identity=_identity(self.target_id))


def test_machine_authorized_shapes_are_exact_and_internal() -> None:
    assert tuple(item.name for item in fields(CapabilityRequirement)) == (
        "name",
        "operator",
        "value",
        "strength",
        "source",
        "minimum_evidence_level",
        "accepted_exposures",
    )
    assert tuple(item.name for item in fields(RequirementSet)) == (
        "schema_version",
        "requirement_set_id",
        "requirements",
        "fallback_authorizations",
        "extensions",
    )
    assert tuple(item.name for item in fields(TargetCapabilitySnapshot)) == (
        "schema_version",
        "snapshot_id",
        "target_identity",
        "scope",
        "captured_at",
        "valid_until",
        "facts",
        "evidence_refs",
        "blockers",
        "extensions",
    )

    import flagquantum as fq
    import flagquantum.core as core

    assert not hasattr(fq, "CapabilityRequirement")
    assert not hasattr(core, "CapabilityRequirement")


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("precision.native_dtype", "complex128"),
        ("precision.storage_dtype", "complex64"),
        ("precision.parameter_dtype", "complex64"),
        ("precision.accumulator_dtype", "complex128"),
        ("precision.effective_dtype", "float64"),
    ),
)
def test_precision_fields_reject_the_wrong_dtype_category(
    name: str, value: str
) -> None:
    with pytest.raises(CapabilityContractError):
        _fact(name, value)
    with pytest.raises(CapabilityContractError):
        _requirement(name, value, operator=ComparisonOperator.EQUALS)


def test_native_precision_path_must_have_consistent_dtypes() -> None:
    facts = (
        _fact("precision.native_dtype", "float64"),
        _fact("precision.effective_dtype", "complex128"),
        _fact("precision.storage_dtype", "float32"),
        _fact("precision.software_mechanism", "none"),
    )

    with pytest.raises(CapabilityContractError, match="inconsistent"):
        _snapshot(*facts)


def test_strict_round_trip_and_canonical_identity_are_order_independent() -> None:
    first = _requirement()
    second = _requirement(
        "target.class",
        "local_runtime",
        operator=ComparisonOperator.EQUALS,
        minimum_evidence_level=EvidenceLevel.BASIC,
        accepted_exposures=(FactExposure.DECLARED, FactExposure.OBSERVED),
    )
    requirements = RequirementSet(
        requirements=(first, second, first),
        fallback_authorizations=FallbackAuthorizations(cpu=True),
        extensions={"org.flagquantum.test": {"z": [2, 1], "a": True}},
    )
    reordered = RequirementSet(
        requirements=(second, first),
        fallback_authorizations=FallbackAuthorizations(cpu=True),
        extensions={"org.flagquantum.test": {"a": True, "z": [2, 1]}},
    )

    assert requirements.requirement_set_id == reordered.requirement_set_id
    assert requirements.to_json() == reordered.to_json()
    assert RequirementSet.from_json(requirements.to_json()) == requirements
    with pytest.raises(FrozenInstanceError):
        requirements.requirement_set_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        requirements.extensions["org.flagquantum.test"] = {}  # type: ignore[index]

    snapshot = _snapshot()
    assert TargetCapabilitySnapshot.from_json(snapshot.to_json()) == snapshot
    assert json.loads(snapshot.to_json())["snapshot_id"] == snapshot.snapshot_id


def test_minimal_snapshot_matches_the_exact_versioned_json_fixture() -> None:
    fixture_payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    snapshot = _snapshot()

    assert snapshot.to_dict() == fixture_payload
    assert TargetCapabilitySnapshot.from_dict(fixture_payload) == snapshot
    assert set(fixture_payload) == {
        "schema_version",
        "snapshot_id",
        "target_identity",
        "scope",
        "captured_at",
        "valid_until",
        "facts",
        "evidence_refs",
        "blockers",
        "extensions",
    }


@pytest.mark.parametrize("kind", ["requirement", "snapshot"])
def test_strict_reader_rejects_unknown_fields_versions_and_wrong_identity(
    kind: str,
) -> None:
    value = (
        RequirementSet(requirements=(_requirement(),))
        if kind == "requirement"
        else _snapshot()
    )
    payload = value.to_dict()
    payload["unexpected"] = True
    reader = (
        RequirementSet.from_dict
        if kind == "requirement"
        else TargetCapabilitySnapshot.from_dict
    )
    with pytest.raises(UnknownCapabilityFieldError):
        reader(payload)

    payload = value.to_dict()
    payload["schema_version"] = "2.0"
    with pytest.raises(CapabilityVersionError):
        reader(payload)

    payload = value.to_dict()
    identity_name = "requirement_set_id" if kind == "requirement" else "snapshot_id"
    payload[identity_name] = "0" * 64
    with pytest.raises(CapabilityIdentityError):
        reader(payload)


def test_rejects_unknown_enum_name_non_json_and_conflicting_mandatory_predicates() -> (
    None
):
    payload = _requirement().to_dict()
    payload["operator"] = "approximately"
    with pytest.raises(CapabilityContractError, match="operator"):
        CapabilityRequirement.from_dict(payload)
    with pytest.raises(CapabilityContractError, match="unknown v1"):
        _requirement("runtime.secret", "x", operator=ComparisonOperator.EQUALS)
    with pytest.raises(CapabilityContractError, match="JSON-safe"):
        _requirement(
            "gates.native",
            {"h"},
            operator=ComparisonOperator.CONTAINS_ALL,
        )
    with pytest.raises(CapabilityContractError, match="conflicting mandatory"):
        RequirementSet(
            requirements=(
                _requirement(value=8),
                _requirement(value=4, operator=ComparisonOperator.AT_MOST),
            )
        )


def test_scope_rejects_a_string_instead_of_a_device_id_sequence() -> None:
    with pytest.raises(CapabilityContractError, match="array or tuple"):
        CapabilityScope(device_ids="cpu:0")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("operator", "available", "required", "executable"),
    [
        (ComparisonOperator.EQUALS, "gpu", "gpu", True),
        (ComparisonOperator.AT_LEAST, 4, 2, True),
        (ComparisonOperator.AT_MOST, 4, 2, False),
        (ComparisonOperator.CONTAINS_ALL, ["h", "cx"], ["cx"], True),
        (ComparisonOperator.COVERS, ["h", "cx"], ["h"], True),
    ],
)
def test_pure_matcher_implements_all_closed_operators(
    operator: ComparisonOperator,
    available: object,
    required: object,
    executable: bool,
) -> None:
    name = "device.kind" if operator is ComparisonOperator.EQUALS else "device.count"
    if operator in {ComparisonOperator.CONTAINS_ALL, ComparisonOperator.COVERS}:
        name = "gates.native"
    requirements = RequirementSet(
        requirements=(_requirement(name, required, operator=operator),)
    )
    result = match_target_capabilities(
        requirements,
        _snapshot(_fact(name, available)),
        evaluated_at=_NOW,
    )
    assert result.executable is executable
    assert bool(result.blockers) is not executable


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (SupportStatus.UNKNOWN, "fact_unknown"),
        (SupportStatus.UNMEASURED, "fact_unmeasured"),
        (SupportStatus.UNSUPPORTED, "fact_unsupported"),
    ],
)
def test_mandatory_unknown_unmeasured_and_unsupported_fail_closed(
    status: SupportStatus, expected_code: str
) -> None:
    result = match_target_capabilities(
        RequirementSet(requirements=(_requirement(),)),
        _snapshot(_fact(status=status)),
        evaluated_at=_NOW,
    )
    assert result.executable is False
    assert [item.code for item in result.blockers] == [expected_code]


def test_missing_not_exposed_stale_scope_and_identity_fail_closed() -> None:
    requirements = RequirementSet(requirements=(_requirement(),))
    missing = _snapshot(_fact("target.class", "local_runtime"))
    result = match_target_capabilities(requirements, missing, evaluated_at=_NOW)
    assert result.blockers[0].code == "missing_fact"

    with pytest.raises(CapabilityContractError, match="verified facts cannot use"):
        _fact(exposure=FactExposure.NOT_EXPOSED)

    result = match_target_capabilities(
        requirements,
        _snapshot(),
        evaluated_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        expected_target_identity=_identity("another"),
        required_scope=_scope(world_size=2),
    )
    assert {item.code for item in result.blockers} == {
        "snapshot_stale",
        "target_identity_mismatch",
        "scope_mismatch",
    }


@pytest.mark.parametrize(
    "exposure",
    [
        FactExposure.UNKNOWN,
        FactExposure.NOT_EXPOSED,
    ],
)
def test_unknown_and_not_exposed_verified_combinations_are_rejected(
    exposure: FactExposure,
) -> None:
    with pytest.raises(CapabilityContractError, match="verified facts cannot use"):
        _fact("target.class", "local_runtime", exposure=exposure)


def test_snapshot_blockers_fail_closed_and_are_deterministic() -> None:
    blocker_a = CapabilityBlocker("z_blocker", "second")
    blocker_b = CapabilityBlocker("a_blocker", "first")
    snapshot = replace(
        _snapshot(),
        blockers=(blocker_a, blocker_b),
        snapshot_id="",
    )

    result = match_target_capabilities(
        RequirementSet(requirements=(_requirement(),)),
        snapshot,
        evaluated_at=_NOW,
    )

    assert result.executable is False
    assert [item.code for item in result.blockers] == ["a_blocker", "z_blocker"]


def test_verified_fact_cannot_carry_an_unclassified_blocker() -> None:
    with pytest.raises(CapabilityContractError, match="verified facts"):
        CapabilityFact(
            name="device.count",
            value=4,
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.OBSERVED,
            source=FactSource("probe", "probe-1"),
            blockers=(CapabilityBlocker("partial", "unclassified limitation"),),
        )


def test_equivalent_timezone_offsets_have_one_canonical_snapshot_identity() -> None:
    utc_snapshot = _snapshot()
    offset_snapshot = replace(
        utc_snapshot,
        captured_at="2026-09-03T08:00:00+08:00",
        valid_until="2026-09-04T08:00:00+08:00",
        snapshot_id="",
    )

    assert offset_snapshot.captured_at == "2026-09-03T00:00:00Z"
    assert offset_snapshot.valid_until == "2026-09-04T00:00:00Z"
    assert offset_snapshot.snapshot_id == utc_snapshot.snapshot_id


def test_evidence_threshold_is_strongest_mandatory_and_weak_evidence_fails() -> None:
    requirements = RequirementSet(
        requirements=(
            _requirement(minimum_evidence_level=EvidenceLevel.BASIC),
            _requirement(
                "target.class",
                "local_runtime",
                operator=ComparisonOperator.EQUALS,
                minimum_evidence_level=EvidenceLevel.CERTIFICATION,
                accepted_exposures=(FactExposure.DECLARED,),
            ),
        )
    )
    snapshot = _snapshot(
        _fact(),
        _fact(
            "target.class",
            "local_runtime",
            exposure=FactExposure.DECLARED,
            evidence_id="spec-1",
        ),
        evidence_refs=(
            _evidence(level=EvidenceLevel.OBSERVABLE),
            _evidence("spec-1", level=EvidenceLevel.CERTIFICATION),
        ),
    )
    result = match_target_capabilities(requirements, snapshot, evaluated_at=_NOW)
    assert result.executable is False
    assert [item.code for item in result.blockers] == ["insufficient_evidence"]


def test_preference_does_not_reject_candidate_or_authorize_fallback() -> None:
    requirements = RequirementSet(
        requirements=(
            _requirement(),
            _requirement(
                "precision.native_dtype",
                "float64",
                operator=ComparisonOperator.EQUALS,
                strength=RequirementStrength.PREFERENCE,
            ),
        )
    )
    result = match_target_capabilities(requirements, _snapshot(), evaluated_at=_NOW)
    assert result.executable is True
    assert result.satisfied_preferences == 0
    assert result.total_preferences == 1
    assert requirements.fallback_authorizations == FallbackAuthorizations()


def test_double_single_can_satisfy_effective_but_never_native_precision() -> None:
    facts = (
        _fact("precision.effective_dtype", "complex128", evidence_id="effective"),
        _fact("precision.native_dtype", "float32", evidence_id="native"),
        _fact("precision.storage_dtype", "float32", evidence_id="storage"),
        _fact("precision.software_mechanism", "double-single", evidence_id="mechanism"),
    )
    evidences = tuple(
        _evidence(item, level=EvidenceLevel.OBSERVABLE)
        for item in ("effective", "native", "storage", "mechanism")
    )
    effective = RequirementSet(
        requirements=(
            _requirement(
                "precision.effective_dtype",
                "complex128",
                operator=ComparisonOperator.EQUALS,
            ),
        )
    )
    native = RequirementSet(
        requirements=(
            _requirement(
                "precision.native_dtype",
                "float64",
                operator=ComparisonOperator.EQUALS,
            ),
        )
    )
    snapshot = _snapshot(*facts, evidence_refs=evidences)

    assert match_target_capabilities(effective, snapshot, evaluated_at=_NOW).executable
    native_result = match_target_capabilities(native, snapshot, evaluated_at=_NOW)
    assert native_result.executable is False
    assert native_result.blockers[0].code == "value_mismatch"


def test_fallback_axes_are_independent_and_default_forbidden() -> None:
    assert FallbackAuthorizations().to_dict() == {
        "backend": False,
        "device": False,
        "cpu": False,
        "precision": False,
        "algorithm": False,
        "approximation": False,
    }
    only_backend = FallbackAuthorizations(backend=True)
    assert only_backend.backend is True
    assert only_backend.cpu is False
    assert only_backend.precision is False


def test_two_fake_producers_conform_without_consumer_changes() -> None:
    requirements = RequirementSet(requirements=(_requirement(),))
    for producer in (
        FakeCapabilityProducer("fake-a"),
        FakeCapabilityProducer("fake-b"),
    ):
        result = match_target_capabilities(
            requirements,
            producer.discover(),
            evaluated_at=_NOW,
        )
        assert result.executable is True


def test_core_covers_is_conservative_for_structured_artifact_profiles() -> None:
    available = [{"format": "qir", "version": "1.0", "vendor": "example"}]
    required_subset = [{"format": "qir", "version": "1.0"}]
    requirement = _requirement(
        "artifacts.profiles",
        required_subset,
        operator=ComparisonOperator.COVERS,
    )
    snapshot = _snapshot(_fact("artifacts.profiles", available))

    result = match_target_capabilities(
        RequirementSet(requirements=(requirement,)),
        snapshot,
        evaluated_at=_NOW,
    )

    assert result.executable is False
    assert result.blockers[0].code == "value_mismatch"


def test_unknown_requirement_extension_handler_fails_closed_but_snapshot_round_trips() -> (
    None
):
    requirements = RequirementSet(
        requirements=(_requirement(),),
        extensions={"org.flagquantum.future": {"required": True}},
    )
    snapshot = replace(
        _snapshot(),
        extensions={"org.flagquantum.vendor": {"opaque": [1, 2]}},
        snapshot_id="",
    )
    assert TargetCapabilitySnapshot.from_json(snapshot.to_json()) == snapshot
    result = match_target_capabilities(requirements, snapshot, evaluated_at=_NOW)
    assert result.executable is False
    assert result.blockers[0].code == "unknown_extension_handler"
