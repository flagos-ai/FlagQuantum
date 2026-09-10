from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest

import flagquantum.experimental.targets as targets
from flagquantum.core import target_capabilities as core
from flagquantum.experimental.targets import (
    CapabilityMatch,
    RequirementSet,
    TargetSnapshot,
    dump_requirement_set,
    dump_target_snapshot,
    load_requirement_set,
    load_target_snapshot,
    match_capabilities,
)

pytestmark = pytest.mark.unit
_NOW = datetime(2026, 9, 10, 20, 0, tzinfo=timezone.utc)


def _requirement_set() -> core.RequirementSet:
    common = {
        "source": core.RequirementSource.COMPILER,
        "minimum_evidence_level": core.EvidenceLevel.BASIC,
        "accepted_exposures": (core.FactExposure.DECLARED,),
    }
    return core.RequirementSet(
        requirements=(
            core.CapabilityRequirement(
                name="qubits.logical_capacity",
                operator=core.ComparisonOperator.AT_LEAST,
                value=2,
                strength=core.RequirementStrength.MANDATORY,
                **common,
            ),
            core.CapabilityRequirement(
                name="gates.native",
                operator=core.ComparisonOperator.CONTAINS_ALL,
                value=("h", "cx"),
                strength=core.RequirementStrength.PREFERENCE,
                **common,
            ),
        )
    )


def _snapshot() -> core.TargetCapabilitySnapshot:
    scope = core.CapabilityScope(device_ids=("preview:0",))
    source = core.FactSource(kind="phase53_test", ref="phase53-evidence")
    return core.TargetCapabilitySnapshot(
        target_identity=core.TargetIdentity(
            target_id="phase53-target",
            target_class="simulator",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase53-environment",
        ),
        scope=scope,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=(
            core.CapabilityFact(
                name="qubits.logical_capacity",
                value=4,
                support_status=core.SupportStatus.VERIFIED,
                fact_exposure=core.FactExposure.DECLARED,
                source=source,
            ),
            core.CapabilityFact(
                name="gates.native",
                value=("h", "cx", "rz"),
                support_status=core.SupportStatus.VERIFIED,
                fact_exposure=core.FactExposure.DECLARED,
                source=source,
            ),
            core.CapabilityFact(
                name="device.count",
                value=None,
                support_status=core.SupportStatus.UNMEASURED,
                fact_exposure=core.FactExposure.UNKNOWN,
                source=source,
                blockers=(
                    core.CapabilityBlocker(
                        code="measurement_unavailable",
                        message="device count was not measured",
                        capability_name="device.count",
                    ),
                ),
            ),
        ),
        evidence_refs=(
            core.EvidenceReference(
                evidence_id="phase53-evidence",
                sha256="a" * 64,
                level=core.EvidenceLevel.BASIC,
                scope=scope,
            ),
        ),
    )


def test_round_trip_and_fact_state_inspection() -> None:
    snapshot_value, requirements_value = _snapshot(), _requirement_set()
    snapshot = load_target_snapshot(snapshot_value.to_json())
    requirements = load_requirement_set(requirements_value.to_json())
    assert (snapshot.version, requirements.version) == ("1.0", "1.0")
    assert snapshot.identity == snapshot_value.snapshot_id
    assert requirements.identity == requirements_value.requirement_set_id
    assert (snapshot.target_id, snapshot.target_class, snapshot.provider) == (
        "phase53-target",
        "simulator",
        "flagquantum.test",
    )
    assert snapshot.capability_names == (
        "device.count",
        "gates.native",
        "qubits.logical_capacity",
    )
    unmeasured = snapshot.fact("device.count")
    assert isinstance(unmeasured, MappingProxyType)
    assert (unmeasured["support_status"], unmeasured["fact_exposure"]) == (
        "unmeasured",
        "unknown",
    )
    assert snapshot.fact("memory.available_bytes") is None
    with pytest.raises(ValueError, match="unknown v1 capability"):
        snapshot.fact("provider.secret")
    assert requirements.requirement_count == 2
    assert dump_target_snapshot(snapshot) == snapshot_value.to_json()
    assert dump_requirement_set(requirements) == requirements_value.to_json()


def test_match_binds_inputs_time_preferences_and_stale_blocker() -> None:
    snapshot = load_target_snapshot(_snapshot().to_json())
    requirements = load_requirement_set(_requirement_set().to_json())
    matched = match_capabilities(
        requirements,
        snapshot,
        evaluated_at=datetime(
            2026, 9, 11, 4, 30, 0, 123, tzinfo=timezone(timedelta(hours=8))
        ),
    )
    assert isinstance(matched, CapabilityMatch)
    assert matched.executable and matched.blockers == ()
    assert (matched.satisfied_preferences, matched.total_preferences) == (1, 1)
    assert matched.requirement_identity == requirements.identity
    assert matched.snapshot_identity == snapshot.identity
    assert matched.evaluated_at == "2026-09-10T20:30:00.000123Z"
    stale = match_capabilities(
        requirements, snapshot, evaluated_at=_NOW + timedelta(hours=1)
    )
    assert not stale.executable
    assert stale.blockers[0]["code"] == "snapshot_stale"
    assert isinstance(stale.blockers[0], MappingProxyType)
    future = match_capabilities(
        requirements, snapshot, evaluated_at=_NOW - timedelta(microseconds=1)
    )
    assert not future.executable
    assert future.blockers[0]["code"] == "snapshot_not_yet_valid"


def test_views_are_frozen_detached_and_reject_lookalikes() -> None:
    snapshot_value, requirements_value = _snapshot(), _requirement_set()
    snapshot = load_target_snapshot(snapshot_value.to_json())
    requirements = load_requirement_set(requirements_value.to_json())
    detached = snapshot.to_dict()
    detached["captured_at"] = "changed"
    assert snapshot.captured_at == "2026-09-10T20:00:00Z"
    assert requirements.requirement_count == 2
    with pytest.raises(FrozenInstanceError):
        snapshot._value = object()  # type: ignore[misc]
    with pytest.raises(TypeError, match="TargetSnapshot view"):
        dump_target_snapshot(snapshot_value)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RequirementSet view"):
        dump_requirement_set(requirements_value)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="loaded Core snapshot"):
        TargetSnapshot({})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="loaded Core requirement set"):
        RequirementSet({})  # type: ignore[arg-type]


@pytest.mark.parametrize("loader", [load_target_snapshot, load_requirement_set])
def test_loaders_reject_invalid_boundary_inputs(loader: object) -> None:
    with pytest.raises(TypeError, match="JSON text"):
        loader(b"{}")  # type: ignore[operator]
    with pytest.raises(ValueError, match="duplicate .* field 'schema_version'"):
        loader('{"schema_version":"1.0","schema_version":"1.0"}')  # type: ignore[operator]
    with pytest.raises(ValueError, match="duplicate .* field 'nested'"):
        loader('{"outer":{"nested":1,"nested":2}}')  # type: ignore[operator]
    with pytest.raises(ValueError, match="must be valid JSON"):
        loader("{")  # type: ignore[operator]
    with pytest.raises(ValueError, match="must be a JSON object"):
        loader("[]")  # type: ignore[operator]
    with pytest.raises(ValueError, match="maximum UTF-8 bytes"):
        loader(" " * (16 * 1024 * 1024 + 1))  # type: ignore[operator]


def test_match_requires_exact_views_and_aware_datetime() -> None:
    snapshot = load_target_snapshot(_snapshot().to_json())
    requirements = load_requirement_set(_requirement_set().to_json())
    with pytest.raises(TypeError, match="datetime"):
        match_capabilities(requirements, snapshot, evaluated_at="now")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="include a timezone"):
        match_capabilities(
            requirements, snapshot, evaluated_at=datetime(2026, 9, 10, 20, 30)
        )
    with pytest.raises(TypeError, match="TargetSnapshot view"):
        match_capabilities(
            requirements, _snapshot(), evaluated_at=_NOW  # type: ignore[arg-type]
        )


def test_exact_exports_lazy_import_and_stable_api_preservation() -> None:
    assert targets.__all__ == (
        "CapabilityMatch",
        "RequirementSet",
        "TargetSnapshot",
        "dump_requirement_set",
        "dump_target_snapshot",
        "load_requirement_set",
        "load_target_snapshot",
        "match_capabilities",
    )
    script = (
        "import sys, flagquantum, flagquantum.experimental; "
        "print('flagquantum.experimental.targets' in sys.modules); "
        "print('TargetSnapshot' in flagquantum.__all__)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, check=True
    )
    assert completed.stdout.splitlines() == ["False", "False"]


def test_identity_tampering_and_unknown_fields_fail_closed() -> None:
    snapshot = json.loads(_snapshot().to_json())
    snapshot["snapshot_id"] = "0" * 64
    with pytest.raises(ValueError, match="snapshot_id does not match"):
        load_target_snapshot(json.dumps(snapshot))
    requirements = json.loads(_requirement_set().to_json())
    requirements["unexpected"] = True
    with pytest.raises(ValueError, match="unknown requirement set field"):
        load_requirement_set(json.dumps(requirements))
