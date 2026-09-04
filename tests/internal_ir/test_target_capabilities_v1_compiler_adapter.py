from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from flagquantum._compiler.target_capabilities import (
    AncillaPolicy,
    ArtifactFormat,
    ArtifactProfile,
    ControlFlowProfile,
    GateCapability,
    MeasurementResult,
    ParameterConstraint,
    TargetCapabilities,
    TargetClass,
)
from flagquantum._compiler.target_capabilities_adapter import (
    ADAPTER_EXIT_CONDITION,
    ADAPTER_OWNER,
    COMPILER_TARGET_ADAPTER_VERSION,
    CompilerProjectionLossCode,
    legacy_semantic_field_accounting,
    target_capabilities_to_requirement_set,
)
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    FallbackAuthorizations,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
    match_target_capabilities,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/internal_ir/phase3_batch_a_target_capabilities.json"
NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _target(**updates: object) -> TargetCapabilities:
    target = TargetCapabilities(
        target_class=TargetClass.LOCAL_RUNTIME,
        logical_qubit_capacity=2,
        physical_qubit_capacity=4,
        native_gates=(
            GateCapability("cx"),
            GateCapability("rx", (ParameterConstraint("theta", -1.0, 1.0),)),
        ),
        measurement_results=(
            MeasurementResult.EXPECTATION,
            MeasurementResult.STATE,
        ),
        artifact_profiles=(ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0"),),
        maximum_shots=1024,
        maximum_program_operations=4096,
        ancilla_policy=AncillaPolicy.EXPLICIT_ONLY,
        maximum_compiler_ancillas=0,
        display_label="non-semantic-label",
    )
    return replace(target, **updates)


def _snapshot_from_available(target: TargetCapabilities) -> TargetCapabilitySnapshot:
    """Test-only fact fixture; production discovery remains Platform-owned."""

    projected = target_capabilities_to_requirement_set(target).requirement_set
    scope = CapabilityScope()
    evidence_id = "test-only-compiler-static-facts"
    facts = tuple(
        CapabilityFact(
            name=requirement.name,
            value=requirement.to_dict()["value"],
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=FactSource(kind="test_fixture", ref=evidence_id),
        )
        for requirement in projected.requirements
    )
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="test-target",
            target_class=target.target_class.value,
            provider="test-only",
            provider_version="1",
            target_revision="fixture",
            environment_id="unit-test",
        ),
        scope=scope,
        captured_at="2026-09-04T00:00:00Z",
        valid_until="2026-09-05T00:00:00Z",
        facts=facts,
        evidence_refs=(
            EvidenceReference(
                evidence_id=evidence_id,
                sha256="a" * 64,
                level=EvidenceLevel.BASIC,
                scope=scope,
            ),
        ),
    )


def test_adapter_projects_only_compiler_requirements_and_forbids_fallback() -> None:
    projection = target_capabilities_to_requirement_set(_target())
    requirements = {item.name: item for item in projection.requirement_set.requirements}

    assert set(requirements) == {
        "target.class",
        "qubits.logical_capacity",
        "qubits.physical_capacity",
        "measurements.results",
        "artifacts.profiles",
        "limits.maximum_shots",
        "limits.maximum_program_operations",
        "ancillas.maximum_compiler",
    }
    assert (
        projection.requirement_set.fallback_authorizations == FallbackAuthorizations()
    )
    assert not projection.requirement_set.extensions
    assert all(item.source.value == "compiler" for item in requirements.values())
    assert ADAPTER_OWNER == "compiler"
    assert COMPILER_TARGET_ADAPTER_VERSION == "1.0"
    assert "golden legacy schema" in ADAPTER_EXIT_CONDITION
    assert projection.requires_legacy_comparator is True


def test_exact_projection_matches_with_core_using_test_only_static_facts() -> None:
    target = _target()
    requirements = target_capabilities_to_requirement_set(target).requirement_set
    result = match_target_capabilities(
        requirements,
        _snapshot_from_available(target),
        evaluated_at=NOW,
    )

    assert result.executable is True
    assert result.blockers == ()


def test_adapter_round_trip_preserves_every_golden_payload_and_fingerprint() -> None:
    before = json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"]

    for record in before:
        target = TargetCapabilities.from_dict(record["payload"])
        fingerprint = target.semantic_fingerprint
        payload = target.to_dict()
        projection = target_capabilities_to_requirement_set(target)

        restored = projection.restore_legacy(display_label=target.display_label)
        assert restored.to_dict() == payload
        assert restored.semantic_fingerprint == fingerprint
        assert projection.legacy_semantic_fingerprint == record["expected_fingerprint"]
        assert target.to_dict() == payload
        assert target.semantic_fingerprint == fingerprint


def test_every_legacy_semantic_field_is_projected_or_has_a_typed_loss() -> None:
    target = _target(
        control_flow=ControlFlowProfile.ADAPTIVE_REAL_TIME,
        supports_mid_circuit_measurement=True,
        supports_reset=True,
        supports_timing=True,
        supports_pulse=True,
        supports_noise=True,
        supports_parameter_binding=True,
        calibration_snapshot_hash="b" * 64,
        calibration_valid_until="2026-09-05T00:00:00Z",
    )
    projection = target_capabilities_to_requirement_set(target)
    losses = {item.field: item for item in projection.losses}
    accounting = legacy_semantic_field_accounting()
    semantic_fields = set(target.semantic_dict()) - {"schema_version"}

    assert set(accounting) == semantic_fields
    assert {field for field, state in accounting.items() if state == "deferred"} <= set(
        losses
    )
    assert losses["supports_pulse"].active is True
    assert losses["calibration_snapshot_hash"].active is True
    assert all(
        item.authority == "compiler.TargetCapabilities" for item in losses.values()
    )
    assert all(json.dumps(item.to_dict()) for item in losses.values())


def test_inactive_deferred_fields_are_still_explicitly_accounted() -> None:
    losses = {
        item.field: item
        for item in target_capabilities_to_requirement_set(_target()).losses
    }

    assert losses["topology"].active is False
    assert losses["supports_noise"].active is False
    assert losses["calibration_valid_until"].active is False
    assert losses["topology"].code is CompilerProjectionLossCode.DEFERRED_FROM_CORE_V1


def test_richer_gate_coverage_stays_under_legacy_comparator_authority() -> None:
    required = _target()
    available = _target(
        native_gates=(GateCapability("cx"), GateCapability("rx")),
    )
    projection = target_capabilities_to_requirement_set(required)
    legacy = projection.compare_available(available)
    core = match_target_capabilities(
        projection.requirement_set,
        _snapshot_from_available(available),
        evaluated_at=NOW,
    )

    assert legacy.compatible is True
    assert core.executable is True
    assert core.blockers == ()
    gate_loss = next(item for item in projection.losses if item.field == "native_gates")
    assert gate_loss.code is CompilerProjectionLossCode.LEGACY_COMPARATOR_REQUIRED


def test_ancilla_policy_ordering_stays_under_legacy_comparator_authority() -> None:
    required = _target()
    available = _target(
        ancilla_policy=AncillaPolicy.CLEAN_ALLOCATABLE,
        maximum_compiler_ancillas=1,
    )
    projection = target_capabilities_to_requirement_set(required)

    assert projection.compare_available(available).compatible is True
    core = match_target_capabilities(
        projection.requirement_set,
        _snapshot_from_available(available),
        evaluated_at=NOW,
    )
    assert core.executable is True
    assert core.blockers == ()


def test_legacy_comparator_still_fails_closed_for_unexpressible_requirements() -> None:
    required = _target(supports_noise=True)
    available = _target(supports_noise=False)
    projection = target_capabilities_to_requirement_set(required)

    comparison = projection.compare_available(available)
    assert comparison.compatible is False
    assert {item.field for item in comparison.differences} == {"supports_noise"}
    noise_loss = next(
        item for item in projection.losses if item.field == "supports_noise"
    )
    assert noise_loss.active is True
    assert noise_loss.code is CompilerProjectionLossCode.DEFERRED_FROM_CORE_V1


@pytest.mark.parametrize("bad_value", [None, object(), "not-a-target"])
def test_adapter_rejects_non_target_inputs(bad_value: object) -> None:
    with pytest.raises(TypeError, match="Compiler TargetCapabilities"):
        target_capabilities_to_requirement_set(bad_value)  # type: ignore[arg-type]


def test_compiler_adapter_does_not_claim_platform_snapshot_authority() -> None:
    import flagquantum._compiler.target_capabilities_adapter as adapter

    assert not hasattr(adapter, "target_capabilities_to_snapshot")
    assert not hasattr(adapter, "CompilerSnapshotProjection")
