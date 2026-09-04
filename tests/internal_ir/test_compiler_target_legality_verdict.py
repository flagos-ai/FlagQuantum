from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from flagquantum._compiler.target_capabilities import (
    AncillaPolicy,
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    ParameterConstraint,
    TargetCapabilities,
    TargetClass,
)
from flagquantum._compiler.target_capabilities_adapter import (
    target_capabilities_to_requirement_set,
)
from flagquantum._compiler.target_legality_verdict import (
    COMPILER_TARGET_LEGALITY_SCOPE,
    COMPILER_TARGET_LEGALITY_VERDICT_SCHEMA,
    evaluate_compiler_target_legality,
)
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
    match_target_capabilities,
)

pytestmark = pytest.mark.unit

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
        measurement_results=(MeasurementResult.EXPECTATION, MeasurementResult.STATE),
        artifact_profiles=(ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0"),),
        maximum_shots=1024,
        maximum_program_operations=4096,
        ancilla_policy=AncillaPolicy.EXPLICIT_ONLY,
        maximum_compiler_ancillas=0,
        display_label="required-label",
    )
    return replace(target, **updates)


def _snapshot_for_projection(projection) -> TargetCapabilitySnapshot:
    scope = CapabilityScope()
    evidence_id = "test-only-compiler-legality-facts"
    facts = tuple(
        CapabilityFact(
            name=requirement.name,
            value=requirement.to_dict()["value"],
            support_status=SupportStatus.VERIFIED,
            fact_exposure=FactExposure.DECLARED,
            source=FactSource(kind="test_fixture", ref=evidence_id),
        )
        for requirement in projection.requirement_set.requirements
    )
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="test-target",
            target_class="local_runtime",
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


def test_success_verdict_binds_projection_available_and_loss_accounting() -> None:
    required = _target()
    projection = target_capabilities_to_requirement_set(required)
    verdict = evaluate_compiler_target_legality(projection, required)

    assert verdict.verdict is True
    assert verdict.schema == COMPILER_TARGET_LEGALITY_VERDICT_SCHEMA
    assert verdict.scope == COMPILER_TARGET_LEGALITY_SCOPE
    assert verdict.required_legacy_semantic_fingerprint == required.semantic_fingerprint
    assert (
        verdict.available_legacy_semantic_fingerprint == required.semantic_fingerprint
    )
    assert verdict.requirement_set_id == projection.requirement_set.requirement_set_id
    assert verdict.requires_legacy_comparator is True
    assert verdict.legacy_comparison_compatible is True
    assert verdict.legacy_comparison.differences == ()
    assert verdict.issue_paths == ()
    assert verdict.identity_valid is True
    assert verdict.to_dict()["identity"] == verdict.identity


def test_core_success_cannot_override_gate_parameter_legality_failure() -> None:
    """A provider's narrow RX domain is not legal for this static circuit."""

    required = _target()
    available = _target(
        native_gates=(
            GateCapability("cx"),
            GateCapability("rx", (ParameterConstraint("theta", -0.5, 0.5),)),
        )
    )
    projection = target_capabilities_to_requirement_set(required)
    core = match_target_capabilities(
        projection.requirement_set,
        _snapshot_for_projection(projection),
        evaluated_at=NOW,
    )
    verdict = evaluate_compiler_target_legality(projection, available)

    assert core.executable is True
    assert verdict.verdict is False
    assert verdict.legacy_comparison_compatible is False
    assert verdict.issue_paths == ("native_gates",)
    assert verdict.legacy_comparison.differences[0].field == "native_gates"
    assert verdict.legacy_comparison.diagnostics[0].notes[0].startswith("required=")


def test_core_success_cannot_override_ancilla_policy_or_deferred_failure() -> None:
    for required, available, expected_path in (
        (
            _target(ancilla_policy=AncillaPolicy.EXPLICIT_ONLY),
            _target(ancilla_policy=AncillaPolicy.UNSUPPORTED),
            "ancilla_policy",
        ),
        (
            _target(supports_noise=True),
            _target(supports_noise=False),
            "supports_noise",
        ),
    ):
        projection = target_capabilities_to_requirement_set(required)
        core = match_target_capabilities(
            projection.requirement_set,
            _snapshot_for_projection(projection),
            evaluated_at=NOW,
        )
        verdict = evaluate_compiler_target_legality(projection, available)
        assert core.executable is True
        assert verdict.verdict is False
        assert verdict.issue_paths == (expected_path,)
        loss = next(item for item in verdict.losses if item.field == expected_path)
        assert loss.active is True


def test_display_label_does_not_change_fingerprint_or_verdict_identity() -> None:
    required = _target()
    projection = target_capabilities_to_requirement_set(required)
    first = evaluate_compiler_target_legality(projection, _target(display_label="one"))
    second = evaluate_compiler_target_legality(projection, _target(display_label="two"))

    assert (
        first.available_legacy_semantic_fingerprint
        == second.available_legacy_semantic_fingerprint
    )
    assert first.identity == second.identity


def test_available_semantic_change_changes_identity_and_fails_closed() -> None:
    required = _target()
    projection = target_capabilities_to_requirement_set(required)
    baseline = evaluate_compiler_target_legality(projection, required)
    changed = evaluate_compiler_target_legality(
        projection, _target(maximum_program_operations=2048)
    )

    assert (
        changed.available_legacy_semantic_fingerprint
        != baseline.available_legacy_semantic_fingerprint
    )
    assert changed.identity != baseline.identity
    assert changed.verdict is False
    assert changed.issue_paths == ("maximum_program_operations",)


def test_comparator_issue_order_and_paths_are_preserved() -> None:
    required = _target(supports_noise=True)
    available = _target(
        logical_qubit_capacity=1,
        physical_qubit_capacity=2,
        native_gates=(GateCapability("cx"),),
        measurement_results=(MeasurementResult.STATE,),
        maximum_shots=1,
        maximum_program_operations=1,
        supports_noise=False,
    )
    verdict = evaluate_compiler_target_legality(
        target_capabilities_to_requirement_set(required), available
    )

    assert verdict.verdict is False
    assert verdict.issue_paths == (
        "logical_qubit_capacity",
        "physical_qubit_capacity",
        "native_gates",
        "measurement_results",
        "supports_noise",
        "maximum_shots",
        "maximum_program_operations",
    )
    assert len(verdict.legacy_comparison.diagnostics) == len(verdict.issue_paths)


def test_reordered_inputs_have_same_identity() -> None:
    required = _target(
        native_gates=(
            GateCapability("rx", (ParameterConstraint("theta", -1.0, 1.0),)),
            GateCapability("cx"),
        ),
        measurement_results=(MeasurementResult.STATE, MeasurementResult.EXPECTATION),
    )
    canonical_required = _target(
        native_gates=(
            GateCapability("cx"),
            GateCapability("rx", (ParameterConstraint("theta", -1.0, 1.0),)),
        ),
        measurement_results=(MeasurementResult.EXPECTATION, MeasurementResult.STATE),
    )
    available = _target(
        native_gates=(GateCapability("rx"), GateCapability("cx")),
        measurement_results=(MeasurementResult.EXPECTATION, MeasurementResult.STATE),
    )
    first = evaluate_compiler_target_legality(
        target_capabilities_to_requirement_set(required), available
    )
    second = evaluate_compiler_target_legality(
        target_capabilities_to_requirement_set(canonical_required),
        available,
    )

    assert first.to_dict() == second.to_dict()


def test_identity_is_stable_across_independent_hash_seeds() -> None:
    code = (
        "from flagquantum._compiler.target_capabilities import "
        "ArtifactFormat, ArtifactProfile, GateCapability, MeasurementResult, "
        "TargetCapabilities, TargetClass, AncillaPolicy; "
        "from flagquantum._compiler.target_capabilities_adapter import "
        "target_capabilities_to_requirement_set; "
        "from flagquantum._compiler.target_legality_verdict import "
        "evaluate_compiler_target_legality; "
        "t=TargetCapabilities(target_class=TargetClass.LOCAL_RUNTIME, "
        "logical_qubit_capacity=2, physical_qubit_capacity=4, "
        "native_gates=(GateCapability('cx'),GateCapability('rx')), "
        "measurement_results=(MeasurementResult.STATE,MeasurementResult.EXPECTATION), "
        "artifact_profiles=(ArtifactProfile(ArtifactFormat.RUNTIME_PLAN,'1.0'),), "
        "maximum_shots=1024, maximum_program_operations=4096, "
        "ancilla_policy=AncillaPolicy.EXPLICIT_ONLY); "
        "p=target_capabilities_to_requirement_set(t); "
        "print(evaluate_compiler_target_legality(p,t).identity)"
    )
    values = []
    for seed in ("1", "987654"):
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        values.append(result.stdout.strip())
    assert values[0] == values[1]


def test_tampered_attestation_fails_closed() -> None:
    verdict = evaluate_compiler_target_legality(
        target_capabilities_to_requirement_set(_target()), _target()
    )
    object.__setattr__(verdict, "verdict", False)

    assert verdict.identity_valid is False
    with pytest.raises(ValueError, match="identity is invalid"):
        verdict.require_valid()
    with pytest.raises(ValueError, match="identity is invalid"):
        verdict.to_dict()


def test_verdict_rejects_non_projection_inputs() -> None:
    with pytest.raises(TypeError, match="CompilerRequirementProjection"):
        evaluate_compiler_target_legality(object(), _target())  # type: ignore[arg-type]
