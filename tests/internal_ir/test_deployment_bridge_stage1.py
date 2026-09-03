from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler.deployment_compatibility import (
    CompatibilityFinding,
    CompatibilityInspectionLimits,
    CompatibilityReport,
    CompatibilityStatus,
    inspect_deployment_compatibility,
)
from flagquantum._compiler.target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    TargetCapabilities,
    TargetClass,
)
from flagquantum.deployment.cloud import (
    CloudBackendProfile,
    create_deployment_package,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/internal_ir/deployment_bridge_stage1.json"


def _limits(**changes: int) -> CompatibilityInspectionLimits:
    values = {
        "maximum_program_bytes": 65536,
        "maximum_metadata_entries": 256,
        "maximum_metadata_depth": 16,
        "maximum_metadata_encoded_bytes": 262144,
        "maximum_findings": 16,
    }
    values.update(changes)
    return CompatibilityInspectionLimits(**values)


def _profile(format_name: str) -> ArtifactProfile:
    values = {
        "openqasm_2": (ArtifactFormat.OPENQASM_2, "2.0"),
        "openqasm_3_static": (ArtifactFormat.OPENQASM_3_STATIC, "3.0"),
        "qcis_1": (ArtifactFormat.QCIS_1, "1.0"),
    }
    artifact_format, version = values[format_name]
    return ArtifactProfile(artifact_format, version)


def _package(format_name: str, *, provider: str = "anonymous"):
    circuit = fq.Circuit(2).rx(0, 0.25).cx(0, 1)
    if format_name == "qcis_1":
        backend = CloudBackendProfile(
            provider=provider,
            name="offline-simulator",
            n_wires=2,
            supports_openqasm=False,
            supports_qcis=True,
            is_simulator=True,
        )
        version = 2.0
    else:
        backend = CloudBackendProfile(
            provider=provider,
            name="offline-simulator",
            n_wires=2,
            supports_openqasm=True,
            is_simulator=True,
        )
        version = 2.0 if format_name == "openqasm_2" else 3.0
    return create_deployment_package(
        circuit,
        backend=backend,
        name="anonymous-job",
        shots=100,
        qasm_version=version,
    )


def _target(package, format_name: str, **changes: object) -> TargetCapabilities:
    target_class = (
        TargetClass.NON_QASM_ARTIFACT
        if format_name == "qcis_1"
        else TargetClass.QASM_TEXT
    )
    values = {
        "target_class": target_class,
        "logical_qubit_capacity": 2,
        "physical_qubit_capacity": 2,
        "native_gates": tuple(
            GateCapability(name)
            for name in sorted({item.name for item in package.ir.instructions})
        ),
        "measurement_results": (MeasurementResult.COUNTS,),
        "artifact_profiles": (_profile(format_name),),
        "maximum_shots": 4096,
        "maximum_program_operations": 1024,
    }
    values.update(changes)
    return TargetCapabilities(**values)


def _records() -> list[dict[str, str]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["positive_cases"]


def _run(record: dict[str, str]) -> CompatibilityReport:
    package = _package(record["format"])
    return inspect_deployment_compatibility(
        package, _target(package, record["format"]), _limits()
    )


def test_positive_profiles_are_eligible_only_for_verified_recompile() -> None:
    for record in _records():
        report = _run(record)
        assert report.status is CompatibilityStatus.ELIGIBLE_FOR_VERIFIED_RECOMPILE
        assert report.findings == (
            CompatibilityFinding.ELIGIBLE_FOR_VERIFIED_RECOMPILE,
            CompatibilityFinding.SOURCE_IR_REQUIRES_VERIFIED_IMPORT,
        )
        assert report.report_identity == record["expected_report_identity"]


def test_report_is_immutable_bounded_and_contains_no_raw_or_locator_fields() -> None:
    package = _package("openqasm_2", provider="private-provider-label")
    report = inspect_deployment_compatibility(
        package, _target(package, "openqasm_2"), _limits()
    )

    assert {item.name for item in fields(CompatibilityReport)}.isdisjoint(
        {"program", "qasm", "qcis", "ir", "metadata", "provider", "backend"}
    )
    assert package.qasm not in repr(report)
    assert "private-provider-label" not in repr(report)
    with pytest.raises(FrozenInstanceError):
        report.status = CompatibilityStatus.INVALID


def test_inspection_does_not_mutate_package_target_or_nested_metadata() -> None:
    package = _package("openqasm_2")
    target = _target(package, "openqasm_2")
    package_before = deepcopy(package)
    target_before = deepcopy(target)

    inspect_deployment_compatibility(package, target, _limits())

    assert package == package_before
    assert target == target_before


def test_schema_identity_routing_and_sensitive_metadata_fail_closed() -> None:
    package = _package("openqasm_2")
    target = _target(package, "openqasm_2")
    bad_schema = replace(
        package,
        metadata=dict(package.metadata) | {"deployment_package_schema": "unknown"},
    )
    schema_report = inspect_deployment_compatibility(bad_schema, target, _limits())
    assert schema_report.status is CompatibilityStatus.INVALID
    assert CompatibilityFinding.PACKAGE_SCHEMA_INVALID in schema_report.findings

    bad_identity = replace(
        package,
        metadata=dict(package.metadata) | {"deployment_artifact_sha256": "0" * 64},
    )
    identity_report = inspect_deployment_compatibility(bad_identity, target, _limits())
    assert CompatibilityFinding.PACKAGE_IDENTITY_INVALID in identity_report.findings

    routing = deepcopy(package.metadata)
    routing["routing_evidence"] = dict(routing["routing_evidence"]) | {
        "routing_reused": "yes"
    }
    bad_routing = replace(package, metadata=routing)
    routing_report = inspect_deployment_compatibility(bad_routing, target, _limits())
    assert CompatibilityFinding.ROUTING_EVIDENCE_INVALID in routing_report.findings

    private_package = _package("openqasm_2")
    private_package.metadata["api_key"] = "do-not-leak"
    private_report = inspect_deployment_compatibility(
        private_package, _target(private_package, "openqasm_2"), _limits()
    )
    assert private_report.status is CompatibilityStatus.INVALID
    assert CompatibilityFinding.PROVIDER_METADATA_REJECTED in private_report.findings
    assert "do-not-leak" not in repr(private_report)

    malformed = replace(package, metadata=object(), shots=-1)
    malformed_report = inspect_deployment_compatibility(malformed, target, _limits())
    assert malformed_report.status is CompatibilityStatus.INVALID
    assert CompatibilityFinding.PROVIDER_METADATA_REJECTED in malformed_report.findings
    assert CompatibilityFinding.PACKAGE_IDENTITY_INVALID in malformed_report.findings


def test_profile_capacity_shots_and_dynamic_features_are_unsupported() -> None:
    package = _package("openqasm_2")
    mismatched = _target(
        package,
        "openqasm_2",
        artifact_profiles=(ArtifactProfile(ArtifactFormat.OPENQASM_3_STATIC, "3.0"),),
    )
    assert (
        inspect_deployment_compatibility(package, mismatched, _limits()).status
        is CompatibilityStatus.UNSUPPORTED
    )

    capacity = _target(
        package,
        "openqasm_2",
        logical_qubit_capacity=1,
        physical_qubit_capacity=1,
    )
    capacity_report = inspect_deployment_compatibility(package, capacity, _limits())
    assert CompatibilityFinding.TARGET_CAPACITY_EXCEEDED in capacity_report.findings

    shots = _target(package, "openqasm_2", maximum_shots=10)
    shots_report = inspect_deployment_compatibility(package, shots, _limits())
    assert CompatibilityFinding.SHOTS_LIMIT_EXCEEDED in shots_report.findings

    dynamic = replace(
        package, metadata=dict(package.metadata) | {"dynamic_circuit": True}
    )
    dynamic_report = inspect_deployment_compatibility(
        dynamic, target=_target(package, "openqasm_2"), limits=_limits()
    )
    assert CompatibilityFinding.DYNAMIC_PROGRAM_UNSUPPORTED in dynamic_report.findings


def test_hardware_target_without_calibration_requires_enrichment() -> None:
    package = _package("openqasm_2")
    hardware_backend = replace(package.backend, is_simulator=False)
    hardware_package = replace(package, backend=hardware_backend)
    metadata = dict(hardware_package.metadata)
    metadata["deployment_artifact_sha256"] = package.metadata[
        "deployment_artifact_sha256"
    ]
    hardware_package = replace(hardware_package, metadata=metadata)
    report = inspect_deployment_compatibility(
        hardware_package,
        _target(hardware_package, "openqasm_2"),
        _limits(),
    )

    assert report.status is CompatibilityStatus.REQUIRES_TARGET_ENRICHMENT
    assert CompatibilityFinding.TARGET_SNAPSHOT_INCOMPLETE in report.findings


def test_program_metadata_and_finding_limits_fail_closed() -> None:
    package = _package("openqasm_2")
    target = _target(package, "openqasm_2")

    program = inspect_deployment_compatibility(
        package, target, _limits(maximum_program_bytes=1)
    )
    assert program.status is CompatibilityStatus.LIMIT_EXCEEDED
    assert CompatibilityFinding.INPUT_SIZE_LIMIT_EXCEEDED in program.findings

    metadata = inspect_deployment_compatibility(
        package, target, _limits(maximum_metadata_entries=1)
    )
    assert metadata.status is CompatibilityStatus.LIMIT_EXCEEDED
    assert CompatibilityFinding.METADATA_LIMIT_EXCEEDED in metadata.findings

    findings = inspect_deployment_compatibility(
        package, target, _limits(maximum_findings=1)
    )
    assert findings.status is CompatibilityStatus.LIMIT_EXCEEDED
    assert len(findings.findings) == 1


def test_extreme_metadata_depth_is_cut_off_without_recursive_traversal() -> None:
    package = _package("openqasm_2")
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(2000):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    package.metadata["untrusted"] = nested

    report = inspect_deployment_compatibility(
        package,
        _target(package, "openqasm_2"),
        _limits(maximum_metadata_depth=8),
    )

    assert report.status is CompatibilityStatus.LIMIT_EXCEEDED
    assert CompatibilityFinding.METADATA_LIMIT_EXCEEDED in report.findings


def test_provider_labels_are_nonsemantic_but_program_and_target_changes_are_semantic() -> (
    None
):
    first = _package("openqasm_2", provider="anonymous-a")
    second = _package("openqasm_2", provider="anonymous-b")
    target = _target(first, "openqasm_2")
    first_report = inspect_deployment_compatibility(first, target, _limits())
    second_report = inspect_deployment_compatibility(second, target, _limits())
    assert first_report.legacy_artifact_digest != second_report.legacy_artifact_digest
    assert first_report.report_identity == second_report.report_identity

    changed = create_deployment_package(
        fq.Circuit(2).ry(0, 0.5).cx(0, 1),
        backend=first.backend,
        name="anonymous-job",
        shots=100,
    )
    changed_target = _target(changed, "openqasm_2")
    changed_report = inspect_deployment_compatibility(
        changed, changed_target, _limits()
    )
    assert changed_report.report_identity != first_report.report_identity
    wider_target = _target(first, "openqasm_2", maximum_shots=8192)
    assert (
        inspect_deployment_compatibility(first, wider_target, _limits()).report_identity
        != first_report.report_identity
    )


def test_report_identities_are_stable_across_python_hash_seeds() -> None:
    script = f"""
import json
import runpy
namespace = runpy.run_path({str(Path(__file__))!r})
print(json.dumps([
    namespace['_run'](record).report_identity
    for record in namespace['_records']()
]))
"""

    def identities(seed: str) -> list[str]:
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    expected = [item["expected_report_identity"] for item in _records()]
    assert identities("1") == identities("8675309") == expected


def test_checker_has_no_implicit_activation_conversion_or_public_export() -> None:
    import flagquantum._compiler.deployment_compatibility as module

    source = inspect.getsource(module)
    assert "os.environ" not in source
    assert "getenv(" not in source
    assert "requests" not in source
    assert "submit(" not in source
    assert "seal_executable_artifact" not in source
    for name in (
        "CompatibilityReport",
        "CompatibilityInspectionLimits",
        "inspect_deployment_compatibility",
    ):
        assert not hasattr(fq, name)
