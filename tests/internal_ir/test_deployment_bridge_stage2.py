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
    CompatibilityInspectionLimits,
)
from flagquantum._compiler.deployment_dry_run import (
    DeploymentDryRunFinding,
    DeploymentDryRunPolicy,
    DeploymentDryRunReport,
    DeploymentDryRunStage,
    DeploymentDryRunStatus,
    dry_run_deployment_bridge,
)
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
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
FIXTURE = ROOT / "tests/fixtures/internal_ir/deployment_bridge_stage2.json"


def _policy(**changes: int) -> DeploymentDryRunPolicy:
    values = {
        "maximum_source_operations": 1024,
        "maximum_compiled_operations": 8192,
        "maximum_artifact_bytes": 1048576,
        "maximum_evidence_bytes": 65536,
        "maximum_stage_duration_ns": 10_000_000_000,
        "maximum_total_duration_ns": 30_000_000_000,
    }
    values.update(changes)
    return DeploymentDryRunPolicy(
        CompatibilityInspectionLimits(1048576, 1024, 32, 1048576, 32),
        **values,
    )


def _profile(format_name: str) -> ArtifactProfile:
    profiles = {
        "openqasm_2": ArtifactProfile(ArtifactFormat.OPENQASM_2, "2.0"),
        "openqasm_3_static": ArtifactProfile(ArtifactFormat.OPENQASM_3_STATIC, "3.0"),
        "qcis_1": ArtifactProfile(ArtifactFormat.QCIS_1, "1.0"),
    }
    return profiles[format_name]


def _package(
    format_name: str,
    *,
    provider: str = "anonymous",
    name: str = "anonymous-job",
    shots: int = 100,
):
    backend = CloudBackendProfile(
        provider=provider,
        name="offline-simulator",
        n_wires=2,
        supports_openqasm=format_name != "qcis_1",
        supports_qcis=format_name == "qcis_1",
        is_simulator=True,
    )
    return create_deployment_package(
        fq.Circuit(2).rx(0, 0.25).cx(0, 1),
        backend=backend,
        name=name,
        shots=shots,
        qasm_version=3.0 if format_name == "openqasm_3_static" else 2.0,
    )


def _target(package, format_name: str, **changes: object) -> TargetCapabilities:
    values = {
        "target_class": (
            TargetClass.NON_QASM_ARTIFACT
            if format_name == "qcis_1"
            else TargetClass.QASM_TEXT
        ),
        "logical_qubit_capacity": 2,
        "physical_qubit_capacity": 2,
        "native_gates": tuple(
            GateCapability(name)
            for name in sorted({item.name for item in package.ir.instructions})
        ),
        "measurement_results": (MeasurementResult.COUNTS,),
        "artifact_profiles": (_profile(format_name),),
        "topology": DirectedCouplingGraph(2, ((0, 1), (1, 0))),
        "maximum_shots": 4096,
        "maximum_program_operations": 8192,
    }
    values.update(changes)
    return TargetCapabilities(**values)


def _records() -> list[dict[str, str]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["positive_cases"]


def _run(record: dict[str, str]) -> DeploymentDryRunReport:
    package = _package(record["format"])
    return dry_run_deployment_bridge(
        package,
        _target(package, record["format"]),
        _policy(),
    )


def test_three_profiles_complete_the_ephemeral_identity_chain() -> None:
    for record in _records():
        report = _run(record)
        assert report.status is DeploymentDryRunStatus.READY_FOR_OPERATOR_REVIEW
        assert report.findings == (DeploymentDryRunFinding.READY_FOR_OPERATOR_REVIEW,)
        assert report.artifact_profile == _profile(record["format"])
        assert report.artifact_bytes > 0
        assert report.requested_shots == 100
        assert report.source_program_identity is not None
        assert report.compilation_identity is not None
        assert report.target_program_identity is not None
        assert report.artifact_identity is not None
        assert report.payload_content_hash is not None
        assert report.evidence_identity == record["expected_evidence_identity"]
        assert report.artifact_identity == record["expected_artifact_identity"]
        assert tuple(item.stage for item in report.stage_timings_ns) == tuple(
            DeploymentDryRunStage
        )


def test_report_is_immutable_privacy_safe_and_contains_no_candidate_objects() -> None:
    package = _package(
        "openqasm_2",
        provider="private-provider-label",
        name="private-package-label",
    )
    report = dry_run_deployment_bridge(
        package, _target(package, "openqasm_2"), _policy()
    )

    assert {item.name for item in fields(DeploymentDryRunReport)}.isdisjoint(
        {"program", "qasm", "qcis", "ir", "payload", "artifact", "metadata"}
    )
    assert package.qasm not in repr(report)
    assert "private-provider-label" not in repr(report)
    assert "private-package-label" not in repr(report)
    with pytest.raises(FrozenInstanceError):
        report.status = DeploymentDryRunStatus.INTERNAL_FAILURE


def test_dry_run_does_not_mutate_package_target_or_policy() -> None:
    package = _package("openqasm_2")
    target = _target(package, "openqasm_2")
    policy = _policy()
    before = (deepcopy(package), deepcopy(target), deepcopy(policy))

    dry_run_deployment_bridge(package, target, policy)

    assert package == before[0]
    assert target == before[1]
    assert policy == before[2]


def test_provider_and_package_labels_are_nonsemantic_but_shots_are_execution_intent() -> (
    None
):
    first = _package("openqasm_2", provider="anonymous-a", name="label-a", shots=100)
    second = _package("openqasm_2", provider="anonymous-b", name="label-b", shots=100)
    changed_shots = _package(
        "openqasm_2", provider="anonymous-a", name="label-a", shots=200
    )
    target = _target(first, "openqasm_2")
    first_report = dry_run_deployment_bridge(first, target, _policy())
    second_report = dry_run_deployment_bridge(second, target, _policy())
    shots_report = dry_run_deployment_bridge(changed_shots, target, _policy())

    assert first_report.legacy_artifact_digest != second_report.legacy_artifact_digest
    assert first_report.evidence_identity == second_report.evidence_identity
    assert first_report.artifact_identity == second_report.artifact_identity
    assert first_report.artifact_identity == shots_report.artifact_identity
    assert first_report.evidence_identity != shots_report.evidence_identity


def test_program_target_and_calibration_changes_are_semantic() -> None:
    package = _package("openqasm_2")
    target = _target(package, "openqasm_2")
    original = dry_run_deployment_bridge(package, target, _policy())
    changed_package = create_deployment_package(
        fq.Circuit(2).ry(0, 0.5).cx(0, 1),
        backend=package.backend,
        name=package.name,
        shots=package.shots,
    )
    changed_program = dry_run_deployment_bridge(
        changed_package, _target(changed_package, "openqasm_2"), _policy()
    )
    changed_target = replace(target, calibration_snapshot_hash="c" * 64)
    changed_calibration = dry_run_deployment_bridge(package, changed_target, _policy())

    assert changed_program.artifact_identity != original.artifact_identity
    assert changed_program.evidence_identity != original.evidence_identity
    assert changed_calibration.compilation_identity != original.compilation_identity
    assert changed_calibration.artifact_identity != original.artifact_identity
    assert changed_calibration.evidence_identity != original.evidence_identity


def test_sensitive_metadata_and_incomplete_target_fail_closed() -> None:
    package = _package("openqasm_2")
    package.metadata["token"] = "do-not-leak"
    rejected = dry_run_deployment_bridge(
        package, _target(package, "openqasm_2"), _policy()
    )
    assert rejected.status is DeploymentDryRunStatus.COMPATIBILITY_REJECTED
    assert DeploymentDryRunFinding.COMPATIBILITY_NOT_ELIGIBLE in rejected.findings
    assert "do-not-leak" not in repr(rejected)

    clean = _package("openqasm_2")
    incomplete = _target(clean, "openqasm_2", topology=None)
    compilation = dry_run_deployment_bridge(clean, incomplete, _policy())
    assert compilation.status is DeploymentDryRunStatus.COMPILATION_REJECTED
    assert DeploymentDryRunFinding.OFFLINE_COMPILATION_FAILED in compilation.findings


def test_operation_artifact_evidence_and_deadline_limits_fail_closed() -> None:
    package = _package("openqasm_2")
    target = _target(package, "openqasm_2")

    source = dry_run_deployment_bridge(
        package, target, _policy(maximum_source_operations=1)
    )
    assert source.status is DeploymentDryRunStatus.LIMIT_EXCEEDED
    assert DeploymentDryRunFinding.OPERATION_LIMIT_EXCEEDED in source.findings

    compiled = dry_run_deployment_bridge(
        package, target, _policy(maximum_compiled_operations=1)
    )
    assert compiled.status is DeploymentDryRunStatus.LIMIT_EXCEEDED
    assert DeploymentDryRunFinding.OPERATION_LIMIT_EXCEEDED in compiled.findings

    artifact = dry_run_deployment_bridge(
        package, target, _policy(maximum_artifact_bytes=1)
    )
    assert artifact.status is DeploymentDryRunStatus.LIMIT_EXCEEDED
    assert DeploymentDryRunFinding.ARTIFACT_SIZE_LIMIT_EXCEEDED in artifact.findings

    evidence = dry_run_deployment_bridge(
        package, target, _policy(maximum_evidence_bytes=1)
    )
    assert evidence.status is DeploymentDryRunStatus.LIMIT_EXCEEDED
    assert DeploymentDryRunFinding.EVIDENCE_LIMIT_EXCEEDED in evidence.findings

    deadline = dry_run_deployment_bridge(
        package, target, _policy(maximum_stage_duration_ns=1)
    )
    assert deadline.status is DeploymentDryRunStatus.LIMIT_EXCEEDED
    assert DeploymentDryRunFinding.DEADLINE_EXCEEDED in deadline.findings


def test_evidence_identities_are_stable_across_python_hash_seeds() -> None:
    script = f"""
import json
import runpy
namespace = runpy.run_path({str(Path(__file__))!r})
print(json.dumps([
    namespace['_run'](record).evidence_identity
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

    expected = [item["expected_evidence_identity"] for item in _records()]
    assert identities("1") == identities("8675309") == expected


def test_dry_run_has_no_activation_execution_submission_or_public_export() -> None:
    import flagquantum._compiler.deployment_dry_run as module

    source = inspect.getsource(module)
    assert "os.environ" not in source
    assert "getenv(" not in source
    assert "requests" not in source
    assert "create_execution_binding" not in source
    assert "submit(" not in source
    assert "fq.run" not in source
    for name in (
        "DeploymentDryRunPolicy",
        "DeploymentDryRunReport",
        "dry_run_deployment_bridge",
    ):
        assert not hasattr(fq, name)
