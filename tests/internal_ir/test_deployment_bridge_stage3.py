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
from flagquantum._compiler.deployment_canary_readiness import (
    CanaryBudgetSnapshot,
    CanaryControlSnapshot,
    CanaryReadinessFinding,
    CanaryReadinessPolicy,
    CanaryReadinessReport,
    CanaryReadinessStatus,
    ConformanceAttestationState,
    ProviderConformanceAttestation,
    evaluate_deployment_canary_readiness,
)
from flagquantum._compiler.deployment_compatibility import (
    CompatibilityInspectionLimits,
)
from flagquantum._compiler.deployment_dry_run import (
    DeploymentDryRunPolicy,
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
from flagquantum.deployment.cloud import CloudBackendProfile, create_deployment_package
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/internal_ir/deployment_bridge_stage3.json"
PROPOSAL = ROOT / "contracts/deployment-bridge-stage3-entry-proposal.json"


def _dry_run_report():
    circuit = fq.Circuit(2).rx(0, 0.25).cx(0, 1)
    package = create_deployment_package(
        circuit,
        backend=CloudBackendProfile(
            provider="anonymous",
            name="offline-simulator",
            n_wires=2,
            supports_openqasm=True,
            is_simulator=True,
        ),
        name="anonymous-job",
        shots=100,
    )
    profile = ArtifactProfile(ArtifactFormat.OPENQASM_2, "2.0")
    target = TargetCapabilities(
        target_class=TargetClass.QASM_TEXT,
        logical_qubit_capacity=2,
        physical_qubit_capacity=2,
        native_gates=(GateCapability("cx"), GateCapability("rx")),
        measurement_results=(MeasurementResult.COUNTS,),
        artifact_profiles=(profile,),
        topology=DirectedCouplingGraph(2, ((0, 1), (1, 0))),
        maximum_shots=4096,
        maximum_program_operations=8192,
    )
    policy = DeploymentDryRunPolicy(
        CompatibilityInspectionLimits(1048576, 1024, 32, 1048576, 32),
        maximum_source_operations=1024,
        maximum_compiled_operations=8192,
        maximum_artifact_bytes=1048576,
        maximum_evidence_bytes=65536,
        maximum_stage_duration_ns=10_000_000_000,
        maximum_total_duration_ns=30_000_000_000,
    )
    report = dry_run_deployment_bridge(package, target, policy)
    assert report.status is DeploymentDryRunStatus.READY_FOR_OPERATOR_REVIEW
    return report


def _conformance(report, **changes: object) -> ProviderConformanceAttestation:
    values = {
        "passed": True,
        "state": ConformanceAttestationState.VALID,
        "target_capability_fingerprint": report.target_capability_fingerprint,
        "artifact_profile": report.artifact_profile,
        "adapter_family": "anonymous.synthetic",
        "adapter_version": "v1",
        "conformance_suite_identity": "c" * 64,
    }
    values.update(changes)
    return ProviderConformanceAttestation(**values)


def _controls(**changes: bool) -> CanaryControlSnapshot:
    values = {item.name: True for item in fields(CanaryControlSnapshot)}
    values.update(changes)
    return CanaryControlSnapshot(**values)


def _budgets(**changes: bool) -> CanaryBudgetSnapshot:
    values = {item.name: True for item in fields(CanaryBudgetSnapshot)}
    values.update(changes)
    return CanaryBudgetSnapshot(**values)


def _evaluate(*, activation: bool = True) -> CanaryReadinessReport:
    report = _dry_run_report()
    return evaluate_deployment_canary_readiness(
        report,
        _conformance(report),
        _controls(separate_activation_approval=activation),
        _budgets(),
        CanaryReadinessPolicy(65536),
    )


def test_runtime_contract_exactly_matches_approved_closed_proposal() -> None:
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))

    assert {item.value for item in CanaryReadinessStatus} == set(
        proposal["readiness_status_taxonomy"]
    )
    assert {item.value for item in CanaryReadinessFinding} == set(
        proposal["finding_taxonomy"]
    )


def test_complete_anonymous_evidence_only_reaches_separate_review() -> None:
    report = _evaluate()
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert report.status is CanaryReadinessStatus.READY_FOR_SEPARATE_ACTIVATION_REVIEW
    assert report.findings == (
        CanaryReadinessFinding.READY_FOR_SEPARATE_ACTIVATION_REVIEW,
    )
    assert report.evidence_identity == expected["ready_evidence_identity"]
    assert report.encoded_size <= 65536


def test_missing_separate_approval_is_offline_rehearsal_not_activation() -> None:
    report = _evaluate(activation=False)
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert report.status is CanaryReadinessStatus.ELIGIBLE_FOR_OFFLINE_REHEARSAL
    assert report.findings == (
        CanaryReadinessFinding.SEPARATE_ACTIVATION_APPROVAL_MISSING,
    )
    assert report.evidence_identity == expected["rehearsal_evidence_identity"]


def test_every_control_and_budget_requirement_fails_closed() -> None:
    dry_run = _dry_run_report()
    conformance = _conformance(dry_run)
    policy = CanaryReadinessPolicy(65536)
    control_findings = {
        "explicit_opt_in_contract": CanaryReadinessFinding.OPT_IN_CONTRACT_MISSING,
        "legacy_authority_preserved": CanaryReadinessFinding.LEGACY_AUTHORITY_NOT_PRESERVED,
        "kill_switch_available": CanaryReadinessFinding.KILL_SWITCH_MISSING,
        "kill_switch_owner_assigned": CanaryReadinessFinding.KILL_SWITCH_OWNER_MISSING,
        "rollback_proven": CanaryReadinessFinding.ROLLBACK_PROOF_MISSING,
        "backpressure_contract": CanaryReadinessFinding.BACKPRESSURE_CONTRACT_MISSING,
        "observability_contract": CanaryReadinessFinding.OBSERVABILITY_CONTRACT_MISSING,
        "retention_deletion_contract": CanaryReadinessFinding.RETENTION_OR_DELETION_CONTRACT_MISSING,
        "incident_owner_and_recovery_objective": CanaryReadinessFinding.INCIDENT_OWNER_OR_RECOVERY_OBJECTIVE_MISSING,
        "dispatch_idempotency_contract": CanaryReadinessFinding.DISPATCH_IDEMPOTENCY_CONTRACT_MISSING,
        "uncertain_submission_contract": CanaryReadinessFinding.UNCERTAIN_SUBMISSION_CONTRACT_MISSING,
    }
    for name, finding in control_findings.items():
        result = evaluate_deployment_canary_readiness(
            dry_run, conformance, _controls(**{name: False}), _budgets(), policy
        )
        assert result.status is CanaryReadinessStatus.BLOCKED
        assert finding in result.findings

    budget_findings = {
        "correctness": CanaryReadinessFinding.CORRECTNESS_BUDGET_MISSING,
        "privacy": CanaryReadinessFinding.PRIVACY_BUDGET_MISSING,
        "latency": CanaryReadinessFinding.LATENCY_BUDGET_MISSING,
        "memory": CanaryReadinessFinding.MEMORY_BUDGET_MISSING,
        "sampling": CanaryReadinessFinding.SAMPLING_LIMIT_MISSING,
        "concurrency": CanaryReadinessFinding.CONCURRENCY_LIMIT_MISSING,
        "queue": CanaryReadinessFinding.QUEUE_BUDGET_MISSING,
        "provider_quota": CanaryReadinessFinding.COST_BUDGET_MISSING,
        "monetary_cost": CanaryReadinessFinding.COST_BUDGET_MISSING,
    }
    for name, finding in budget_findings.items():
        result = evaluate_deployment_canary_readiness(
            dry_run, conformance, _controls(), _budgets(**{name: False}), policy
        )
        assert result.status is CanaryReadinessStatus.BLOCKED
        assert finding in result.findings


def test_conformance_missing_stale_revoked_and_target_mismatch_fail_closed() -> None:
    dry_run = _dry_run_report()
    controls = _controls()
    budgets = _budgets()
    policy = CanaryReadinessPolicy(65536)

    cases = (
        (
            _conformance(dry_run, passed=False),
            CanaryReadinessFinding.PROVIDER_CONFORMANCE_MISSING,
        ),
        (
            _conformance(dry_run, state=ConformanceAttestationState.STALE),
            CanaryReadinessFinding.PROVIDER_CONFORMANCE_STALE,
        ),
        (
            _conformance(dry_run, state=ConformanceAttestationState.REVOKED),
            CanaryReadinessFinding.PROVIDER_CONFORMANCE_STALE,
        ),
        (
            _conformance(dry_run, target_capability_fingerprint="d" * 64),
            CanaryReadinessFinding.PROVIDER_TARGET_MISMATCH,
        ),
    )
    for conformance, finding in cases:
        result = evaluate_deployment_canary_readiness(
            dry_run, conformance, controls, budgets, policy
        )
        assert result.status is CanaryReadinessStatus.BLOCKED
        assert finding in result.findings


def test_inputs_and_report_are_immutable_and_contain_no_executable_state() -> None:
    dry_run = _dry_run_report()
    conformance = _conformance(dry_run)
    controls = _controls()
    budgets = _budgets()
    policy = CanaryReadinessPolicy(65536)
    before = deepcopy((dry_run, conformance, controls, budgets, policy))
    report = evaluate_deployment_canary_readiness(
        dry_run, conformance, controls, budgets, policy
    )

    assert (dry_run, conformance, controls, budgets, policy) == before
    assert {item.name for item in fields(CanaryReadinessReport)}.isdisjoint(
        {
            "adapter",
            "artifact",
            "binding",
            "credential",
            "endpoint",
            "job_id",
            "metadata",
            "payload",
            "program",
            "result",
        }
    )
    assert "anonymous.synthetic" not in repr(report)
    with pytest.raises(FrozenInstanceError):
        report.status = CanaryReadinessStatus.BLOCKED


def test_private_evaluator_has_no_runtime_provider_network_or_public_path() -> None:
    source = inspect.getsource(
        sys.modules[evaluate_deployment_canary_readiness.__module__]
    )

    for forbidden in (
        "from .runtime_adapters import",
        "from .provider_conformance import",
        "from .shadow_harness import",
        "requests",
        "urllib",
        ".submit(",
        ".execute(",
    ):
        assert forbidden not in source
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "evaluate_deployment_canary_readiness")


def test_evidence_identities_are_stable_across_python_hash_seeds() -> None:
    script = f"""
import json
import runpy
namespace = runpy.run_path({str(Path(__file__))!r})
print(json.dumps([
    namespace['_evaluate']().evidence_identity,
    namespace['_evaluate'](activation=False).evidence_identity,
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

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    golden = [
        expected["ready_evidence_identity"],
        expected["rehearsal_evidence_identity"],
    ]
    assert identities("1") == identities("8675309") == golden


def test_policy_and_snapshot_types_reject_ambiguous_values() -> None:
    with pytest.raises(ValueError, match=">= 1024"):
        CanaryReadinessPolicy(100)
    with pytest.raises(ValueError, match="must be boolean"):
        replace(_budgets(), correctness=1)
    with pytest.raises(ValueError, match="must be boolean"):
        replace(_controls(), kill_switch_available=1)
    with pytest.raises(ValueError, match="anonymous safe token"):
        replace(_conformance(_dry_run_report()), adapter_family="https://provider")
    with pytest.raises(ValueError, match="anonymous and synthetic"):
        replace(_conformance(_dry_run_report()), adapter_family="provider.real")
