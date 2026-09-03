"""Private offline readiness evaluation for the legacy deployment bridge."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from enum import Enum

from .deployment_dry_run import DeploymentDryRunReport, DeploymentDryRunStatus
from .target_capabilities import ArtifactProfile

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"canary readiness {label} must be lowercase SHA-256")


def _require_safe_token(value: str, label: str) -> None:
    if not isinstance(value, str) or _SAFE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"canary readiness {label} must be an anonymous safe token")


class CanaryReadinessStatus(str, Enum):
    BLOCKED = "blocked"
    ELIGIBLE_FOR_OFFLINE_REHEARSAL = "eligible_for_offline_rehearsal"
    READY_FOR_SEPARATE_ACTIVATION_REVIEW = "ready_for_separate_activation_review"


class CanaryReadinessFinding(str, Enum):
    DRY_RUN_NOT_READY = "dry_run_not_ready"
    IDENTITY_CHAIN_INCOMPLETE = "identity_chain_incomplete"
    PROVIDER_CONFORMANCE_MISSING = "provider_conformance_missing"
    PROVIDER_CONFORMANCE_STALE = "provider_conformance_stale"
    PROVIDER_TARGET_MISMATCH = "provider_target_mismatch"
    OPT_IN_CONTRACT_MISSING = "opt_in_contract_missing"
    LEGACY_AUTHORITY_NOT_PRESERVED = "legacy_authority_not_preserved"
    KILL_SWITCH_MISSING = "kill_switch_missing"
    KILL_SWITCH_OWNER_MISSING = "kill_switch_owner_missing"
    ROLLBACK_PROOF_MISSING = "rollback_proof_missing"
    SAMPLING_LIMIT_MISSING = "sampling_limit_missing"
    CONCURRENCY_LIMIT_MISSING = "concurrency_limit_missing"
    BACKPRESSURE_CONTRACT_MISSING = "backpressure_contract_missing"
    CORRECTNESS_BUDGET_MISSING = "correctness_budget_missing"
    PRIVACY_BUDGET_MISSING = "privacy_budget_missing"
    LATENCY_BUDGET_MISSING = "latency_budget_missing"
    MEMORY_BUDGET_MISSING = "memory_budget_missing"
    QUEUE_BUDGET_MISSING = "queue_budget_missing"
    COST_BUDGET_MISSING = "cost_budget_missing"
    OBSERVABILITY_CONTRACT_MISSING = "observability_contract_missing"
    RETENTION_OR_DELETION_CONTRACT_MISSING = "retention_or_deletion_contract_missing"
    INCIDENT_OWNER_OR_RECOVERY_OBJECTIVE_MISSING = (
        "incident_owner_or_recovery_objective_missing"
    )
    DISPATCH_IDEMPOTENCY_CONTRACT_MISSING = "dispatch_idempotency_contract_missing"
    UNCERTAIN_SUBMISSION_CONTRACT_MISSING = "uncertain_submission_contract_missing"
    SEPARATE_ACTIVATION_APPROVAL_MISSING = "separate_activation_approval_missing"
    READY_FOR_SEPARATE_ACTIVATION_REVIEW = "ready_for_separate_activation_review"


class ConformanceAttestationState(str, Enum):
    VALID = "valid"
    STALE = "stale"
    REVOKED = "revoked"


@dataclass(frozen=True)
class ProviderConformanceAttestation:
    """Anonymous synthetic conformance evidence; never a provider connection."""

    passed: bool
    state: ConformanceAttestationState
    target_capability_fingerprint: str
    artifact_profile: ArtifactProfile
    adapter_family: str
    adapter_version: str
    conformance_suite_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("canary readiness conformance passed flag must be boolean")
        if not isinstance(self.state, ConformanceAttestationState):
            raise ValueError(
                "canary readiness conformance state must use its closed enum"
            )
        _require_digest(
            self.target_capability_fingerprint, "target capability fingerprint"
        )
        if not isinstance(self.artifact_profile, ArtifactProfile):
            raise ValueError("canary readiness profile must use ArtifactProfile")
        _require_safe_token(self.adapter_family, "adapter family")
        if not self.adapter_family.startswith("anonymous."):
            raise ValueError(
                "canary readiness adapter family must remain anonymous and synthetic"
            )
        _require_safe_token(self.adapter_version, "adapter version")
        _require_digest(self.conformance_suite_identity, "conformance suite identity")

    def identity_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "state": self.state.value,
            "target_capability_fingerprint": self.target_capability_fingerprint,
            "artifact_profile": self.artifact_profile.to_dict(),
            "adapter_family": self.adapter_family,
            "adapter_version": self.adapter_version,
            "conformance_suite_identity": self.conformance_suite_identity,
        }

    @property
    def attestation_identity(self) -> str:
        return _digest(self.identity_dict())


@dataclass(frozen=True)
class CanaryBudgetSnapshot:
    correctness: bool
    privacy: bool
    latency: bool
    memory: bool
    sampling: bool
    concurrency: bool
    queue: bool
    provider_quota: bool
    monetary_cost: bool

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, item.name), bool) for item in fields(self)):
            raise ValueError("canary readiness budget values must be boolean")

    def identity_dict(self) -> dict[str, object]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def budget_identity(self) -> str:
        return _digest(self.identity_dict())


@dataclass(frozen=True)
class CanaryControlSnapshot:
    explicit_opt_in_contract: bool
    legacy_authority_preserved: bool
    kill_switch_available: bool
    kill_switch_owner_assigned: bool
    rollback_proven: bool
    backpressure_contract: bool
    observability_contract: bool
    retention_deletion_contract: bool
    incident_owner_and_recovery_objective: bool
    dispatch_idempotency_contract: bool
    uncertain_submission_contract: bool
    separate_activation_approval: bool

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, item.name), bool) for item in fields(self)):
            raise ValueError("canary readiness control values must be boolean")

    def identity_dict(self) -> dict[str, object]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def control_identity(self) -> str:
        return _digest(self.identity_dict())


@dataclass(frozen=True)
class CanaryReadinessPolicy:
    maximum_evidence_bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_evidence_bytes, bool)
            or not isinstance(self.maximum_evidence_bytes, int)
            or self.maximum_evidence_bytes < 1024
        ):
            raise ValueError(
                "canary readiness maximum evidence bytes must be an integer >= 1024"
            )

    @property
    def policy_identity(self) -> str:
        return _digest({"maximum_evidence_bytes": self.maximum_evidence_bytes})


@dataclass(frozen=True)
class CanaryReadinessReport:
    status: CanaryReadinessStatus
    findings: tuple[CanaryReadinessFinding, ...]
    dry_run_evidence_identity: str
    target_capability_fingerprint: str
    artifact_profile: ArtifactProfile | None
    conformance_attestation_identity: str
    control_identity: str
    budget_identity: str
    policy_identity: str
    evidence_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, CanaryReadinessStatus):
            raise ValueError("canary readiness status must use its closed enum")
        findings = tuple(sorted(self.findings, key=lambda item: item.value))
        if any(not isinstance(item, CanaryReadinessFinding) for item in findings):
            raise ValueError("canary readiness findings must use their closed enum")
        if not findings or len(findings) != len(set(findings)):
            raise ValueError("canary readiness findings must be non-empty and unique")
        object.__setattr__(self, "findings", findings)
        for name in (
            "dry_run_evidence_identity",
            "target_capability_fingerprint",
            "conformance_attestation_identity",
            "control_identity",
            "budget_identity",
            "policy_identity",
            "evidence_identity",
        ):
            _require_digest(getattr(self, name), name.replace("_", " "))
        if self.artifact_profile is not None and not isinstance(
            self.artifact_profile, ArtifactProfile
        ):
            raise ValueError("canary readiness report profile must use ArtifactProfile")
        if self.evidence_identity != _digest(self.identity_dict()):
            raise ValueError(
                "canary readiness evidence identity does not match content"
            )

    def identity_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "findings": [item.value for item in self.findings],
            "dry_run_evidence_identity": self.dry_run_evidence_identity,
            "target_capability_fingerprint": self.target_capability_fingerprint,
            "artifact_profile": (
                None
                if self.artifact_profile is None
                else self.artifact_profile.to_dict()
            ),
            "conformance_attestation_identity": self.conformance_attestation_identity,
            "control_identity": self.control_identity,
            "budget_identity": self.budget_identity,
            "policy_identity": self.policy_identity,
        }

    @property
    def encoded_size(self) -> int:
        return len(
            json.dumps(
                {**self.identity_dict(), "evidence_identity": self.evidence_identity},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )


def _identity_chain_complete(report: DeploymentDryRunReport) -> bool:
    return all(
        value is not None
        for value in (
            report.compatibility_report_identity,
            report.source_program_identity,
            report.compilation_identity,
            report.target_program_identity,
            report.artifact_identity,
            report.payload_content_hash,
            report.artifact_profile,
        )
    )


def _control_findings(
    controls: CanaryControlSnapshot,
) -> set[CanaryReadinessFinding]:
    checks = {
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
        "separate_activation_approval": CanaryReadinessFinding.SEPARATE_ACTIVATION_APPROVAL_MISSING,
    }
    return {finding for name, finding in checks.items() if not getattr(controls, name)}


def _budget_findings(
    budgets: CanaryBudgetSnapshot,
) -> set[CanaryReadinessFinding]:
    findings = set()
    if not budgets.correctness:
        findings.add(CanaryReadinessFinding.CORRECTNESS_BUDGET_MISSING)
    if not budgets.privacy:
        findings.add(CanaryReadinessFinding.PRIVACY_BUDGET_MISSING)
    if not budgets.latency:
        findings.add(CanaryReadinessFinding.LATENCY_BUDGET_MISSING)
    if not budgets.memory:
        findings.add(CanaryReadinessFinding.MEMORY_BUDGET_MISSING)
    if not budgets.sampling:
        findings.add(CanaryReadinessFinding.SAMPLING_LIMIT_MISSING)
    if not budgets.concurrency:
        findings.add(CanaryReadinessFinding.CONCURRENCY_LIMIT_MISSING)
    if not budgets.queue:
        findings.add(CanaryReadinessFinding.QUEUE_BUDGET_MISSING)
    if not budgets.provider_quota or not budgets.monetary_cost:
        findings.add(CanaryReadinessFinding.COST_BUDGET_MISSING)
    return findings


def evaluate_deployment_canary_readiness(
    dry_run_report: DeploymentDryRunReport,
    conformance: ProviderConformanceAttestation,
    controls: CanaryControlSnapshot,
    budgets: CanaryBudgetSnapshot,
    policy: CanaryReadinessPolicy,
) -> CanaryReadinessReport:
    """Evaluate anonymous readiness evidence without activating or executing work."""

    if not isinstance(dry_run_report, DeploymentDryRunReport):
        raise ValueError("canary readiness requires a DeploymentDryRunReport")
    if not isinstance(conformance, ProviderConformanceAttestation):
        raise ValueError("canary readiness requires a conformance attestation")
    if not isinstance(controls, CanaryControlSnapshot):
        raise ValueError("canary readiness requires a control snapshot")
    if not isinstance(budgets, CanaryBudgetSnapshot):
        raise ValueError("canary readiness requires a budget snapshot")
    if not isinstance(policy, CanaryReadinessPolicy):
        raise ValueError("canary readiness requires an explicit policy")
    findings = _control_findings(controls) | _budget_findings(budgets)
    if dry_run_report.status is not DeploymentDryRunStatus.READY_FOR_OPERATOR_REVIEW:
        findings.add(CanaryReadinessFinding.DRY_RUN_NOT_READY)
    if not _identity_chain_complete(dry_run_report):
        findings.add(CanaryReadinessFinding.IDENTITY_CHAIN_INCOMPLETE)
    if not conformance.passed:
        findings.add(CanaryReadinessFinding.PROVIDER_CONFORMANCE_MISSING)
    if conformance.state is not ConformanceAttestationState.VALID:
        findings.add(CanaryReadinessFinding.PROVIDER_CONFORMANCE_STALE)
    if (
        conformance.target_capability_fingerprint
        != dry_run_report.target_capability_fingerprint
        or conformance.artifact_profile != dry_run_report.artifact_profile
    ):
        findings.add(CanaryReadinessFinding.PROVIDER_TARGET_MISMATCH)

    only_approval_missing = findings == {
        CanaryReadinessFinding.SEPARATE_ACTIVATION_APPROVAL_MISSING
    }
    if not findings:
        status = CanaryReadinessStatus.READY_FOR_SEPARATE_ACTIVATION_REVIEW
        findings.add(CanaryReadinessFinding.READY_FOR_SEPARATE_ACTIVATION_REVIEW)
    elif only_approval_missing:
        status = CanaryReadinessStatus.ELIGIBLE_FOR_OFFLINE_REHEARSAL
    else:
        status = CanaryReadinessStatus.BLOCKED

    values = {
        "status": status.value,
        "findings": sorted(item.value for item in findings),
        "dry_run_evidence_identity": dry_run_report.evidence_identity,
        "target_capability_fingerprint": dry_run_report.target_capability_fingerprint,
        "artifact_profile": (
            None
            if dry_run_report.artifact_profile is None
            else dry_run_report.artifact_profile.to_dict()
        ),
        "conformance_attestation_identity": conformance.attestation_identity,
        "control_identity": controls.control_identity,
        "budget_identity": budgets.budget_identity,
        "policy_identity": policy.policy_identity,
    }
    report = CanaryReadinessReport(
        status,
        tuple(findings),
        dry_run_report.evidence_identity,
        dry_run_report.target_capability_fingerprint,
        dry_run_report.artifact_profile,
        conformance.attestation_identity,
        controls.control_identity,
        budgets.budget_identity,
        policy.policy_identity,
        _digest(values),
    )
    if report.encoded_size > policy.maximum_evidence_bytes:
        raise ValueError("canary readiness evidence exceeds its explicit limit")
    return report
