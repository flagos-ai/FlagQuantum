"""Canonical distributed audit data contracts."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedScalabilityAudit:
    """Result of auditing a distributed runtime summary or benchmark payload."""

    valid: bool
    scalability_claim_allowed: bool
    distribution_semantics: str
    claim_evidence_type: str
    release_gate_allowed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "scalability_claim_allowed": self.scalability_claim_allowed,
            "distribution_semantics": self.distribution_semantics,
            "claim_evidence_type": self.claim_evidence_type,
            "release_gate_allowed": self.release_gate_allowed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class StatevectorTrainingClaimabilityGate:
    """Machine-readable gate for distributed statevector training claims."""

    status: str
    claimable_production_training: bool
    errors: tuple[str, ...]
    blockers: tuple[str, ...]
    checks: dict[str, bool]

    def summary(self) -> dict[str, Any]:
        diagnostic_codes = tuple(
            f"statevector_training.{name}"
            for name, passed in self.checks.items()
            if not passed
        )
        return {
            "status": self.status,
            "claimable_production_training": self.claimable_production_training,
            "errors": self.errors,
            "blockers": self.blockers,
            "checks": dict(self.checks),
            "diagnostic_schema": "flagquantum.audit.diagnostics.v1",
            "diagnostic_codes": diagnostic_codes,
        }


@dataclass(frozen=True)
class MPSBackwardReadinessGate:
    """Machine-readable MPS backward-readiness evidence."""

    status: str
    production_training_claimable: bool
    fail_closed: bool
    contract_version: str
    claim_evidence_type: str
    errors: tuple[str, ...]
    blockers: tuple[str, ...]
    checks: dict[str, bool]
    evidence: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        diagnostic_codes = tuple(
            f"mps_backward_readiness.{name}"
            for name, passed in self.checks.items()
            if not passed
        )
        return {
            "status": self.status,
            "mps_backward_readiness_status": self.status,
            # Compatibility field for historical release payloads.
            "phase5_mps_backward_readiness_status": self.status,
            "production_training_claimable": self.production_training_claimable,
            "claimable_production_training": self.production_training_claimable,
            "fail_closed": self.fail_closed,
            "contract_version": self.contract_version,
            "claim_evidence_type": self.claim_evidence_type,
            "errors": self.errors,
            "blockers": self.blockers,
            "checks": dict(self.checks),
            "diagnostic_schema": "flagquantum.audit.diagnostics.v1",
            "diagnostic_codes": diagnostic_codes,
            **dict(self.evidence),
        }


@dataclass(frozen=True)
class DistributedTransportEvidence:
    """Normalized communication-route evidence for distributed payloads."""

    status: str
    node_count: int
    network_backend: str
    topology_scope: str
    route: Any
    communication_bytes_reported: bool
    rank_placement_reported: bool
    topology_dependent: bool
    blockers: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "node_count": self.node_count,
            "network_backend": self.network_backend,
            "topology_scope": self.topology_scope,
            "route": self.route,
            "communication_bytes_reported": self.communication_bytes_reported,
            "rank_placement_reported": self.rank_placement_reported,
            "topology_dependent": self.topology_dependent,
            "blockers": self.blockers,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class DistributedEvidenceContract:
    """Backend-neutral evidence contract for distributed claimability."""

    contract_version: str
    backend_family: str
    status: str
    claimable_production_training: bool
    claim_evidence_type: str
    distribution_semantics: str
    blockers: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checks: dict[str, bool]
    rank_ownership: Any
    memory_plan: dict[str, Any]
    communication_plan: dict[str, Any]
    transport_evidence: dict[str, Any]
    fail_closed: bool

    def summary(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "backend_family": self.backend_family,
            "status": self.status,
            "claimability_status": self.status,
            "claimable_production_training": self.claimable_production_training,
            "claim_evidence_type": self.claim_evidence_type,
            "distribution_semantics": self.distribution_semantics,
            "blockers": self.blockers,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": dict(self.checks),
            "rank_ownership": self.rank_ownership,
            "memory_plan": dict(self.memory_plan),
            "communication_plan": dict(self.communication_plan),
            "transport_evidence": dict(self.transport_evidence),
            "fail_closed": self.fail_closed,
        }


__all__ = [
    "DistributedScalabilityAudit",
    "StatevectorTrainingClaimabilityGate",
    "MPSBackwardReadinessGate",
    "DistributedTransportEvidence",
    "DistributedEvidenceContract",
]
