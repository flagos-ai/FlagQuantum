"""Distributed evidence contracts and release gates."""

from .engine import (
    evaluate_distributed_evidence_contract,
    evaluate_distributed_transport_evidence,
    evaluate_statevector_training_claimability,
)
from .errors import DistributedScalabilityError
from .mps_readiness import (
    _attach_mps_runtime_summary as attach_mps_runtime_summary,
)
from .mps_readiness import evaluate_mps_backward_readiness
from .release_policy import audit_distributed_scalability
from .schema import (
    DistributedEvidenceContract,
    DistributedScalabilityAudit,
    DistributedTransportEvidence,
    MPSBackwardReadinessGate,
    StatevectorTrainingClaimabilityGate,
)
from .statistics import attach_distributed_evidence_contract
from .vocabulary import (
    CLAIM_EVIDENCE_TYPES,
    CLAIMABILITY_STATUSES,
    MPS_BACKWARD_READINESS_STATUSES,
    SCALABLE_DISTRIBUTION_SEMANTICS,
    STATEVECTOR_TRAINING_CLAIMABILITY_STATUSES,
)

__all__ = (
    "DistributedEvidenceContract",
    "DistributedScalabilityAudit",
    "DistributedScalabilityError",
    "DistributedTransportEvidence",
    "MPSBackwardReadinessGate",
    "StatevectorTrainingClaimabilityGate",
    "attach_distributed_evidence_contract",
    "attach_mps_runtime_summary",
    "audit_distributed_scalability",
    "CLAIMABILITY_STATUSES",
    "CLAIM_EVIDENCE_TYPES",
    "MPS_BACKWARD_READINESS_STATUSES",
    "SCALABLE_DISTRIBUTION_SEMANTICS",
    "STATEVECTOR_TRAINING_CLAIMABILITY_STATUSES",
    "evaluate_distributed_evidence_contract",
    "evaluate_distributed_transport_evidence",
    "evaluate_mps_backward_readiness",
    "evaluate_statevector_training_claimability",
)
