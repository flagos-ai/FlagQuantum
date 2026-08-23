"""Correctness certification foundations for repository and extension tests."""

from .correctness import (
    CERTIFICATION_VERSION,
    CertificationCase,
    CertificationResult,
    FailureArtifact,
    TolerancePolicy,
    certification_matrix,
    execute_certification_case,
    generate_circuit_ir,
)
from .mps_capacity_certification import (
    MPSCapacityCertificationError,
    finalize_capacity_source_integrity,
    require_capacity_source_integrity,
    require_general_mps_capacity,
)
from .mps_certification import (
    MPSCertificationError,
    require_mps_numerical_certification,
)
from .mps_communication_certification import (
    MPSCommunicationCertificationError,
    require_mps_communication,
)
from .mps_portability_certification import (
    MPSPortabilityCertificationError,
    require_mps_portability,
)
from .mps_scaling_certification import (
    MPSScalingCertificationError,
    require_mps_scaling,
)
from .mps_stability_certification import (
    MPSStabilityCertificationError,
    require_mps_stability,
)
from .watchdog import PhaseAwareWatchdog, ProgressSnapshot, StallDiagnosis

__all__ = (
    "CERTIFICATION_VERSION",
    "CertificationCase",
    "CertificationResult",
    "FailureArtifact",
    "TolerancePolicy",
    "certification_matrix",
    "execute_certification_case",
    "generate_circuit_ir",
    "PhaseAwareWatchdog",
    "ProgressSnapshot",
    "StallDiagnosis",
    "MPSCertificationError",
    "require_mps_numerical_certification",
    "MPSCapacityCertificationError",
    "finalize_capacity_source_integrity",
    "require_capacity_source_integrity",
    "require_general_mps_capacity",
    "MPSStabilityCertificationError",
    "require_mps_stability",
    "MPSCommunicationCertificationError",
    "require_mps_communication",
    "MPSScalingCertificationError",
    "require_mps_scaling",
    "MPSPortabilityCertificationError",
    "require_mps_portability",
)
