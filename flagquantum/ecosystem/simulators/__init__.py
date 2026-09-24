"""Evidence-scoped simulator recommendations without automatic routing."""

from .advisor import (
    EvidenceLevel,
    RecommendationStatus,
    SimulatorAdvisorEvidenceError,
    SimulatorCandidate,
    SimulatorRecommendation,
    recommend,
)

__all__ = (
    "EvidenceLevel",
    "RecommendationStatus",
    "SimulatorAdvisorEvidenceError",
    "SimulatorCandidate",
    "SimulatorRecommendation",
    "recommend",
)
