"""Evidence-scoped simulator recommendations without automatic routing."""

from .advisor import (
    RecommendationStatus,
    SimulatorAdvisorEvidenceError,
    SimulatorCandidate,
    SimulatorRecommendation,
    recommend,
)

__all__ = (
    "RecommendationStatus",
    "SimulatorAdvisorEvidenceError",
    "SimulatorCandidate",
    "SimulatorRecommendation",
    "recommend",
)
