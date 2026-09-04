"""Runtime-owned selection result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .candidate_plans import CircuitAnalysisView
from .candidates import RuntimeCandidate


@dataclass(frozen=True)
class RuntimeSelectionPlan:
    """Ranked Runtime candidates and the selected execution mode."""

    analysis: CircuitAnalysisView
    recommended_mode: str
    recommended_candidate: RuntimeCandidate
    candidates: tuple[RuntimeCandidate, ...]
    objective: str
    user_tier: str
    usability_contract: str
    world_size: int = 1
    local_world_size: int = 1
    node_count: int = 1

    def summary(self) -> dict[str, Any]:
        return {
            "planner": "runtime_selection",
            "objective": self.objective,
            "recommended_mode": self.recommended_mode,
            "recommended_candidate": self.recommended_candidate.summary(),
            "user_tier": self.user_tier,
            "usability_contract": self.usability_contract,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "n_wires": self.analysis.n_wires,
            "n_instructions": self.analysis.n_instructions,
            "depth": self.analysis.depth,
            "two_qubit_gates": self.analysis.two_qubit_gates,
            "multi_qubit_gates": self.analysis.multi_qubit_gates,
            "has_noise": self.analysis.has_noise,
            "memory_plan": dict(self.recommended_candidate.memory_plan or {}),
            "communication_plan": dict(
                self.recommended_candidate.communication_plan or {}
            ),
            "gradient_plan": dict(self.recommended_candidate.gradient_plan or {}),
            "deployment_plan": dict(self.recommended_candidate.deployment_plan or {}),
            "candidates": tuple(candidate.summary() for candidate in self.candidates),
        }


__all__ = ("RuntimeSelectionPlan",)
