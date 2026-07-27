"""Typed separation of hybrid-model correctness, accuracy, and performance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class HybridAcceptanceReport:
    model: str
    policy: Mapping[str, Any]
    correctness: Mapping[str, Any]
    accuracy: Mapping[str, Any]
    performance: Mapping[str, Any]
    deployment: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "policy": dict(self.policy),
            "correctness": dict(self.correctness),
            "accuracy": dict(self.accuracy),
            "performance": dict(self.performance),
            "deployment": dict(self.deployment),
            "accuracy_is_performance_claim": False,
            "scalability_claim_allowed": False,
        }


__all__ = ["HybridAcceptanceReport"]
