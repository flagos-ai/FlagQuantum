"""Hardware comparison reports for frozen digital-twin predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

_VALIDATION_SCHEMA = "flagquantum.twin_validation_report.v1"


def _total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("probability vectors must have equal length")
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class TwinValidationReport:
    """Distribution-level comparison with one hardware observation."""

    snapshot_identity: str
    circuit_identity: str
    hardware_probabilities: tuple[float, ...]
    ideal_hardware_total_variation: float
    twin_hardware_total_variation: float
    total_variation_improvement: float
    shots: int
    schema: str = _VALIDATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _VALIDATION_SCHEMA:
            raise ValueError("unsupported twin validation report schema")
        if self.shots <= 0:
            raise ValueError("validation report requires positive shots")
        if not self.hardware_probabilities or any(
            not math.isfinite(value) or value < 0
            for value in self.hardware_probabilities
        ):
            raise ValueError("hardware probabilities must be finite and non-negative")
        if not math.isclose(sum(self.hardware_probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("hardware probabilities must sum to one")
        distances = (
            self.ideal_hardware_total_variation,
            self.twin_hardware_total_variation,
        )
        if any(not math.isfinite(value) or value < 0 for value in distances):
            raise ValueError("validation distances must be finite and non-negative")
        expected = (
            self.ideal_hardware_total_variation - self.twin_hardware_total_variation
        )
        if not math.isclose(self.total_variation_improvement, expected, abs_tol=1e-12):
            raise ValueError(
                "total_variation_improvement does not match the reported distances"
            )

    @property
    def outperforms_ideal_baseline(self) -> bool:
        return self.total_variation_improvement > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "snapshot_identity": self.snapshot_identity,
            "circuit_identity": self.circuit_identity,
            "hardware_probabilities": list(self.hardware_probabilities),
            "ideal_hardware_total_variation": self.ideal_hardware_total_variation,
            "twin_hardware_total_variation": self.twin_hardware_total_variation,
            "total_variation_improvement": self.total_variation_improvement,
            "shots": self.shots,
            "outperforms_ideal_baseline": self.outperforms_ideal_baseline,
        }


__all__ = ("TwinValidationReport",)
