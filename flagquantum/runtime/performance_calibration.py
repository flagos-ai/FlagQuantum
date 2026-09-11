"""Measured calibration adapter for planner cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CalibratedPlanCost:
    estimated_seconds: float
    relative_uncertainty: float
    measured_seconds_per_work_unit: float
    estimated_work_units: int
    source_benchmark: str


def calibrate_plan_cost(plan: Any, artifact: Mapping[str, Any]) -> CalibratedPlanCost:
    gate = artifact.get("performance_gate", {})
    if not isinstance(gate, Mapping) or not gate.get("passed"):
        raise ValueError("planner calibration requires a passing performance artifact")
    calibration = artifact.get("cost_model_calibration", {})
    if not isinstance(calibration, Mapping):
        raise ValueError("performance artifact lacks cost-model calibration")
    seconds_per_unit = float(calibration["seconds_per_work_unit"])
    work_units = max(int(plan.analysis.n_instructions), 1)
    return CalibratedPlanCost(
        estimated_seconds=seconds_per_unit * work_units,
        relative_uncertainty=float(calibration["relative_uncertainty"]),
        measured_seconds_per_work_unit=seconds_per_unit,
        estimated_work_units=work_units,
        source_benchmark=str(calibration["source_benchmark"]),
    )


__all__ = [
    "CalibratedPlanCost",
    "calibrate_plan_cost",
]
