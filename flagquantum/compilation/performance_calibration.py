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


@dataclass(frozen=True)
class CalibratedWorldSizeSelection:
    world_size: int
    global_elements: int
    threshold_elements: int | None
    relative_uncertainty: float
    source: str


def select_calibrated_world_size(
    global_elements: int, artifact: Mapping[str, Any]
) -> CalibratedWorldSizeSelection:
    if artifact.get("schema") != "flagquantum_crossover_v1":
        raise ValueError("planner selection requires a crossover artifact")
    thresholds = artifact.get("planner_selection_thresholds", {})
    if not isinstance(thresholds, Mapping):
        raise ValueError("crossover artifact lacks planner selection thresholds")
    selected, selected_threshold, uncertainty = 1, None, 0.0
    for world_text, row in sorted(thresholds.items(), key=lambda item: int(item[0])):
        if not isinstance(row, Mapping):
            continue
        threshold = row.get("minimum_global_elements")
        if threshold is not None and global_elements >= int(threshold):
            selected = int(world_text)
            selected_threshold = int(threshold)
            uncertainty = float(row["uncertainty_margin_fraction"])
    return CalibratedWorldSizeSelection(
        world_size=selected,
        global_elements=global_elements,
        threshold_elements=selected_threshold,
        relative_uncertainty=uncertainty,
        source="matched_workload_crossover",
    )


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
    "CalibratedWorldSizeSelection",
    "calibrate_plan_cost",
    "select_calibrated_world_size",
]
