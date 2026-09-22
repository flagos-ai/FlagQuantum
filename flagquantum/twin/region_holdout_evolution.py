"""Aligned calibration drift and regional holdout evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .drift import TwinCalibrationDrift
from .history import TwinCalibrationHistory
from .region_holdout_history import TwinRegionHoldoutHistory

_EVOLUTION_SCHEMA = "flagquantum.twin_region_holdout_evolution.v1"


def _changes(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        current - previous
        for previous, current in zip(values, values[1:], strict=False)
    )


@dataclass(frozen=True)
class TwinRegionHoldoutEvolution:
    """Calibration drift aligned with fixed regional holdout evidence.

    The alignment exposes observed changes only. It does not infer causality,
    choose a trust threshold, update a Twin, or authorize workload routing.

    Examples:
        evolution = fq.twin.align_region_holdout_history(
            calibration_history,
            holdout_history,
        )
        print(evolution.holdout_twin_qpu_agreement_changes)
    """

    calibration_history: TwinCalibrationHistory
    holdout_history: TwinRegionHoldoutHistory
    schema: str = _EVOLUTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _EVOLUTION_SCHEMA:
            raise ValueError("unsupported Twin region holdout-evolution schema")
        if not isinstance(self.calibration_history, TwinCalibrationHistory):
            raise TypeError("calibration_history must be a TwinCalibrationHistory")
        if not isinstance(self.holdout_history, TwinRegionHoldoutHistory):
            raise TypeError("holdout_history must be a TwinRegionHoldoutHistory")
        calibration = self.calibration_history
        holdout = self.holdout_history
        if (
            calibration.provider,
            calibration.backend_name,
            calibration.physical_qubits,
        ) != (
            holdout.provider,
            holdout.backend_name,
            holdout.physical_qubits,
        ):
            raise ValueError("Twin histories must use the same target and mapping")
        if calibration.snapshot_identities != holdout.snapshot_identities:
            raise ValueError("Twin histories must contain the same snapshot identities")
        if calibration.captured_at != holdout.captured_at:
            raise ValueError("Twin histories must contain the same captured_at values")

    @property
    def observation_count(self) -> int:
        return self.holdout_history.observation_count

    @property
    def reference_twin_qpu_agreement_changes(self) -> tuple[float, ...]:
        return _changes(self.holdout_history.reference_twin_qpu_agreements)

    @property
    def holdout_twin_qpu_agreement_changes(self) -> tuple[float, ...]:
        return _changes(self.holdout_history.holdout_twin_qpu_agreements)

    @property
    def holdout_tv_error_increase_changes(self) -> tuple[float, ...]:
        return _changes(self.holdout_history.holdout_tv_error_increases)

    @property
    def holdout_simultaneous_tv_error_bound_changes(self) -> tuple[float, ...]:
        return _changes(self.holdout_history.holdout_simultaneous_tv_error_bounds)

    @property
    def interval_maximum_relative_t1_changes(self) -> tuple[float, ...]:
        return tuple(
            item.maximum_relative_t1_change
            for item in self.calibration_history.interval_drifts
        )

    @property
    def interval_maximum_relative_t2_changes(self) -> tuple[float, ...]:
        return tuple(
            item.maximum_relative_t2_change
            for item in self.calibration_history.interval_drifts
        )

    @property
    def interval_maximum_readout_tv_distances(
        self,
    ) -> tuple[float | None, ...]:
        return tuple(
            item.maximum_readout_tv_distance
            for item in self.calibration_history.interval_drifts
        )

    @property
    def interval_maximum_relative_gate_duration_changes(
        self,
    ) -> tuple[float | None, ...]:
        return tuple(
            item.maximum_relative_gate_duration_change
            for item in self.calibration_history.interval_drifts
        )

    @property
    def latest_calibration_drift(self) -> TwinCalibrationDrift:
        return self.calibration_history.latest_interval_drift

    @property
    def latest_holdout_twin_qpu_agreement_change(self) -> float:
        return self.holdout_twin_qpu_agreement_changes[-1]

    @property
    def latest_holdout_tv_error_increase_change(self) -> float:
        return self.holdout_tv_error_increase_changes[-1]

    @property
    def latest_holdout_simultaneous_tv_error_bound_change(self) -> float:
        return self.holdout_simultaneous_tv_error_bound_changes[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "calibration_history": self.calibration_history.to_dict(),
            "holdout_history": self.holdout_history.to_dict(),
            "observation_count": self.observation_count,
            "reference_twin_qpu_agreement_changes": list(
                self.reference_twin_qpu_agreement_changes
            ),
            "holdout_twin_qpu_agreement_changes": list(
                self.holdout_twin_qpu_agreement_changes
            ),
            "holdout_tv_error_increase_changes": list(
                self.holdout_tv_error_increase_changes
            ),
            "holdout_simultaneous_tv_error_bound_changes": list(
                self.holdout_simultaneous_tv_error_bound_changes
            ),
            "interval_maximum_relative_t1_changes": list(
                self.interval_maximum_relative_t1_changes
            ),
            "interval_maximum_relative_t2_changes": list(
                self.interval_maximum_relative_t2_changes
            ),
            "interval_maximum_readout_tv_distances": list(
                self.interval_maximum_readout_tv_distances
            ),
            "interval_maximum_relative_gate_duration_changes": list(
                self.interval_maximum_relative_gate_duration_changes
            ),
        }


def align_region_holdout_history(
    calibration_history: TwinCalibrationHistory,
    holdout_history: TwinRegionHoldoutHistory,
) -> TwinRegionHoldoutEvolution:
    """Align calibration drift and holdout changes by exact snapshots."""

    return TwinRegionHoldoutEvolution(
        calibration_history=calibration_history,
        holdout_history=holdout_history,
    )


__all__ = ("align_region_holdout_history", "TwinRegionHoldoutEvolution")
