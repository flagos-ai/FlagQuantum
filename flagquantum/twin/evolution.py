"""Aligned device drift and prediction-validation history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .drift import TwinCalibrationDrift
from .history import TwinCalibrationHistory
from .validation_history import TwinValidationHistory

_EVOLUTION_SCHEMA = "flagquantum.twin_evolution_history.v1"


def _changes(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(current - previous for previous, current in zip(values, values[1:]))


def _optional_changes(
    values: Sequence[float | None],
) -> tuple[float | None, ...]:
    return tuple(
        None if previous is None or current is None else current - previous
        for previous, current in zip(values, values[1:])
    )


@dataclass(frozen=True)
class TwinEvolutionHistory:
    """Device drift aligned with validation changes for the same snapshots."""

    calibration_history: TwinCalibrationHistory
    validation_history: TwinValidationHistory
    schema: str = _EVOLUTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _EVOLUTION_SCHEMA:
            raise ValueError("unsupported Twin evolution-history schema")
        if not isinstance(self.calibration_history, TwinCalibrationHistory):
            raise TypeError("calibration_history must be a TwinCalibrationHistory")
        if not isinstance(self.validation_history, TwinValidationHistory):
            raise TypeError("validation_history must be a TwinValidationHistory")
        calibration = self.calibration_history
        validation = self.validation_history
        if (
            calibration.provider,
            calibration.backend_name,
            calibration.physical_qubits,
        ) != (
            validation.provider,
            validation.backend_name,
            validation.physical_qubits,
        ):
            raise ValueError("Twin histories must use the same target and mapping")
        if calibration.snapshot_identities != validation.snapshot_identities:
            raise ValueError("Twin histories must contain the same snapshot identities")
        if calibration.captured_at != validation.captured_at:
            raise ValueError("Twin histories must contain the same captured_at values")

    @property
    def observation_count(self) -> int:
        return self.validation_history.observation_count

    @property
    def twin_qpu_agreement_changes(self) -> tuple[float, ...]:
        return _changes(self.validation_history.mean_twin_qpu_agreements)

    @property
    def ideal_qpu_agreement_changes(self) -> tuple[float, ...]:
        return _changes(self.validation_history.mean_ideal_qpu_agreements)

    @property
    def qpu_repeatability_changes(self) -> tuple[float | None, ...]:
        return _optional_changes(self.validation_history.mean_qpu_repeatabilities)

    @property
    def verified_tv_error_bound_changes(self) -> tuple[float, ...]:
        return _changes(self.validation_history.verified_tv_error_bounds)

    @property
    def latest_calibration_drift(self) -> TwinCalibrationDrift:
        """Return the latest adjacent-interval calibration drift."""

        return self.calibration_history.latest_interval_drift

    @property
    def latest_twin_agreement_change(self) -> float:
        return self.twin_qpu_agreement_changes[-1]

    @property
    def latest_ideal_agreement_change(self) -> float:
        return self.ideal_qpu_agreement_changes[-1]

    @property
    def latest_qpu_repeatability_change(self) -> float | None:
        return self.qpu_repeatability_changes[-1]

    @property
    def latest_verified_bound_change(self) -> float:
        return self.verified_tv_error_bound_changes[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "calibration_history": self.calibration_history.to_dict(),
            "validation_history": self.validation_history.to_dict(),
            "observation_count": self.observation_count,
            "twin_qpu_agreement_changes": list(self.twin_qpu_agreement_changes),
            "ideal_qpu_agreement_changes": list(self.ideal_qpu_agreement_changes),
            "qpu_repeatability_changes": list(self.qpu_repeatability_changes),
            "verified_tv_error_bound_changes": list(
                self.verified_tv_error_bound_changes
            ),
        }


def align_histories(
    calibration_history: TwinCalibrationHistory,
    validation_history: TwinValidationHistory,
) -> TwinEvolutionHistory:
    """Align device drift and validation changes by exact frozen snapshots.

    Examples:
        evolution = fq.twin.align_histories(calibration_history, validation_history)
        print(evolution.latest_twin_agreement_change)

    This function reports aligned observations; it does not infer causality.
    """

    return TwinEvolutionHistory(
        calibration_history=calibration_history,
        validation_history=validation_history,
    )


__all__ = ("align_histories", "TwinEvolutionHistory")
