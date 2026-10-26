"""Caller-configured freshness assessment for frozen QPU Twin evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from .drift import TwinCalibrationDrift
from .history import TwinCalibrationHistory
from .model import QPUDigitalTwin
from .series import TwinValidationSeries
from .validation_history import TwinValidationHistory

TwinFreshnessStatus = Literal[
    "current_under_policy",
    "refresh_due",
    "insufficient_evidence",
    "incompatible_history",
]

_CURRENT: TwinFreshnessStatus = "current_under_policy"
_DUE: TwinFreshnessStatus = "refresh_due"
_INSUFFICIENT: TwinFreshnessStatus = "insufficient_evidence"
_INCOMPATIBLE: TwinFreshnessStatus = "incompatible_history"


def _duration(value: timedelta | None, name: str) -> timedelta | None:
    if value is None:
        return None
    if not isinstance(value, timedelta):
        raise TypeError(f"{name} must be a datetime.timedelta")
    seconds = value.total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _threshold(
    value: float | None,
    name: str,
    *,
    unit_interval: bool = False,
) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    if unit_interval and normalized > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Twin freshness timestamps must include a timezone")
    return parsed


def _as_of(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("as_of must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("as_of must include a timezone")
    return value


def _age_seconds(as_of: datetime, captured_at: str) -> float:
    return (as_of - _timestamp(captured_at)).total_seconds()


def _target(twin: QPUDigitalTwin) -> tuple[str, str, tuple[int, ...]]:
    snapshot = twin.snapshot
    return snapshot.provider, snapshot.backend_name, snapshot.physical_qubits


@dataclass(frozen=True)
class TwinFreshnessAssessment:
    """Evidence-freshness facts evaluated under one explicit caller policy."""

    status: TwinFreshnessStatus
    snapshot_identity: str
    evaluated_at: str
    calibration_age_seconds: float
    validation_snapshot_age_seconds: float | None
    next_refresh_at: str | None
    calibration_drift: TwinCalibrationDrift | None
    validation_series: TwinValidationSeries | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {
            _CURRENT,
            _DUE,
            _INSUFFICIENT,
            _INCOMPATIBLE,
        }:
            raise ValueError("unsupported Twin freshness status")
        if len(self.snapshot_identity) != 64 or any(
            character not in "0123456789abcdef" for character in self.snapshot_identity
        ):
            raise ValueError("snapshot_identity must be a lowercase SHA-256 digest")
        _timestamp(self.evaluated_at)
        if (
            not math.isfinite(self.calibration_age_seconds)
            or self.calibration_age_seconds < 0
        ):
            raise ValueError("calibration_age_seconds must be finite and non-negative")
        validation_age = self.validation_snapshot_age_seconds
        if validation_age is not None and (
            not math.isfinite(validation_age) or validation_age < 0
        ):
            raise ValueError(
                "validation_snapshot_age_seconds must be finite and non-negative"
            )
        if self.next_refresh_at is not None:
            _timestamp(self.next_refresh_at)
        if self.calibration_drift is not None and not isinstance(
            self.calibration_drift, TwinCalibrationDrift
        ):
            raise TypeError("calibration_drift must be a TwinCalibrationDrift")
        if self.validation_series is not None and not isinstance(
            self.validation_series, TwinValidationSeries
        ):
            raise TypeError("validation_series must be a TwinValidationSeries")
        reasons = tuple(dict.fromkeys(str(reason) for reason in self.reasons))
        if any(not reason.strip() for reason in reasons):
            raise ValueError("freshness reasons must be non-empty strings")
        if (self.status == _CURRENT) == bool(reasons):
            raise ValueError(
                "current_under_policy requires no reasons and other statuses require reasons"
            )
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True)
class TwinFreshnessPolicy:
    """Explicit application thresholds for offline Twin freshness assessment.

    ``maximum_validation_age`` measures the age of the calibration snapshot
    associated with the latest validation series. The current validation
    artifact does not carry an authoritative hardware-completion timestamp, so
    this policy never presents that age as task or result age.

    Examples:
        policy = fq.twin.TwinFreshnessPolicy(
            maximum_calibration_age=timedelta(hours=4),
            maximum_validation_age=timedelta(hours=8),
            maximum_readout_tv_drift=0.02,
            maximum_verified_tv_error_bound=0.10,
        )
        assessment = policy.assess(
            current_twin,
            calibration_history=calibration_history,
            validation_history=validation_history,
            as_of=datetime.now(timezone.utc),
        )
    """

    maximum_calibration_age: timedelta | None = None
    maximum_validation_age: timedelta | None = None
    maximum_relative_t1_drift: float | None = None
    maximum_relative_t2_drift: float | None = None
    maximum_readout_tv_drift: float | None = None
    maximum_relative_gate_duration_drift: float | None = None
    maximum_verified_tv_error_bound: float | None = None

    def __post_init__(self) -> None:
        calibration_age = _duration(
            self.maximum_calibration_age, "maximum_calibration_age"
        )
        validation_age = _duration(
            self.maximum_validation_age, "maximum_validation_age"
        )
        relative_t1 = _threshold(
            self.maximum_relative_t1_drift, "maximum_relative_t1_drift"
        )
        relative_t2 = _threshold(
            self.maximum_relative_t2_drift, "maximum_relative_t2_drift"
        )
        readout = _threshold(
            self.maximum_readout_tv_drift,
            "maximum_readout_tv_drift",
            unit_interval=True,
        )
        gate_duration = _threshold(
            self.maximum_relative_gate_duration_drift,
            "maximum_relative_gate_duration_drift",
        )
        error_bound = _threshold(
            self.maximum_verified_tv_error_bound,
            "maximum_verified_tv_error_bound",
            unit_interval=True,
        )
        thresholds = (
            calibration_age,
            validation_age,
            relative_t1,
            relative_t2,
            readout,
            gate_duration,
            error_bound,
        )
        if all(value is None for value in thresholds):
            raise ValueError("Twin freshness policy requires at least one threshold")
        object.__setattr__(self, "maximum_calibration_age", calibration_age)
        object.__setattr__(self, "maximum_validation_age", validation_age)
        object.__setattr__(self, "maximum_relative_t1_drift", relative_t1)
        object.__setattr__(self, "maximum_relative_t2_drift", relative_t2)
        object.__setattr__(self, "maximum_readout_tv_drift", readout)
        object.__setattr__(self, "maximum_relative_gate_duration_drift", gate_duration)
        object.__setattr__(self, "maximum_verified_tv_error_bound", error_bound)

    def assess(
        self,
        twin: QPUDigitalTwin,
        *,
        calibration_history: TwinCalibrationHistory | None = None,
        validation_history: TwinValidationHistory | None = None,
        as_of: datetime,
    ) -> TwinFreshnessAssessment:
        """Assess one frozen Twin without provider I/O or model mutation."""

        if not isinstance(twin, QPUDigitalTwin):
            raise TypeError("twin must be a QPUDigitalTwin")
        if calibration_history is not None and not isinstance(
            calibration_history, TwinCalibrationHistory
        ):
            raise TypeError("calibration_history must be a TwinCalibrationHistory")
        if validation_history is not None and not isinstance(
            validation_history, TwinValidationHistory
        ):
            raise TypeError("validation_history must be a TwinValidationHistory")
        evaluated = _as_of(as_of)
        snapshot = twin.snapshot
        calibration_age_seconds = _age_seconds(evaluated, snapshot.captured_at)

        incompatible: list[str] = []
        insufficient: list[str] = []
        due: list[str] = []
        deadlines: list[datetime] = []

        if calibration_age_seconds < 0:
            incompatible.append("as_of_precedes_calibration_snapshot")
            calibration_age_seconds = 0.0
        if self.maximum_calibration_age is not None:
            deadline = _timestamp(snapshot.captured_at) + self.maximum_calibration_age
            deadlines.append(deadline)
            if evaluated > deadline:
                due.append("maximum_calibration_age_exceeded")

        drift_required = any(
            threshold is not None
            for threshold in (
                self.maximum_relative_t1_drift,
                self.maximum_relative_t2_drift,
                self.maximum_readout_tv_drift,
                self.maximum_relative_gate_duration_drift,
            )
        )
        drift: TwinCalibrationDrift | None = None
        if calibration_history is None:
            if drift_required:
                insufficient.append("calibration_history_required")
        else:
            calibration_target = (
                calibration_history.provider,
                calibration_history.backend_name,
                calibration_history.physical_qubits,
            )
            if calibration_target != _target(twin):
                incompatible.append("calibration_history_target_mismatch")
            elif calibration_history.snapshot_identities[-1] != snapshot.identity:
                incompatible.append("calibration_history_current_snapshot_mismatch")
            else:
                drift = calibration_history.latest_interval_drift
                _assess_drift(self, drift, due, insufficient)

        validation_required = (
            self.maximum_validation_age is not None
            or self.maximum_verified_tv_error_bound is not None
        )
        latest_series: TwinValidationSeries | None = None
        validation_age_seconds: float | None = None
        if validation_history is None:
            if validation_required:
                insufficient.append("validation_history_required")
        else:
            validation_target = (
                validation_history.provider,
                validation_history.backend_name,
                validation_history.physical_qubits,
            )
            if validation_target != _target(twin):
                incompatible.append("validation_history_target_mismatch")
            else:
                latest_series = validation_history.validation_series[-1]
                validation_captured_at = validation_history.captured_at[-1]
                validation_age_seconds = _age_seconds(evaluated, validation_captured_at)
                if validation_age_seconds < 0:
                    incompatible.append("as_of_precedes_validation_snapshot")
                    validation_age_seconds = 0.0
                if latest_series.snapshot_identity != snapshot.identity:
                    due.append("current_snapshot_not_validated")
                if self.maximum_validation_age is not None:
                    deadline = (
                        _timestamp(validation_captured_at) + self.maximum_validation_age
                    )
                    deadlines.append(deadline)
                    if evaluated > deadline:
                        due.append("maximum_validation_age_exceeded")
                if (
                    self.maximum_verified_tv_error_bound is not None
                    and latest_series.verified_tv_error_bound
                    > self.maximum_verified_tv_error_bound
                ):
                    due.append("maximum_verified_tv_error_bound_exceeded")

        reasons: tuple[str, ...]
        if incompatible:
            status = _INCOMPATIBLE
            reasons = tuple(dict.fromkeys(incompatible))
        elif insufficient:
            status = _INSUFFICIENT
            reasons = tuple(dict.fromkeys(insufficient))
        elif due:
            status = _DUE
            reasons = tuple(dict.fromkeys(due))
        else:
            status = _CURRENT
            reasons = ()

        next_refresh: datetime | None
        if status == _DUE:
            next_refresh = evaluated
        elif status == _CURRENT and deadlines:
            next_refresh = min(deadlines)
        else:
            next_refresh = None
        return TwinFreshnessAssessment(
            status=status,
            snapshot_identity=snapshot.identity,
            evaluated_at=evaluated.isoformat(),
            calibration_age_seconds=calibration_age_seconds,
            validation_snapshot_age_seconds=validation_age_seconds,
            next_refresh_at=(
                None if next_refresh is None else next_refresh.isoformat()
            ),
            calibration_drift=drift,
            validation_series=latest_series,
            reasons=reasons,
        )


def _assess_drift(
    policy: TwinFreshnessPolicy,
    drift: TwinCalibrationDrift,
    due: list[str],
    insufficient: list[str],
) -> None:
    if (
        policy.maximum_relative_t1_drift is not None
        and drift.maximum_relative_t1_change > policy.maximum_relative_t1_drift
    ):
        due.append("maximum_relative_t1_drift_exceeded")
    if (
        policy.maximum_relative_t2_drift is not None
        and drift.maximum_relative_t2_change > policy.maximum_relative_t2_drift
    ):
        due.append("maximum_relative_t2_drift_exceeded")
    if policy.maximum_readout_tv_drift is not None:
        if drift.maximum_readout_tv_distance is None:
            insufficient.append("readout_drift_unavailable")
        elif drift.maximum_readout_tv_distance > policy.maximum_readout_tv_drift:
            due.append("maximum_readout_tv_drift_exceeded")
    if policy.maximum_relative_gate_duration_drift is not None:
        if drift.maximum_relative_gate_duration_change is None:
            insufficient.append("gate_duration_drift_unavailable")
        elif (
            drift.maximum_relative_gate_duration_change
            > policy.maximum_relative_gate_duration_drift
        ):
            due.append("maximum_relative_gate_duration_drift_exceeded")


__all__ = (
    "TwinFreshnessAssessment",
    "TwinFreshnessPolicy",
    "TwinFreshnessStatus",
)
