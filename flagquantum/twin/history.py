"""Chronological calibration history for comparable frozen QPU Twins."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from .drift import TwinCalibrationDrift, compare_calibrations
from .model import QPUDigitalTwin

_HISTORY_SCHEMA = "flagquantum.twin_calibration_history.v1"


@dataclass(frozen=True)
class TwinCalibrationHistory:
    """Cumulative and interval drift for an ordered series of frozen Twins."""

    provider: str
    backend_name: str
    physical_qubits: tuple[int, ...]
    snapshot_identities: tuple[str, ...]
    captured_at: tuple[str, ...]
    baseline_drifts: tuple[TwinCalibrationDrift, ...]
    interval_drifts: tuple[TwinCalibrationDrift, ...]
    schema: str = _HISTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _HISTORY_SCHEMA:
            raise ValueError("unsupported Twin calibration-history schema")
        physical_qubits = tuple(int(qubit) for qubit in self.physical_qubits)
        identities = tuple(self.snapshot_identities)
        timestamps = tuple(self.captured_at)
        baseline = tuple(self.baseline_drifts)
        intervals = tuple(self.interval_drifts)
        if not self.provider.strip() or not self.backend_name.strip():
            raise ValueError("calibration history requires a provider and backend")
        if (
            not physical_qubits
            or len(physical_qubits) != len(set(physical_qubits))
            or any(qubit < 0 for qubit in physical_qubits)
        ):
            raise ValueError("physical_qubits must be unique and non-negative")
        if len(identities) < 2:
            raise ValueError("calibration history requires at least two Twin snapshots")
        if len(set(identities)) != len(identities):
            raise ValueError("Twin snapshot identities must be unique")
        if len(timestamps) != len(identities):
            raise ValueError("captured_at must align with snapshot identities")
        if (
            len(baseline) != len(identities) - 1
            or len(intervals) != len(identities) - 1
        ):
            raise ValueError("calibration drift sequences must cover every transition")
        if any(
            not isinstance(item, TwinCalibrationDrift)
            for item in (*baseline, *intervals)
        ):
            raise TypeError("calibration history requires TwinCalibrationDrift records")
        parsed = tuple(datetime.fromisoformat(value) for value in timestamps)
        if any(right <= left for left, right in zip(parsed, parsed[1:])):
            raise ValueError(
                "Twin snapshots must have strictly increasing captured_at values"
            )
        target = (self.provider, self.backend_name, physical_qubits)
        for index, drift in enumerate(baseline):
            if (drift.provider, drift.backend_name, drift.physical_qubits) != target:
                raise ValueError(
                    "baseline drift target does not match calibration history"
                )
            if (
                drift.reference_snapshot_identity != identities[0]
                or drift.current_snapshot_identity != identities[index + 1]
            ):
                raise ValueError(
                    "baseline drift identities do not match calibration history"
                )
        for index, drift in enumerate(intervals):
            if (drift.provider, drift.backend_name, drift.physical_qubits) != target:
                raise ValueError(
                    "interval drift target does not match calibration history"
                )
            if (
                drift.reference_snapshot_identity != identities[index]
                or drift.current_snapshot_identity != identities[index + 1]
            ):
                raise ValueError(
                    "interval drift identities do not match calibration history"
                )
        object.__setattr__(self, "physical_qubits", physical_qubits)
        object.__setattr__(self, "snapshot_identities", identities)
        object.__setattr__(self, "captured_at", timestamps)
        object.__setattr__(self, "baseline_drifts", baseline)
        object.__setattr__(self, "interval_drifts", intervals)

    @property
    def observation_count(self) -> int:
        return len(self.snapshot_identities)

    @property
    def latest_baseline_drift(self) -> TwinCalibrationDrift:
        return self.baseline_drifts[-1]

    @property
    def latest_interval_drift(self) -> TwinCalibrationDrift:
        return self.interval_drifts[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provider": self.provider,
            "backend_name": self.backend_name,
            "physical_qubits": list(self.physical_qubits),
            "snapshot_identities": list(self.snapshot_identities),
            "captured_at": list(self.captured_at),
            "observation_count": self.observation_count,
            "baseline_drifts": [item.to_dict() for item in self.baseline_drifts],
            "interval_drifts": [item.to_dict() for item in self.interval_drifts],
        }


def build_calibration_history(
    twins: Sequence[QPUDigitalTwin],
) -> TwinCalibrationHistory:
    """Build offline drift series from strictly chronological frozen Twins."""

    if isinstance(twins, (str, bytes)):
        raise TypeError("twins must be a sequence of QPUDigitalTwin objects")
    ordered = tuple(twins)
    if len(ordered) < 2:
        raise ValueError("calibration history requires at least two Twin snapshots")
    if any(not isinstance(twin, QPUDigitalTwin) for twin in ordered):
        raise TypeError("twins must be a sequence of QPUDigitalTwin objects")
    identities = tuple(twin.snapshot.identity for twin in ordered)
    if len(set(identities)) != len(identities):
        raise ValueError("Twin snapshot identities must be unique")
    captured_at = tuple(twin.snapshot.captured_at for twin in ordered)
    parsed = tuple(datetime.fromisoformat(value) for value in captured_at)
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError(
            "Twin snapshots must have strictly increasing captured_at values"
        )

    baseline = tuple(compare_calibrations(ordered[0], twin) for twin in ordered[1:])
    intervals = tuple(
        compare_calibrations(reference, current)
        for reference, current in zip(ordered, ordered[1:])
    )
    snapshot = ordered[0].snapshot
    return TwinCalibrationHistory(
        provider=snapshot.provider,
        backend_name=snapshot.backend_name,
        physical_qubits=snapshot.physical_qubits,
        snapshot_identities=identities,
        captured_at=captured_at,
        baseline_drifts=baseline,
        interval_drifts=intervals,
    )


__all__ = ("build_calibration_history", "TwinCalibrationHistory")
