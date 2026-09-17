"""Chronological calibration history for comparable frozen QPU Twins."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

from .drift import (
    TwinCalibrationDrift,
    TwinGateDurationDrift,
    TwinQubitCalibrationDrift,
    compare_calibrations,
)
from .model import QPUDigitalTwin

_HISTORY_SCHEMA = "flagquantum.twin_calibration_history.v1"
_HISTORY_FIELDS = {
    "schema",
    "provider",
    "backend_name",
    "physical_qubits",
    "snapshot_identities",
    "captured_at",
    "observation_count",
    "baseline_drifts",
    "interval_drifts",
}
_DRIFT_FIELDS = {
    "schema",
    "reference_snapshot_identity",
    "current_snapshot_identity",
    "provider",
    "backend_name",
    "physical_qubits",
    "elapsed_seconds",
    "qubit_drifts",
    "gate_duration_drifts",
    "channel_model_changed",
    "maximum_relative_t1_change",
    "maximum_relative_t2_change",
    "maximum_readout_tv_distance",
    "maximum_relative_gate_duration_change",
    "has_observed_drift",
}
_QUBIT_DRIFT_FIELDS = {
    "schema",
    "physical_qubit",
    "reference_t1_seconds",
    "current_t1_seconds",
    "relative_t1_delta",
    "reference_t2_seconds",
    "current_t2_seconds",
    "relative_t2_delta",
    "maximum_readout_row_tv_distance",
}
_GATE_DRIFT_FIELDS = {
    "schema",
    "gate_name",
    "physical_qubits",
    "reference_duration_seconds",
    "current_duration_seconds",
    "absolute_duration_delta_seconds",
    "relative_duration_delta",
}


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _require_fields(
    payload: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} fields do not match the v1 schema: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _qubit_drift_from_dict(payload: Mapping[str, Any]) -> TwinQubitCalibrationDrift:
    _require_fields(payload, _QUBIT_DRIFT_FIELDS, name="Twin qubit drift")
    return TwinQubitCalibrationDrift(
        schema=str(payload["schema"]),
        physical_qubit=int(payload["physical_qubit"]),
        reference_t1_seconds=float(payload["reference_t1_seconds"]),
        current_t1_seconds=float(payload["current_t1_seconds"]),
        relative_t1_delta=float(payload["relative_t1_delta"]),
        reference_t2_seconds=float(payload["reference_t2_seconds"]),
        current_t2_seconds=float(payload["current_t2_seconds"]),
        relative_t2_delta=float(payload["relative_t2_delta"]),
        maximum_readout_row_tv_distance=_optional_float(
            payload["maximum_readout_row_tv_distance"]
        ),
    )


def _gate_drift_from_dict(payload: Mapping[str, Any]) -> TwinGateDurationDrift:
    _require_fields(payload, _GATE_DRIFT_FIELDS, name="Twin gate-duration drift")
    physical_qubits = payload["physical_qubits"]
    return TwinGateDurationDrift(
        schema=str(payload["schema"]),
        gate_name=str(payload["gate_name"]),
        physical_qubits=(
            None
            if physical_qubits is None
            else tuple(int(qubit) for qubit in physical_qubits)
        ),
        reference_duration_seconds=float(payload["reference_duration_seconds"]),
        current_duration_seconds=float(payload["current_duration_seconds"]),
        absolute_duration_delta_seconds=float(
            payload["absolute_duration_delta_seconds"]
        ),
        relative_duration_delta=_optional_float(payload["relative_duration_delta"]),
    )


def _drift_from_dict(payload: Mapping[str, Any]) -> TwinCalibrationDrift:
    _require_fields(payload, _DRIFT_FIELDS, name="Twin calibration drift")
    qubit_payloads = payload["qubit_drifts"]
    gate_payloads = payload["gate_duration_drifts"]
    if not isinstance(qubit_payloads, list) or not isinstance(gate_payloads, list):
        raise ValueError("Twin calibration drift details must be JSON arrays")
    if any(not isinstance(item, Mapping) for item in (*qubit_payloads, *gate_payloads)):
        raise ValueError("Twin calibration drift details must be JSON objects")
    drift = TwinCalibrationDrift(
        schema=str(payload["schema"]),
        reference_snapshot_identity=str(payload["reference_snapshot_identity"]),
        current_snapshot_identity=str(payload["current_snapshot_identity"]),
        provider=str(payload["provider"]),
        backend_name=str(payload["backend_name"]),
        physical_qubits=tuple(int(qubit) for qubit in payload["physical_qubits"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        qubit_drifts=tuple(_qubit_drift_from_dict(item) for item in qubit_payloads),
        gate_duration_drifts=tuple(
            _gate_drift_from_dict(item) for item in gate_payloads
        ),
        channel_model_changed=payload["channel_model_changed"],
    )
    if _canonical(drift.to_dict()) != _canonical(payload):
        raise ValueError("Twin calibration drift is not in canonical v1 form")
    return drift


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


def load_calibration_history(
    path: str | PathLike[str],
) -> TwinCalibrationHistory:
    """Load one calibration history without contacting a provider.

    Examples:
        history = fq.twin.load_calibration_history("calibration-history.json")

    Raises:
        ValueError: If the file is not a canonical calibration history.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin calibration history from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin calibration-history file must contain a JSON object")
    _require_fields(payload, _HISTORY_FIELDS, name="Twin calibration history")
    baseline_payloads = payload["baseline_drifts"]
    interval_payloads = payload["interval_drifts"]
    if not isinstance(baseline_payloads, list) or not isinstance(
        interval_payloads, list
    ):
        raise ValueError("Twin calibration-history drifts must be JSON arrays")
    if any(
        not isinstance(item, Mapping)
        for item in (*baseline_payloads, *interval_payloads)
    ):
        raise ValueError("Twin calibration-history drifts must be JSON objects")
    try:
        history = TwinCalibrationHistory(
            schema=str(payload["schema"]),
            provider=str(payload["provider"]),
            backend_name=str(payload["backend_name"]),
            physical_qubits=tuple(int(qubit) for qubit in payload["physical_qubits"]),
            snapshot_identities=tuple(payload["snapshot_identities"]),
            captured_at=tuple(payload["captured_at"]),
            baseline_drifts=tuple(_drift_from_dict(item) for item in baseline_payloads),
            interval_drifts=tuple(_drift_from_dict(item) for item in interval_payloads),
        )
        if _canonical(history.to_dict()) != _canonical(payload):
            raise ValueError("Twin calibration history is not in canonical v1 form")
        return history
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("Invalid Twin calibration history") from error


def dump_calibration_history(
    history: TwinCalibrationHistory,
    path: str | PathLike[str],
) -> None:
    """Write one private history file without replacing different content.

    Examples:
        fq.twin.dump_calibration_history(history, "calibration-history.json")

    Raises:
        TypeError: If ``history`` is not a Twin calibration history.
        ValueError: If the destination cannot be written safely.
    """

    if not isinstance(history, TwinCalibrationHistory):
        raise TypeError("history must be a TwinCalibrationHistory")
    destination = Path(path)
    encoded = _canonical(history.to_dict()) + "\n"
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        try:
            existing = load_calibration_history(destination)
        except ValueError as error:
            raise ValueError(
                "Refusing to replace invalid Twin calibration history at "
                f"{destination}"
            ) from error
        if _canonical(existing.to_dict()) != _canonical(history.to_dict()):
            raise ValueError(
                "Refusing to replace different Twin calibration history at "
                f"{destination}"
            )
    except OSError as error:
        raise ValueError(
            f"Cannot write Twin calibration history to {destination}"
        ) from error


__all__ = (
    "build_calibration_history",
    "dump_calibration_history",
    "load_calibration_history",
    "TwinCalibrationHistory",
)
