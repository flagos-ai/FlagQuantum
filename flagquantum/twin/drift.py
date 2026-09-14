"""Provider-neutral comparison of frozen QPU Twin calibrations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .model import QPUDigitalTwin

_QUBIT_SCHEMA = "flagquantum.twin_qubit_calibration_drift.v1"
_GATE_SCHEMA = "flagquantum.twin_gate_duration_drift.v1"
_REPORT_SCHEMA = "flagquantum.twin_calibration_drift.v1"
_TIME_SCALES = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def _relative_delta(reference: float, current: float) -> float | None:
    if reference == 0.0:
        return 0.0 if current == 0.0 else None
    return (current - reference) / reference


def _readout_distance(reference: Any, current: Any) -> float | None:
    if reference is None and current is None:
        return None
    if reference is None or current is None:
        raise ValueError("readout calibration availability changed")
    return max(
        0.5 * sum(abs(left - right) for left, right in zip(left_row, right_row))
        for left_row, right_row in zip(reference.probabilities, current.probabilities)
    )


@dataclass(frozen=True)
class TwinQubitCalibrationDrift:
    """Calibration deltas for one mapped physical qubit."""

    physical_qubit: int
    reference_t1_seconds: float
    current_t1_seconds: float
    relative_t1_delta: float
    reference_t2_seconds: float
    current_t2_seconds: float
    relative_t2_delta: float
    maximum_readout_row_tv_distance: float | None
    schema: str = _QUBIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _QUBIT_SCHEMA:
            raise ValueError("unsupported Twin qubit-calibration drift schema")
        physical_qubit = int(self.physical_qubit)
        if physical_qubit < 0:
            raise ValueError("physical_qubit must be non-negative")
        for name, value in (
            ("reference_t1_seconds", self.reference_t1_seconds),
            ("current_t1_seconds", self.current_t1_seconds),
            ("reference_t2_seconds", self.reference_t2_seconds),
            ("current_t2_seconds", self.current_t2_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("relative_t1_delta", self.relative_t1_delta),
            ("relative_t2_delta", self.relative_t2_delta),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        expected_t1 = _relative_delta(
            self.reference_t1_seconds, self.current_t1_seconds
        )
        expected_t2 = _relative_delta(
            self.reference_t2_seconds, self.current_t2_seconds
        )
        if expected_t1 is None or not math.isclose(
            self.relative_t1_delta, expected_t1, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("relative_t1_delta does not match T1 values")
        if expected_t2 is None or not math.isclose(
            self.relative_t2_delta, expected_t2, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("relative_t2_delta does not match T2 values")
        distance = self.maximum_readout_row_tv_distance
        if distance is not None and (
            not math.isfinite(distance) or not 0.0 <= distance <= 1.0
        ):
            raise ValueError(
                "maximum_readout_row_tv_distance must be finite and in [0, 1]"
            )
        object.__setattr__(self, "physical_qubit", physical_qubit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "physical_qubit": self.physical_qubit,
            "reference_t1_seconds": self.reference_t1_seconds,
            "current_t1_seconds": self.current_t1_seconds,
            "relative_t1_delta": self.relative_t1_delta,
            "reference_t2_seconds": self.reference_t2_seconds,
            "current_t2_seconds": self.current_t2_seconds,
            "relative_t2_delta": self.relative_t2_delta,
            "maximum_readout_row_tv_distance": (self.maximum_readout_row_tv_distance),
        }


@dataclass(frozen=True)
class TwinGateDurationDrift:
    """Duration delta for one gate calibration scope."""

    gate_name: str
    physical_qubits: tuple[int, ...] | None
    reference_duration_seconds: float
    current_duration_seconds: float
    absolute_duration_delta_seconds: float
    relative_duration_delta: float | None
    schema: str = _GATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _GATE_SCHEMA:
            raise ValueError("unsupported Twin gate-duration drift schema")
        if not self.gate_name.strip() or self.gate_name != self.gate_name.lower():
            raise ValueError("gate_name must be non-empty and lowercase")
        physical_qubits = (
            None
            if self.physical_qubits is None
            else tuple(int(qubit) for qubit in self.physical_qubits)
        )
        if physical_qubits is not None and (
            not physical_qubits
            or len(physical_qubits) != len(set(physical_qubits))
            or any(qubit < 0 for qubit in physical_qubits)
        ):
            raise ValueError("physical_qubits must be unique and non-negative")
        for name, value in (
            ("reference_duration_seconds", self.reference_duration_seconds),
            ("current_duration_seconds", self.current_duration_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(self.absolute_duration_delta_seconds)
            or self.absolute_duration_delta_seconds < 0
        ):
            raise ValueError(
                "absolute_duration_delta_seconds must be finite and non-negative"
            )
        if self.relative_duration_delta is not None and not math.isfinite(
            self.relative_duration_delta
        ):
            raise ValueError("relative_duration_delta must be finite when present")
        expected_absolute = abs(
            self.current_duration_seconds - self.reference_duration_seconds
        )
        if not math.isclose(
            self.absolute_duration_delta_seconds,
            expected_absolute,
            rel_tol=1e-12,
            abs_tol=1e-18,
        ):
            raise ValueError("absolute duration delta does not match duration values")
        expected_relative = _relative_delta(
            self.reference_duration_seconds, self.current_duration_seconds
        )
        if expected_relative is None:
            if self.relative_duration_delta is not None:
                raise ValueError("relative duration delta is undefined")
        elif self.relative_duration_delta is None or not math.isclose(
            self.relative_duration_delta,
            expected_relative,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("relative duration delta does not match duration values")
        object.__setattr__(self, "physical_qubits", physical_qubits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "gate_name": self.gate_name,
            "physical_qubits": (
                None if self.physical_qubits is None else list(self.physical_qubits)
            ),
            "reference_duration_seconds": self.reference_duration_seconds,
            "current_duration_seconds": self.current_duration_seconds,
            "absolute_duration_delta_seconds": self.absolute_duration_delta_seconds,
            "relative_duration_delta": self.relative_duration_delta,
        }


@dataclass(frozen=True)
class TwinCalibrationDrift:
    """Observed calibration changes between two comparable frozen Twins."""

    reference_snapshot_identity: str
    current_snapshot_identity: str
    provider: str
    backend_name: str
    physical_qubits: tuple[int, ...]
    elapsed_seconds: float
    qubit_drifts: tuple[TwinQubitCalibrationDrift, ...]
    gate_duration_drifts: tuple[TwinGateDurationDrift, ...]
    channel_model_changed: bool
    schema: str = _REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _REPORT_SCHEMA:
            raise ValueError("unsupported Twin calibration-drift schema")
        for name, value in (
            ("reference_snapshot_identity", self.reference_snapshot_identity),
            ("current_snapshot_identity", self.current_snapshot_identity),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.provider.strip() or not self.backend_name.strip():
            raise ValueError("calibration drift requires a provider and backend")
        physical_qubits = tuple(int(qubit) for qubit in self.physical_qubits)
        qubit_drifts = tuple(self.qubit_drifts)
        gate_drifts = tuple(self.gate_duration_drifts)
        if (
            not physical_qubits
            or len(physical_qubits) != len(set(physical_qubits))
            or any(qubit < 0 for qubit in physical_qubits)
        ):
            raise ValueError("physical_qubits must be unique and non-negative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if any(
            not isinstance(item, TwinQubitCalibrationDrift) for item in qubit_drifts
        ) or any(not isinstance(item, TwinGateDurationDrift) for item in gate_drifts):
            raise TypeError("calibration drift details use Twin drift record types")
        if tuple(item.physical_qubit for item in qubit_drifts) != physical_qubits:
            raise ValueError("qubit drifts must follow the physical mapping order")
        if not isinstance(self.channel_model_changed, bool):
            raise TypeError("channel_model_changed must be a bool")
        object.__setattr__(self, "physical_qubits", physical_qubits)
        object.__setattr__(self, "qubit_drifts", qubit_drifts)
        object.__setattr__(self, "gate_duration_drifts", gate_drifts)

    @property
    def maximum_relative_t1_change(self) -> float:
        return max(abs(item.relative_t1_delta) for item in self.qubit_drifts)

    @property
    def maximum_relative_t2_change(self) -> float:
        return max(abs(item.relative_t2_delta) for item in self.qubit_drifts)

    @property
    def maximum_readout_tv_distance(self) -> float | None:
        values = tuple(
            item.maximum_readout_row_tv_distance
            for item in self.qubit_drifts
            if item.maximum_readout_row_tv_distance is not None
        )
        return None if not values else max(values)

    @property
    def maximum_relative_gate_duration_change(self) -> float | None:
        values = tuple(
            abs(item.relative_duration_delta)
            for item in self.gate_duration_drifts
            if item.relative_duration_delta is not None
        )
        return None if not values else max(values)

    @property
    def has_observed_drift(self) -> bool:
        return (
            self.maximum_relative_t1_change > 0
            or self.maximum_relative_t2_change > 0
            or (self.maximum_readout_tv_distance or 0.0) > 0
            or any(
                item.absolute_duration_delta_seconds > 0
                for item in self.gate_duration_drifts
            )
            or self.channel_model_changed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reference_snapshot_identity": self.reference_snapshot_identity,
            "current_snapshot_identity": self.current_snapshot_identity,
            "provider": self.provider,
            "backend_name": self.backend_name,
            "physical_qubits": list(self.physical_qubits),
            "elapsed_seconds": self.elapsed_seconds,
            "qubit_drifts": [item.to_dict() for item in self.qubit_drifts],
            "gate_duration_drifts": [
                item.to_dict() for item in self.gate_duration_drifts
            ],
            "channel_model_changed": self.channel_model_changed,
            "maximum_relative_t1_change": self.maximum_relative_t1_change,
            "maximum_relative_t2_change": self.maximum_relative_t2_change,
            "maximum_readout_tv_distance": self.maximum_readout_tv_distance,
            "maximum_relative_gate_duration_change": (
                self.maximum_relative_gate_duration_change
            ),
            "has_observed_drift": self.has_observed_drift,
        }


def _duration_map(
    twin: QPUDigitalTwin,
) -> dict[tuple[str, tuple[int, ...] | None], float]:
    profile = twin.noise_model.device_profile
    assert profile is not None
    scale = _TIME_SCALES[profile.time_unit]
    return {
        (item.gate_name, item.wires): item.duration * scale
        for item in profile.gate_durations
    }


def compare_calibrations(
    reference: QPUDigitalTwin,
    current: QPUDigitalTwin,
) -> TwinCalibrationDrift:
    """Compare two frozen Twins for the same QPU mapping without provider I/O."""

    if not isinstance(reference, QPUDigitalTwin) or not isinstance(
        current, QPUDigitalTwin
    ):
        raise TypeError("reference and current must be QPUDigitalTwin objects")
    left = reference.snapshot
    right = current.snapshot
    if (left.provider, left.backend_name, left.physical_qubits) != (
        right.provider,
        right.backend_name,
        right.physical_qubits,
    ):
        raise ValueError("Twin calibrations require the same target and mapping")
    reference_time = datetime.fromisoformat(left.captured_at)
    current_time = datetime.fromisoformat(right.captured_at)
    elapsed = (current_time - reference_time).total_seconds()
    if elapsed < 0:
        raise ValueError("current Twin calibration predates the reference")
    reference_profile = reference.noise_model.device_profile
    current_profile = current.noise_model.device_profile
    if reference_profile is None or current_profile is None:
        raise ValueError("Twin calibration comparison requires device profiles")
    if tuple(item.wire for item in reference_profile.qubits) != tuple(
        item.wire for item in current_profile.qubits
    ):
        raise ValueError("Twin calibration logical-wire structures differ")
    reference_durations = _duration_map(reference)
    current_durations = _duration_map(current)
    if set(reference_durations) != set(current_durations):
        raise ValueError("Twin gate-duration calibration scopes differ")

    reference_scale = _TIME_SCALES[reference_profile.time_unit]
    current_scale = _TIME_SCALES[current_profile.time_unit]
    qubits = []
    for physical_qubit, before, after in zip(
        left.physical_qubits,
        reference_profile.qubits,
        current_profile.qubits,
    ):
        before_t1 = before.t1 * reference_scale
        after_t1 = after.t1 * current_scale
        before_t2 = before.t2 * reference_scale
        after_t2 = after.t2 * current_scale
        t1_delta = _relative_delta(before_t1, after_t1)
        t2_delta = _relative_delta(before_t2, after_t2)
        assert t1_delta is not None and t2_delta is not None
        qubits.append(
            TwinQubitCalibrationDrift(
                physical_qubit=physical_qubit,
                reference_t1_seconds=before_t1,
                current_t1_seconds=after_t1,
                relative_t1_delta=t1_delta,
                reference_t2_seconds=before_t2,
                current_t2_seconds=after_t2,
                relative_t2_delta=t2_delta,
                maximum_readout_row_tv_distance=_readout_distance(
                    before.readout_error, after.readout_error
                ),
            )
        )

    gates = []
    for (gate_name, logical_wires), before in sorted(
        reference_durations.items(),
        key=lambda item: (item[0][0], item[0][1] or ()),
    ):
        after = current_durations[(gate_name, logical_wires)]
        physical_scope = (
            None
            if logical_wires is None
            else tuple(left.physical_qubits[wire] for wire in logical_wires)
        )
        gates.append(
            TwinGateDurationDrift(
                gate_name=gate_name,
                physical_qubits=physical_scope,
                reference_duration_seconds=before,
                current_duration_seconds=after,
                absolute_duration_delta_seconds=abs(after - before),
                relative_duration_delta=_relative_delta(before, after),
            )
        )

    reference_model = reference.noise_model.to_dict()
    current_model = current.noise_model.to_dict()
    channel_changed = (
        reference_model["rules"] != current_model["rules"]
        or reference_model["readout_rules"] != current_model["readout_rules"]
    )
    return TwinCalibrationDrift(
        reference_snapshot_identity=left.identity,
        current_snapshot_identity=right.identity,
        provider=left.provider,
        backend_name=left.backend_name,
        physical_qubits=left.physical_qubits,
        elapsed_seconds=elapsed,
        qubit_drifts=tuple(qubits),
        gate_duration_drifts=tuple(gates),
        channel_model_changed=channel_changed,
    )


__all__ = (
    "compare_calibrations",
    "TwinCalibrationDrift",
    "TwinGateDurationDrift",
    "TwinQubitCalibrationDrift",
)
