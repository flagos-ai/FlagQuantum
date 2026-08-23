"""Versioned hardware-noise calibration specifications."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .model import ReadoutError


@dataclass(frozen=True)
class QubitNoiseCalibration:
    wire: int
    t1: float
    t2: float
    excited_population: float = 0.0
    readout_error: ReadoutError | None = None

    def __post_init__(self) -> None:
        if self.wire < 0:
            raise ValueError("calibration wire must be non-negative")
        if self.t1 <= 0 or self.t2 <= 0 or self.t2 > 2 * self.t1:
            raise ValueError("calibration requires t1 > 0 and 0 < t2 <= 2*t1")
        if not 0 <= self.excited_population <= 1:
            raise ValueError("excited_population must be between 0 and 1")


@dataclass(frozen=True)
class GateDuration:
    gate_name: str
    duration: float
    wires: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        name = self.gate_name.strip().lower()
        if not name:
            raise ValueError("gate duration requires a gate name")
        if self.duration < 0:
            raise ValueError("gate duration must be non-negative")
        object.__setattr__(self, "gate_name", name)
        if self.wires is not None:
            wires = tuple(int(wire) for wire in self.wires)
            if (
                not wires
                or any(wire < 0 for wire in wires)
                or len(wires) != len(set(wires))
            ):
                raise ValueError("scoped gate duration wires must be non-negative")
            object.__setattr__(self, "wires", wires)


@dataclass(frozen=True)
class DeviceNoiseProfile:
    """Device calibration with explicit provenance and timing units."""

    qubits: tuple[QubitNoiseCalibration, ...]
    gate_durations: tuple[GateDuration, ...]
    source: str
    captured_at: str
    time_unit: str = "ns"
    schema: str = "flagquantum.device_noise_profile.v1"

    def __post_init__(self) -> None:
        if self.schema != "flagquantum.device_noise_profile.v1":
            raise ValueError("unsupported device noise profile schema")
        if not self.source.strip() or not self.captured_at.strip():
            raise ValueError(
                "device profile requires source and captured_at provenance"
            )
        try:
            captured = datetime.fromisoformat(self.captured_at)
        except ValueError as error:
            raise ValueError("captured_at must be an ISO-8601 timestamp") from error
        if captured.tzinfo is None:
            raise ValueError("captured_at must include a timezone")
        if self.time_unit not in {"s", "ms", "us", "ns"}:
            raise ValueError("time_unit must be one of s, ms, us, or ns")
        if not self.qubits or not self.gate_durations:
            raise ValueError("device profile requires qubit and gate-duration data")
        wires = tuple(item.wire for item in self.qubits)
        if len(wires) != len(set(wires)):
            raise ValueError("device profile cannot repeat qubit calibrations")
        duration_keys = tuple(
            (item.gate_name, item.wires) for item in self.gate_durations
        )
        if len(duration_keys) != len(set(duration_keys)):
            raise ValueError("device profile cannot repeat gate-duration scopes")

    def calibration_for(self, wire: int) -> QubitNoiseCalibration:
        match = next((item for item in self.qubits if item.wire == int(wire)), None)
        if match is None:
            raise ValueError(f"device profile has no calibration for wire {wire}")
        return match

    def duration_for(self, gate_name: str, wires: tuple[int, ...]) -> float:
        name = gate_name.lower()
        scoped = next(
            (
                item
                for item in self.gate_durations
                if item.gate_name == name and item.wires == tuple(wires)
            ),
            None,
        )
        generic = next(
            (
                item
                for item in self.gate_durations
                if item.gate_name == name and item.wires is None
            ),
            None,
        )
        selected = scoped or generic
        if selected is None:
            raise ValueError(
                f"device profile has no duration for gate {name!r} on wires {wires}"
            )
        return selected.duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "captured_at": self.captured_at,
            "time_unit": self.time_unit,
            "qubits": [
                {
                    "wire": item.wire,
                    "t1": item.t1,
                    "t2": item.t2,
                    "excited_population": item.excited_population,
                    "readout_error": (
                        None
                        if item.readout_error is None
                        else [list(row) for row in item.readout_error.probabilities]
                    ),
                }
                for item in self.qubits
            ],
            "gate_durations": [
                {
                    "gate_name": item.gate_name,
                    "duration": item.duration,
                    "wires": None if item.wires is None else list(item.wires),
                }
                for item in self.gate_durations
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeviceNoiseProfile":
        qubits = tuple(
            QubitNoiseCalibration(
                wire=int(item["wire"]),
                t1=float(item["t1"]),
                t2=float(item["t2"]),
                excited_population=float(item.get("excited_population", 0.0)),
                readout_error=(
                    None
                    if item.get("readout_error") is None
                    else ReadoutError(
                        tuple(
                            tuple(float(value) for value in row)
                            for row in item["readout_error"]
                        )
                    )
                ),
            )
            for item in payload.get("qubits", ())
        )
        durations = tuple(
            GateDuration(
                gate_name=str(item["gate_name"]),
                duration=float(item["duration"]),
                wires=(
                    None
                    if item.get("wires") is None
                    else tuple(int(wire) for wire in item["wires"])
                ),
            )
            for item in payload.get("gate_durations", ())
        )
        return cls(
            qubits=qubits,
            gate_durations=durations,
            source=str(payload.get("source", "")),
            captured_at=str(payload.get("captured_at", "")),
            time_unit=str(payload.get("time_unit", "")),
            schema=str(payload.get("schema", "")),
        )

    @property
    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


__all__ = (
    "DeviceNoiseProfile",
    "GateDuration",
    "QubitNoiseCalibration",
)
