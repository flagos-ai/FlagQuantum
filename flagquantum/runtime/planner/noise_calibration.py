"""Versioned performance calibration for Runtime noisy-backend selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

NOISE_SELECTOR_CALIBRATION_SCHEMA = "flagquantum.noise_selector_calibration.v1"


@dataclass(frozen=True)
class NoiseSelectorCalibrationRecord:
    n_wires: int
    depth: int
    channel_count: int
    mode: str
    median_seconds: float
    executed_trajectories: int | None
    max_cuda_peak_allocated_bytes: int
    circuit_digest: str
    noise_model_identity: str
    noise_kind: str

    def __post_init__(self) -> None:
        if self.n_wires <= 0 or self.depth <= 0 or self.channel_count < 0:
            raise ValueError("invalid calibration workload dimensions")
        if self.mode not in {"density_matrix", "noisy_statevector", "noisy_mps"}:
            raise ValueError(f"unsupported calibrated noise mode {self.mode!r}")
        if self.median_seconds <= 0 or self.max_cuda_peak_allocated_bytes < 0:
            raise ValueError("calibration timing and memory must be non-negative")
        if self.mode == "density_matrix" and self.executed_trajectories is not None:
            raise ValueError("density calibration cannot have a trajectory count")
        if self.mode != "density_matrix" and (
            self.executed_trajectories is None or self.executed_trajectories <= 0
        ):
            raise ValueError("trajectory calibration requires a positive count")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NoiseSelectorCalibrationRecord":
        executed = payload.get("executed_trajectories")
        return cls(
            n_wires=int(payload["n_wires"]),
            depth=int(payload["depth"]),
            channel_count=int(payload["channel_count"]),
            mode=str(payload["mode"]),
            median_seconds=float(payload["median_seconds"]),
            executed_trajectories=None if executed is None else int(executed),
            max_cuda_peak_allocated_bytes=int(payload["max_cuda_peak_allocated_bytes"]),
            circuit_digest=str(payload["circuit_digest"]),
            noise_model_identity=str(payload["noise_model_identity"]),
            noise_kind=str(payload.get("noise_kind", "unspecified")),
        )


@dataclass(frozen=True)
class NoiseSelectorCalibration:
    """Validated device-specific noisy execution measurements."""

    device_name: str
    torch_version: str
    trajectory_batch_size: int
    requested_trajectories: int
    records: tuple[NoiseSelectorCalibrationRecord, ...]
    world_size: int = 1
    schema: str = NOISE_SELECTOR_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NOISE_SELECTOR_CALIBRATION_SCHEMA:
            raise ValueError(f"unsupported noise selector calibration {self.schema!r}")
        if not self.device_name or not self.torch_version:
            raise ValueError("calibration requires device and Torch provenance")
        if (
            self.trajectory_batch_size <= 0
            or self.requested_trajectories <= 0
            or self.world_size <= 0
        ):
            raise ValueError("calibration trajectory controls must be positive")
        if not self.records:
            raise ValueError("calibration requires at least one record")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NoiseSelectorCalibration":
        return cls(
            schema=str(payload.get("schema", "")),
            device_name=str(payload.get("device_name", "")),
            torch_version=str(payload.get("torch_version", "")),
            trajectory_batch_size=int(payload.get("trajectory_batch_size", 0)),
            requested_trajectories=int(payload.get("requested_trajectories", 0)),
            world_size=int(payload.get("world_size", 1)),
            records=tuple(
                NoiseSelectorCalibrationRecord.from_dict(item)
                for item in payload.get("records", ())
            ),
        )

    def estimate_seconds(
        self,
        *,
        mode: str,
        n_wires: int,
        channel_count: int,
        circuit_digest: str,
        noise_model_identity: str,
        trajectories: int | None,
        trajectory_batch_size: int,
        world_size: int,
    ) -> tuple[float, NoiseSelectorCalibrationRecord] | None:
        """Return an estimate only for an exactly covered workload shape."""

        if world_size != self.world_size:
            return None

        matches = tuple(
            item
            for item in self.records
            if item.mode == mode
            and item.n_wires == n_wires
            and item.channel_count == channel_count
            and item.circuit_digest == circuit_digest
            and item.noise_model_identity == noise_model_identity
        )
        if not matches:
            return None
        record = matches[0]
        estimate = record.median_seconds
        if record.executed_trajectories is not None:
            if trajectories is None:
                return None
            estimate *= trajectories / record.executed_trajectories
            # Batch-size extrapolation is intentionally disallowed.
            if trajectory_batch_size != self.trajectory_batch_size:
                return None
        return estimate, record


def load_noise_selector_calibration(
    source: str | Path | Mapping[str, Any] | NoiseSelectorCalibration,
) -> NoiseSelectorCalibration:
    """Load a calibration from an object, mapping, or JSON file."""

    if isinstance(source, NoiseSelectorCalibration):
        return source
    if isinstance(source, Mapping):
        return NoiseSelectorCalibration.from_dict(source)
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("noise selector calibration JSON must contain an object")
    return NoiseSelectorCalibration.from_dict(payload)


__all__ = (
    "NOISE_SELECTOR_CALIBRATION_SCHEMA",
    "NoiseSelectorCalibration",
    "NoiseSelectorCalibrationRecord",
    "load_noise_selector_calibration",
)
