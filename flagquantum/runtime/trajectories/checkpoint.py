"""Versioned, atomic checkpoints for trajectory execution."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .ownership import owned_trajectory_ids
from .result import TrajectoryFailure
from .statistics import TensorWelford

TRAJECTORY_CHECKPOINT_VERSION = "flagquantum_trajectory_checkpoint_v1"


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    """Progress and online moments for a resumable trajectory workload."""

    requested_count: int
    base_seed: int
    completed_ids: tuple[int, ...] = ()
    failures: tuple[TrajectoryFailure, ...] = ()
    statistics_state: Mapping[str, Any] | None = None
    noise_model_identity: str | None = None
    circuit_digest: str | None = None
    execution_metadata: Mapping[str, Any] | None = None
    version: str = TRAJECTORY_CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        if self.version != TRAJECTORY_CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported trajectory checkpoint version {self.version!r}"
            )
        if self.requested_count <= 0:
            raise ValueError("requested_count must be positive")
        completed = tuple(int(item) for item in self.completed_ids)
        if completed != tuple(sorted(set(completed))):
            raise ValueError("completed trajectory IDs must be sorted and unique")
        failed_ids = tuple(item.trajectory_id for item in self.failures)
        if len(failed_ids) != len(set(failed_ids)):
            raise ValueError("failed trajectory IDs must be unique")
        all_ids = completed + failed_ids
        if any(item < 0 or item >= self.requested_count for item in all_ids):
            raise ValueError("checkpoint trajectory ID is outside requested range")
        if set(completed).intersection(failed_ids):
            raise ValueError("trajectory cannot be both completed and failed")
        if self.statistics_state is not None:
            statistics = TensorWelford.from_state_dict(self.statistics_state)
            if statistics.count != len(completed):
                raise ValueError(
                    "statistics count must match completed trajectory count"
                )
        elif completed:
            raise ValueError("completed trajectories require statistics state")

    def pending_ids(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        retry_failed: bool = True,
    ) -> tuple[int, ...]:
        """Return rank-owned work not already completed."""

        completed = set(self.completed_ids)
        terminal_failures = {
            item.trajectory_id
            for item in self.failures
            if not retry_failed or not item.retryable
        }
        return tuple(
            item
            for item in owned_trajectory_ids(
                self.requested_count,
                rank=rank,
                world_size=world_size,
            )
            if item not in completed and item not in terminal_failures
        )

    def accumulator(self) -> TensorWelford:
        """Restore the online statistics accumulator."""

        return TensorWelford.from_state_dict(self.statistics_state or {})

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "requested_count": self.requested_count,
            "base_seed": self.base_seed,
            "completed_ids": self.completed_ids,
            "failures": tuple(item.summary() for item in self.failures),
            "statistics_state": dict(self.statistics_state or {}),
            "noise_model_identity": self.noise_model_identity,
            "circuit_digest": self.circuit_digest,
            "execution_metadata": dict(self.execution_metadata or {}),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrajectoryCheckpoint":
        failures = tuple(
            TrajectoryFailure(
                trajectory_id=int(item["trajectory_id"]),
                error_type=str(item["error_type"]),
                message=str(item["message"]),
                retryable=bool(item["retryable"]),
            )
            for item in payload.get("failures", ())
        )
        return cls(
            version=str(payload.get("version", "")),
            requested_count=int(payload["requested_count"]),
            base_seed=int(payload["base_seed"]),
            completed_ids=tuple(int(item) for item in payload.get("completed_ids", ())),
            failures=failures,
            statistics_state=payload.get("statistics_state") or None,
            noise_model_identity=(
                None
                if payload.get("noise_model_identity") is None
                else str(payload["noise_model_identity"])
            ),
            circuit_digest=(
                None
                if payload.get("circuit_digest") is None
                else str(payload["circuit_digest"])
            ),
            execution_metadata=payload.get("execution_metadata") or None,
        )


def save_trajectory_checkpoint(
    checkpoint: TrajectoryCheckpoint,
    path: str | Path,
) -> Path:
    """Atomically persist a safe tensor/primitives-only checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(checkpoint.to_payload(), temporary_path)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def load_trajectory_checkpoint(path: str | Path) -> TrajectoryCheckpoint:
    """Load and validate a trajectory checkpoint without arbitrary unpickling."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory checkpoint payload must be a mapping")
    return TrajectoryCheckpoint.from_payload(payload)


__all__ = (
    "TRAJECTORY_CHECKPOINT_VERSION",
    "TrajectoryCheckpoint",
    "load_trajectory_checkpoint",
    "save_trajectory_checkpoint",
)
