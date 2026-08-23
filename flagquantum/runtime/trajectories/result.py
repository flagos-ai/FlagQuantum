"""Backend-neutral result contracts for trajectory aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrajectoryStatistics:
    """Population statistics accumulated across successful trajectories."""

    count: int
    mean: torch.Tensor
    variance: torch.Tensor
    standard_error: torch.Tensor

    def summary(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "standard_error": self.standard_error,
        }


@dataclass(frozen=True)
class TrajectoryFailure:
    """Serializable failure record for one global trajectory."""

    trajectory_id: int
    error_type: str
    message: str
    retryable: bool = True

    def __post_init__(self) -> None:
        if self.trajectory_id < 0:
            raise ValueError("trajectory_id must be non-negative")

    def summary(self) -> dict[str, object]:
        return {
            "trajectory_id": self.trajectory_id,
            "error_type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
        }


__all__ = ("TrajectoryFailure", "TrajectoryStatistics")
