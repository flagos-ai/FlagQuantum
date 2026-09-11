"""Backend-neutral result contracts for trajectory aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class MPSMonteCarloResult:
    """Aggregated result of one noisy MPS trajectory run."""

    trajectories: tuple[Any, ...]
    expectation_z_mean: torch.Tensor
    expectation_z_variance: torch.Tensor
    n_trajectories: int
    trajectory_ids: tuple[int, ...] = ()
    trajectory_seeds: tuple[int, ...] = ()
    statistics: TrajectoryStatistics | None = None
    requested_trajectories: int | None = None
    retained_trajectory_ids: tuple[int, ...] = ()
    failures: tuple[TrajectoryFailure, ...] = ()
    rank: int = 0
    world_size: int = 1
    target_standard_error: float | None = None
    min_trajectories: int = 1
    converged: bool = False
    stopped_early: bool = False
    noise_model_identity: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "state_mode": "mps_trajectory",
            "n_trajectories": self.n_trajectories,
            "requested_trajectories": (
                self.n_trajectories
                if self.requested_trajectories is None
                else self.requested_trajectories
            ),
            "trajectory_ids": self.trajectory_ids,
            "trajectory_seeds": self.trajectory_seeds,
            "retained_trajectory_ids": self.retained_trajectory_ids,
            "retained_trajectory_count": len(self.trajectories),
            "failures": tuple(item.summary() for item in self.failures),
            "rank": self.rank,
            "world_size": self.world_size,
            "target_standard_error": self.target_standard_error,
            "min_trajectories": self.min_trajectories,
            "converged": self.converged,
            "stopped_early": self.stopped_early,
            "noise_model_identity": self.noise_model_identity,
            "expectation_z_mean": self.expectation_z_mean,
            "expectation_z_variance": self.expectation_z_variance,
            "expectation_z_standard_error": (
                None if self.statistics is None else self.statistics.standard_error
            ),
        }


__all__ = ("MPSMonteCarloResult", "TrajectoryFailure", "TrajectoryStatistics")
