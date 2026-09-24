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

    @property
    def device(self) -> torch.device:
        """Device shared by the retained trajectory states."""

        self._require_retained_trajectories()
        return torch.device(self.trajectories[0].device)

    @property
    def n_wires(self) -> int:
        """Qubit count shared by the retained trajectory states."""

        self._require_retained_trajectories()
        return int(self.trajectories[0].n_wires)

    @property
    def bsz(self) -> int:
        """Batch size shared by the retained trajectory states."""

        self._require_retained_trajectories()
        return int(self.trajectories[0].bsz)

    def _require_retained_trajectories(self) -> None:
        if not self.trajectories:
            raise RuntimeError(
                "noisy MPS sampling requires retained trajectory states; "
                "run with retain_trajectories=True"
            )

    def sample(
        self,
        shots: int = 1,
        *,
        generator: torch.Generator | None = None,
        format: str = "bits",
    ) -> torch.Tensor:
        """Sample the empirical mixture of retained noisy MPS trajectories.

        Every requested shot first selects one retained noise trajectory
        uniformly and then performs a computational-basis measurement of that
        trajectory.  This keeps the measurement memory linear in the qubit
        count and avoids materialising a dense state or density matrix.
        """

        if format not in {"bits", "index"}:
            raise ValueError("sample format must be 'bits' or 'index'.")
        if type(shots) is not int or shots <= 0:
            raise ValueError("sample shots must be a positive integer")
        self._require_retained_trajectories()

        first = self.trajectories[0]
        n_wires = int(first.n_wires)
        bsz = int(first.bsz)
        device = torch.device(first.device)
        for state in self.trajectories[1:]:
            if (
                int(state.n_wires) != n_wires
                or int(state.bsz) != bsz
                or torch.device(state.device) != device
            ):
                raise ValueError("retained noisy MPS trajectories are incompatible")

        selected = torch.randint(
            len(self.trajectories),
            (bsz, shots),
            device=device,
            generator=generator,
        )
        samples = torch.empty(
            bsz,
            shots,
            n_wires,
            dtype=torch.int64,
            device=device,
        )
        for trajectory_index, state in enumerate(self.trajectories):
            per_batch = [
                torch.nonzero(
                    selected[row] == trajectory_index, as_tuple=False
                ).flatten()
                for row in range(bsz)
            ]
            draw_count = max(
                (int(positions.numel()) for positions in per_batch), default=0
            )
            if draw_count == 0:
                continue
            draws = state.sample(draw_count, generator=generator, format="bits")
            for row, positions in enumerate(per_batch):
                if positions.numel():
                    samples[row, positions] = draws[row, : positions.numel()]

        if format == "bits":
            return samples
        if n_wires > 63:
            raise ValueError(
                "integer-index noisy MPS samples support at most 63 qubits; "
                "request format='bits' for wider circuits"
            )
        shifts = torch.arange(n_wires - 1, -1, -1, device=device)
        return torch.sum(samples << shifts, dim=-1)

    def sampling_summary(self) -> dict[str, Any]:
        """Describe the approximation used for shot sampling."""

        trajectory_errors = tuple(
            float(sum(state.truncation_errors)) for state in self.trajectories
        )
        discarded_weights = tuple(
            float(error)
            for state in self.trajectories
            for error in state.truncation_errors
        )
        first_config = self.trajectories[0].config if self.trajectories else None
        return {
            "sampling_semantics": "empirical_noisy_trajectory_mixture",
            "trajectory_count": self.n_trajectories,
            "retained_trajectory_count": len(self.trajectories),
            "noise_model_identity": self.noise_model_identity,
            "observed_max_bond": max(
                (int(state.max_bond) for state in self.trajectories),
                default=None,
            ),
            "configured_max_bond": (
                None if first_config is None else first_config.max_bond
            ),
            "configured_cutoff": (
                None if first_config is None else float(first_config.cutoff)
            ),
            "max_trajectory_truncation_error": max(
                trajectory_errors,
                default=None,
            ),
            "max_discarded_weight": max(discarded_weights, default=None),
        }

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
