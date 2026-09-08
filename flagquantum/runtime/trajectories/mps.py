"""MPS trajectory scheduling, recovery, and result aggregation."""

from __future__ import annotations

import os
from typing import Any, Callable

import torch

from .checkpoint import (
    TrajectoryCheckpoint,
    load_trajectory_checkpoint,
    save_trajectory_checkpoint,
)
from .distributed import merge_trajectory_statistics
from .ownership import owned_trajectory_ids
from .result import MPSMonteCarloResult, TrajectoryFailure
from .rng import derive_trajectory_seed, trajectory_generator
from .statistics import TensorWelford


def run_single_mps_trajectory_runtime(
    lowered_ir: Any,
    *,
    generator: torch.Generator | None,
    seed: int | None,
    trajectory_id: int,
    device: torch.device | str,
    trajectory_executor: Callable[..., Any],
    executor_options: dict[str, Any],
) -> Any:
    """Resolve one trajectory random stream and invoke Simulation."""

    if generator is not None and seed is not None:
        raise ValueError("pass either generator or seed, not both")
    if seed is not None:
        generator = trajectory_generator(seed, trajectory_id, device=device)
    return trajectory_executor(
        lowered_ir,
        generator=generator,
        **executor_options,
    )


def run_noisy_mps_runtime(
    circuit_or_ir: Any,
    noise_model: Any | None = None,
    *,
    trajectories: int = 32,
    min_trajectories: int = 1,
    target_standard_error: float | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    resume: bool = False,
    checkpoint_interval: int = 1,
    max_trajectories_per_run: int | None = None,
    retain_trajectories: bool = True,
    retry_failed: bool = True,
    continue_on_error: bool = False,
    rank: int = 0,
    world_size: int = 1,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    trajectory_executor: Callable[..., Any],
) -> MPSMonteCarloResult:
    """Run multiple noisy MPS trajectories and aggregate Z expectations."""

    if int(trajectories) <= 0:
        raise ValueError("trajectories must be positive")
    rank = int(rank)
    world_size = int(world_size)
    min_trajectories = int(min_trajectories)
    if min_trajectories <= 0 or min_trajectories > int(trajectories):
        raise ValueError("min_trajectories must be in [1, trajectories]")
    if target_standard_error is not None:
        target_standard_error = float(target_standard_error)
        if target_standard_error <= 0:
            raise ValueError("target_standard_error must be positive")
        if world_size != 1:
            raise ValueError(
                "adaptive trajectory stopping currently requires world_size=1; "
                "distributed convergence needs a collective statistics reduction"
            )
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if max_trajectories_per_run is not None and max_trajectories_per_run <= 0:
        raise ValueError("max_trajectories_per_run must be positive")
    owned_trajectory_ids(0, rank=rank, world_size=world_size)
    if world_size > int(trajectories):
        raise ValueError("world_size cannot exceed trajectory count")
    if generator is not None and seed is not None:
        raise ValueError("pass either generator or seed, not both")
    if checkpoint_path is not None and seed is None:
        raise ValueError("checkpointed trajectory execution requires seed")
    if resume and checkpoint_path is None:
        raise ValueError("resume requires checkpoint_path")
    if resume and retain_trajectories:
        raise ValueError(
            "resumed execution requires retain_trajectories=False because "
            "trajectory states are not stored in the statistics checkpoint"
        )
    if checkpoint_path is not None and world_size > 1:
        checkpoint_template = os.fspath(checkpoint_path)
        if "{rank}" not in checkpoint_template:
            raise ValueError(
                "distributed trajectory checkpoint_path must contain '{rank}'"
            )
        checkpoint_path = checkpoint_template.format(rank=rank)

    states = []
    retained_ids: list[int] = []
    failures: list[TrajectoryFailure] = []
    completed_ids: list[int] = []
    noise_model_identity = getattr(noise_model, "identity", None)
    if resume:
        checkpoint = load_trajectory_checkpoint(checkpoint_path)
        if checkpoint.requested_count != int(trajectories):
            raise ValueError("checkpoint requested trajectory count does not match")
        if checkpoint.base_seed != int(seed):
            raise ValueError("checkpoint base seed does not match")
        if checkpoint.noise_model_identity != noise_model_identity:
            raise ValueError("checkpoint noise model identity does not match")
        accumulator = checkpoint.accumulator()
        completed_ids.extend(checkpoint.completed_ids)
        failures.extend(checkpoint.failures)
        owned_ids = set(
            owned_trajectory_ids(
                int(trajectories),
                rank=rank,
                world_size=world_size,
            )
        )
        if any(item not in owned_ids for item in checkpoint.completed_ids):
            raise ValueError("checkpoint contains trajectories owned by another rank")
        pending_ids = checkpoint.pending_ids(
            rank=rank,
            world_size=world_size,
            retry_failed=retry_failed,
        )
    else:
        accumulator = TensorWelford()
        pending_ids = owned_trajectory_ids(
            int(trajectories),
            rank=rank,
            world_size=world_size,
        )
    if max_trajectories_per_run is not None:
        pending_ids = pending_ids[: int(max_trajectories_per_run)]
    effective_device = (
        device if not hasattr(circuit_or_ir, "device") else circuit_or_ir.device
    )
    completed_since_checkpoint = 0

    def target_reached() -> bool:
        if target_standard_error is None or accumulator.count < min_trajectories:
            return False
        standard_error = accumulator.finalize().standard_error
        return bool(torch.all(standard_error <= target_standard_error))

    def save_progress() -> None:
        if checkpoint_path is None:
            return
        save_trajectory_checkpoint(
            TrajectoryCheckpoint(
                requested_count=int(trajectories),
                base_seed=int(seed),
                completed_ids=tuple(sorted(completed_ids)),
                failures=tuple(sorted(failures, key=lambda item: item.trajectory_id)),
                statistics_state=(
                    accumulator.state_dict() if accumulator.count else None
                ),
                noise_model_identity=noise_model_identity,
            ),
            checkpoint_path,
        )

    for trajectory_id in () if target_reached() else pending_ids:
        trajectory_rng = (
            trajectory_generator(seed, trajectory_id, device=effective_device)
            if seed is not None
            else generator
        )
        try:
            state = trajectory_executor(
                circuit_or_ir,
                noise_model,
                generator=trajectory_rng,
                bsz=bsz,
                device=device,
                dtype=dtype,
                max_bond=max_bond,
                cutoff=cutoff,
            )
        except Exception as error:
            if not continue_on_error:
                save_progress()
                raise
            failures = [
                item for item in failures if item.trajectory_id != trajectory_id
            ]
            failures.append(
                TrajectoryFailure(
                    trajectory_id=trajectory_id,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            save_progress()
            continue
        completed_ids.append(trajectory_id)
        failures = [item for item in failures if item.trajectory_id != trajectory_id]
        expectation_z = state.expectation_z()
        if noise_model is not None and hasattr(
            noise_model, "apply_readout_expectation_z"
        ):
            expectation_z = noise_model.apply_readout_expectation_z(expectation_z)
        accumulator.update(expectation_z)
        if retain_trajectories:
            states.append(state)
            retained_ids.append(trajectory_id)
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= checkpoint_interval:
            save_progress()
            completed_since_checkpoint = 0
        if target_reached():
            break
    save_progress()
    if accumulator.count == 0:
        raise RuntimeError("no noisy MPS trajectories completed successfully")
    statistics = accumulator.finalize()
    converged = target_reached()
    trajectory_ids = tuple(sorted(completed_ids))
    trajectory_seeds = (
        tuple(derive_trajectory_seed(seed, item) for item in trajectory_ids)
        if seed is not None
        else ()
    )
    return MPSMonteCarloResult(
        trajectories=tuple(states),
        expectation_z_mean=statistics.mean,
        expectation_z_variance=statistics.variance,
        n_trajectories=statistics.count,
        trajectory_ids=trajectory_ids,
        trajectory_seeds=trajectory_seeds,
        statistics=statistics,
        requested_trajectories=int(trajectories),
        retained_trajectory_ids=tuple(retained_ids),
        failures=tuple(sorted(failures, key=lambda item: item.trajectory_id)),
        rank=rank,
        world_size=world_size,
        target_standard_error=target_standard_error,
        min_trajectories=min_trajectories,
        converged=converged,
        stopped_early=converged and statistics.count < int(trajectories),
        noise_model_identity=noise_model_identity,
    )


def merge_noisy_mps_results_runtime(
    results: Any,
    *,
    require_complete: bool = True,
) -> MPSMonteCarloResult:
    """Merge disjoint rank-local noisy MPS results in rank order."""

    items = tuple(results)
    if not items:
        raise ValueError("at least one noisy MPS result is required")
    requested = {
        (
            item.n_trajectories
            if item.requested_trajectories is None
            else item.requested_trajectories
        )
        for item in items
    }
    world_sizes = {item.world_size for item in items}
    noise_model_identities = {item.noise_model_identity for item in items}
    ranks = [item.rank for item in items]
    if len(requested) != 1 or len(world_sizes) != 1 or len(noise_model_identities) != 1:
        raise ValueError("rank-local noisy MPS results have incompatible plans")
    world_size = world_sizes.pop()
    if len(ranks) != len(set(ranks)):
        raise ValueError("rank-local noisy MPS results contain duplicate ranks")
    if any(item.statistics is None for item in items):
        raise ValueError("rank-local noisy MPS result is missing statistics")

    trajectory_ids = tuple(
        sorted(item for result in items for item in result.trajectory_ids)
    )
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("rank-local noisy MPS results contain duplicate trajectories")
    requested_count = requested.pop()
    failures = tuple(
        sorted(
            (failure for result in items for failure in result.failures),
            key=lambda failure: failure.trajectory_id,
        )
    )
    if require_complete and (
        trajectory_ids != tuple(range(requested_count)) or failures
    ):
        raise ValueError("distributed noisy MPS result is incomplete")

    statistics = merge_trajectory_statistics(
        item.statistics for item in sorted(items, key=lambda result: result.rank)
    )
    retained = sorted(
        (trajectory_id, state)
        for result in items
        for trajectory_id, state in zip(
            result.retained_trajectory_ids,
            result.trajectories,
            strict=True,
        )
    )
    seed_by_id = {
        trajectory_id: seed
        for result in items
        for trajectory_id, seed in zip(
            result.trajectory_ids,
            result.trajectory_seeds,
            strict=True,
        )
    }
    return MPSMonteCarloResult(
        trajectories=tuple(state for _, state in retained),
        expectation_z_mean=statistics.mean,
        expectation_z_variance=statistics.variance,
        n_trajectories=statistics.count,
        trajectory_ids=trajectory_ids,
        trajectory_seeds=tuple(seed_by_id[item] for item in trajectory_ids),
        statistics=statistics,
        requested_trajectories=requested_count,
        retained_trajectory_ids=tuple(item for item, _ in retained),
        failures=failures,
        rank=0,
        world_size=world_size,
        noise_model_identity=noise_model_identities.pop(),
    )
