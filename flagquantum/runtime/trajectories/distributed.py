"""Pure aggregation helpers for rank-local trajectory checkpoints."""

from __future__ import annotations

from collections.abc import Iterable

from .checkpoint import TrajectoryCheckpoint
from .result import TrajectoryStatistics
from .statistics import TensorWelford


def merge_trajectory_checkpoints(
    checkpoints: Iterable[TrajectoryCheckpoint],
) -> TrajectoryCheckpoint:
    """Merge disjoint rank-local progress into one resumable checkpoint."""

    items = tuple(checkpoints)
    if not items:
        raise ValueError("at least one trajectory checkpoint is required")
    requested_count = items[0].requested_count
    base_seed = items[0].base_seed
    noise_model_identity = items[0].noise_model_identity
    circuit_digest = items[0].circuit_digest
    execution_metadata = items[0].execution_metadata
    completed: list[int] = []
    failures = []
    accumulator = TensorWelford()
    for checkpoint in items:
        if checkpoint.requested_count != requested_count:
            raise ValueError("trajectory checkpoints have different requested counts")
        if checkpoint.base_seed != base_seed:
            raise ValueError("trajectory checkpoints have different base seeds")
        if (
            checkpoint.noise_model_identity != noise_model_identity
            or checkpoint.circuit_digest != circuit_digest
            or checkpoint.execution_metadata != execution_metadata
        ):
            raise ValueError(
                "trajectory checkpoints have incompatible execution identity"
            )
        completed.extend(checkpoint.completed_ids)
        failures.extend(checkpoint.failures)
        accumulator.merge(checkpoint.accumulator())

    if len(completed) != len(set(completed)):
        raise ValueError("trajectory checkpoints contain duplicate completed IDs")
    failed_ids = [item.trajectory_id for item in failures]
    if len(failed_ids) != len(set(failed_ids)):
        raise ValueError("trajectory checkpoints contain duplicate failed IDs")

    completed_ids = tuple(sorted(completed))
    completed_set = set(completed_ids)
    merged_failures = tuple(
        sorted(
            (item for item in failures if item.trajectory_id not in completed_set),
            key=lambda item: item.trajectory_id,
        )
    )
    return TrajectoryCheckpoint(
        requested_count=requested_count,
        base_seed=base_seed,
        completed_ids=completed_ids,
        failures=merged_failures,
        statistics_state=(accumulator.state_dict() if accumulator.count else None),
        noise_model_identity=noise_model_identity,
        circuit_digest=circuit_digest,
        execution_metadata=execution_metadata,
    )


def merge_trajectory_statistics(
    statistics: Iterable[TrajectoryStatistics],
) -> TrajectoryStatistics:
    """Merge finalized rank-local population moments deterministically."""

    items = tuple(statistics)
    if not items:
        raise ValueError("at least one trajectory statistic is required")
    accumulator = TensorWelford()
    for item in items:
        accumulator.merge(TensorWelford.from_statistics(item))
    return accumulator.finalize()


__all__ = ("merge_trajectory_checkpoints", "merge_trajectory_statistics")
