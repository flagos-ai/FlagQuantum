"""Stable global trajectory ownership independent of execution order."""

from __future__ import annotations


def owned_trajectory_ids(
    count: int,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[int, ...]:
    """Return round-robin global trajectory IDs owned by one rank."""

    count = int(count)
    rank = int(rank)
    world_size = int(world_size)
    if count < 0:
        raise ValueError("trajectory count must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    return tuple(range(rank, count, world_size))


__all__ = ("owned_trajectory_ids",)
