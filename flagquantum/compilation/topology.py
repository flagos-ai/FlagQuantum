"""Rank placement and ownership planning helpers."""

from __future__ import annotations

from typing import Any


def rank_ranges(n_items: int, world_size: int) -> tuple[tuple[int, int], ...]:
    world_size = max(1, int(world_size))
    n_items = max(0, int(n_items))
    return tuple(
        (
            rank * n_items // world_size,
            (rank + 1) * n_items // world_size,
        )
        for rank in range(world_size)
    )


def rank_placement(
    world_size: int, local_world_size: int
) -> tuple[dict[str, int], ...]:
    return tuple(
        {
            "rank": rank,
            "node_id": rank // max(1, int(local_world_size)),
            "local_rank": rank % max(1, int(local_world_size)),
        }
        for rank in range(max(1, int(world_size)))
    )


def rank_ownership(
    *,
    mode: str,
    state: str,
    n_wires: int,
    world_size: int,
    local_world_size: int,
) -> tuple[dict[str, Any], ...]:
    if world_size <= 1:
        return ()
    placement = rank_placement(world_size, local_world_size)
    if state == "statevector" and mode in {
        "distributed_statevector",
        "jax_sharded_statevector",
    }:
        ranges = rank_ranges(2 ** int(n_wires), world_size)
        return tuple(
            dict(
                item,
                amplitude_start=start,
                amplitude_end=end,
                owned_amplitudes=end - start,
            )
            for item, (start, end) in zip(placement, ranges)
        )
    if state == "mps" and ("sharded" in mode or mode == "distributed_mps"):
        ranges = rank_ranges(n_wires, world_size)
        return tuple(
            dict(
                item,
                wire_start=start,
                wire_end=end,
                owned_wires=tuple(range(start, end)),
            )
            for item, (start, end) in zip(placement, ranges)
        )
    if state == "tensor_network" and ("tensor_network" in mode or mode.endswith("_tn")):
        ranges = rank_ranges(max(world_size, n_wires), world_size)
        return tuple(
            dict(
                item,
                slice_start=start,
                slice_end=end,
                owned_slice_count=end - start,
            )
            for item, (start, end) in zip(placement, ranges)
        )
    return tuple(dict(item, replica="full_workload") for item in placement)


__all__ = ["rank_ownership", "rank_placement", "rank_ranges"]
