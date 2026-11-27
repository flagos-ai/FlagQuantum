"""Backend-neutral rank placement counts and environment resolution."""

from __future__ import annotations

import os


def resolve_local_world_size(world_size: int, *, subgroup: bool = False) -> int:
    """How many ranks of `world_size` share one host.

    One definition, because every distributed result reports `local_world_size`
    and derives `node_count` and the intra-node/inter-node communication split
    from it: a component that read the environment differently could report a
    two-node run as a single node while the ranks still exchanged across the
    network.

    `LOCAL_WORLD_SIZE` and `NPROC_PER_NODE` describe the world torchrun was
    given, so a caller running a subgroup `process_group` passes `subgroup=True`
    and gets a single-host placement instead of a topology the environment
    cannot describe.
    """
    world_size = max(1, int(world_size))
    if subgroup:
        return world_size
    raw = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NPROC_PER_NODE")
    if raw is None:
        return world_size
    return max(1, min(world_size, int(raw)))


def resolve_node_count(world_size: int, local_world_size: int) -> int:
    """How many hosts `world_size` ranks span at this placement."""

    return max(
        1,
        (max(1, int(world_size)) + max(1, int(local_world_size)) - 1)
        // max(1, int(local_world_size)),
    )
