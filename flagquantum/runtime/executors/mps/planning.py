"""MPS shard ownership and instruction placement helpers."""

from __future__ import annotations

from typing import Sequence

from ....core.ir import Instruction
from .distributed_state import DistributedBoundarySync, DistributedShardPlan


def _rank_for_wire(wire: int, shards: Sequence[DistributedShardPlan]) -> int:
    for shard in shards:
        if int(wire) in shard.wires:
            return shard.rank
    return 0


def _instruction_owner(
    instruction: Instruction, shards: Sequence[DistributedShardPlan]
) -> int:
    if not instruction.wires:
        return 0
    return _rank_for_wire(min(int(wire) for wire in instruction.wires), shards)


def _instruction_is_site_local(
    instruction: Instruction, shards: Sequence[DistributedShardPlan]
) -> bool:
    wires = tuple(int(wire) for wire in instruction.wires)
    if instruction.metadata.get("is_channel"):
        return False
    if len(wires) == 1:
        return True
    if len(wires) == 2 and abs(wires[0] - wires[1]) == 1:
        return _rank_for_wire(wires[0], shards) == _rank_for_wire(wires[1], shards)
    return False


def _instruction_is_boundary_local(
    instruction: Instruction, shards: Sequence[DistributedShardPlan]
) -> bool:
    wires = tuple(int(wire) for wire in instruction.wires)
    if instruction.metadata.get("is_channel"):
        return False
    if len(wires) != 2 or abs(wires[0] - wires[1]) != 1:
        return False
    return _rank_for_wire(wires[0], shards) != _rank_for_wire(wires[1], shards)


def _boundary_sync_record(
    instruction: Instruction,
    shards: Sequence[DistributedShardPlan],
) -> DistributedBoundarySync:
    wires = tuple(sorted(int(wire) for wire in instruction.wires))
    if len(wires) != 2:
        raise ValueError("Boundary MPS sync requires a two-wire instruction.")
    left_wire, right_wire = wires
    left_rank = _rank_for_wire(left_wire, shards)
    right_rank = _rank_for_wire(right_wire, shards)
    return DistributedBoundarySync(
        left_wire=left_wire,
        right_wire=right_wire,
        left_rank=left_rank,
        right_rank=right_rank,
        owner_rank=left_rank,
    )
