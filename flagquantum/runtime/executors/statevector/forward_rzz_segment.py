"""Native CPU planning for contiguous forward RZZ segments."""

from __future__ import annotations

from typing import Any

import torch

from ....core.ir import Instruction
from ....simulation.native_cpu import (
    fused_rotation_block_forward_,
    fused_rzz_segment_forward_,
    native_cpu_shared_rzz_forward_fusion_available,
)
from ....simulation.native_cpu.rotation import native_cpu_forward_rotation_tile_wires


def _following_rotation_block(
    sweep: Any, cursor: int
) -> tuple[int, torch.Tensor, torch.Tensor, int] | None:
    """Collect the first bounded rotation tile after an RZZ segment."""

    if not native_cpu_shared_rzz_forward_fusion_available():
        return None
    start = cursor
    block_wires: list[int] = []
    block_matrices: list[torch.Tensor] = []
    active_wire: int | None = None
    while cursor < len(sweep.ir.instructions):
        if sweep.swaps_before.get(cursor):
            break
        candidate = sweep.ir.instructions[cursor]
        if (
            candidate.name not in {"rx", "ry", "rz"}
            or len(candidate.wires) != 1
            or sweep.matrices[cursor].ndim != 2
        ):
            break
        wire = sweep.persistent_mapping[int(candidate.wires[0])]
        if wire in sweep.plan.sharded_wires:
            break
        if wire == active_wire:
            block_matrices[-1] = sweep.matrices[cursor] @ block_matrices[-1]
        else:
            if wire in block_wires or len(block_wires) == (
                native_cpu_forward_rotation_tile_wires(sweep.plan.n_wires)
            ):
                break
            block_wires.append(wire)
            block_matrices.append(sweep.matrices[cursor])
            active_wire = wire
        cursor += 1
    if len(block_wires) < 2:
        return None
    return (
        cursor,
        torch.stack(block_matrices).contiguous(),
        torch.tensor(block_wires, dtype=torch.int64),
        cursor - start,
    )


def apply_native_rzz_segment(
    sweep: Any,
    index: int,
    instruction: Instruction,
    *,
    touched: bool,
    cpu_direct: bool,
) -> int | None:
    """Apply consecutive local RZZ gates with one native state traversal."""

    if not (
        sweep.local_compilation
        and sweep.world_size == 1
        and sweep.resolved_device.type == "cpu"
        and cpu_direct
        and instruction.name == "rzz"
        and len(instruction.wires) == 2
        and not touched
    ):
        return None

    indices: list[int] = []
    first_wires: list[int] = []
    second_wires: list[int] = []
    cursor = index
    while cursor < len(sweep.ir.instructions):
        if sweep.swaps_before.get(cursor):
            break
        candidate = sweep.ir.instructions[cursor]
        if candidate.name != "rzz" or len(candidate.wires) != 2:
            break
        first, second = (
            sweep.persistent_mapping[int(wire)] for wire in candidate.wires
        )
        if first in sweep.plan.sharded_wires or second in sweep.plan.sharded_wires:
            break
        indices.append(cursor)
        first_wires.append(first)
        second_wires.append(second)
        cursor += 1
    if len(indices) < 2:
        return None

    real_dtype = torch.float32 if sweep.dtype == torch.complex64 else torch.float64
    angles = torch.stack(
        tuple(
            torch.as_tensor(
                sweep.ir.instructions[item].params["theta"],
                device=sweep.resolved_device,
                dtype=real_dtype,
            ).reshape(())
            for item in indices
        )
    ).contiguous()
    first = torch.tensor(first_wires, dtype=torch.int64)
    second = torch.tensor(second_wires, dtype=torch.int64)
    rotation = _following_rotation_block(sweep, cursor)
    if rotation is not None:
        rotation_cursor, matrices, wires, rotation_count = rotation
        if fused_rotation_block_forward_(
            sweep.shard_state.amplitudes,
            matrices,
            wires,
            n_wires=sweep.plan.n_wires,
            rzz_angles=angles,
            rzz_first_wires=first,
            rzz_second_wires=second,
        ):
            sweep.local_count += len(indices) + rotation_count
            sweep.local_diagonal_count += len(indices)
            sweep.peak_scratch = max(
                sweep.peak_scratch,
                sum(
                    item.numel() * item.element_size()
                    for item in (angles, first, second, matrices, wires)
                ),
            )
            return rotation_cursor
    if not fused_rzz_segment_forward_(
        sweep.shard_state.amplitudes,
        angles,
        first,
        second,
        n_wires=sweep.plan.n_wires,
    ):
        return None
    sweep.local_count += len(indices)
    sweep.local_diagonal_count += len(indices)
    sweep.peak_scratch = max(
        sweep.peak_scratch,
        angles.numel() * angles.element_size()
        + first.numel() * first.element_size()
        + second.numel() * second.element_size(),
    )
    return cursor
