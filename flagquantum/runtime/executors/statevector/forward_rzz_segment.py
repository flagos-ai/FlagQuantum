"""Native CPU planning for contiguous forward RZZ segments."""

from __future__ import annotations

from typing import Any

import torch

from ....core.ir import Instruction
from ....simulation.native_cpu import fused_rzz_segment_forward_


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
