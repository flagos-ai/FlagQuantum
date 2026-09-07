"""Local numerical execution for one lowered noisy MPS trajectory."""

from __future__ import annotations

import torch

from ...core.ir import CircuitIR
from .state import MPSState


def run_local_noisy_mps_trajectory(
    lowered_ir: CircuitIR,
    mps: MPSState,
    *,
    generator: torch.Generator | None,
) -> MPSState:
    """Execute one lowered IR on an initialized MPS with an explicit RNG."""

    for instruction in lowered_ir:
        if instruction.metadata.get("is_channel"):
            mps.apply_channel_trajectory(
                instruction.matrix,
                instruction.wires,
                generator=generator,
            )
        else:
            mps.apply_instruction(instruction)
    return mps
