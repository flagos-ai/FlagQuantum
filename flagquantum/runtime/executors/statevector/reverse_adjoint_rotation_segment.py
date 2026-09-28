"""Native CPU planning for adjoint rotation segments."""

from __future__ import annotations

from typing import Any

import torch

from ....core.ir import Instruction
from ....simulation.native_cpu import (
    fused_rotation_segment_adjoint_,
    native_cpu_rotation_segment_available,
    native_cpu_shared_rotation_gradient_available,
)
from .reverse_adjoint_kernels import _cpu_direct_adjoint_gate_enabled

_KIND_BY_NAME = {"rx": 0, "ry": 1, "rz": 2}


def _apply_local_rotation_segment(
    sweep: Any, index: int, execution_instruction: Instruction
) -> bool:
    """Undo one contiguous rotation layer through a single native dispatch."""

    if not (
        sweep.inplace_local
        and sweep.reversible_state is not None
        and execution_instruction.name in _KIND_BY_NAME
        and len(execution_instruction.wires) == 1
        and sweep.world_size == 1
        and sweep.reversible_state.amplitudes.device.type == "cpu"
        and _cpu_direct_adjoint_gate_enabled()
        and native_cpu_rotation_segment_available()
        and not sweep.swaps_before.get(index)
    ):
        return False

    indices: list[int] = []
    block_wires: list[int] = []
    active_wire: int | None = None
    cursor = index
    while cursor >= 0:
        if sweep.swaps_before.get(cursor):
            break
        candidate = sweep.bound.instructions[cursor]
        if candidate.name not in _KIND_BY_NAME or len(candidate.wires) != 1:
            break
        wire = sweep.persistent_mapping[int(candidate.wires[0])]
        if wire == active_wire:
            indices.append(cursor)
        else:
            if wire in block_wires:
                break
            block_wires.append(wire)
            active_wire = wire
            indices.append(cursor)
        cursor -= 1

    if len(block_wires) < 2:
        return False
    active_by_gate = tuple(
        sorted(sweep.slots_by_instruction.get(gate_index, ())) for gate_index in indices
    )
    if any(len(active) != 1 for active in active_by_gate):
        return False

    real_dtype = torch.float32 if sweep.dtype == torch.complex64 else torch.float64
    angles: list[torch.Tensor] = []
    gate_kinds: list[int] = []
    mapped_wires: list[int] = []
    for gate_index in indices:
        candidate = sweep.bound.instructions[gate_index]
        angles.append(
            torch.as_tensor(
                candidate.params["theta"], device=sweep.device, dtype=real_dtype
            ).reshape(())
        )
        gate_kinds.append(_KIND_BY_NAME[candidate.name])
        mapped_wires.append(sweep.persistent_mapping[int(candidate.wires[0])])

    shared_parameter = (
        len({active[0] for active in active_by_gate}) == 1
        and native_cpu_shared_rotation_gradient_available()
    )
    gradients = fused_rotation_segment_adjoint_(
        sweep.reversible_state.amplitudes,
        sweep.adjoint,
        torch.stack(angles).contiguous(),
        torch.tensor(gate_kinds, dtype=torch.int64),
        torch.tensor(mapped_wires, dtype=torch.int64),
        n_wires=sweep.plan.n_wires,
        aggregate_shared_parameter=shared_parameter,
    )
    if gradients is None:
        return False
    if shared_parameter:
        parameter_index = active_by_gate[0][0]
        sweep._accumulate_parameter_gradient(
            parameter_index,
            gradients[0].to(dtype=sweep.base_parameters[parameter_index].dtype),
            indices[-1],
            occurrence_count=len(indices),
        )
    else:
        for offset, (gate_index, active) in enumerate(
            zip(indices, active_by_gate, strict=True)
        ):
            parameter_index = active[0]
            sweep._accumulate_parameter_gradient(
                parameter_index,
                gradients[offset].to(
                    dtype=sweep.base_parameters[parameter_index].dtype
                ),
                gate_index,
            )
    sweep.evidence.analytic_rotation_derivative_count += len(indices)
    sweep.evidence.fused_parameter_adjoint_count += len(indices)
    sweep.skipped_rotation_indices.update(indices[1:])
    return True
