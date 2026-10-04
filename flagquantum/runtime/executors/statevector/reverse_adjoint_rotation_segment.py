"""Native CPU planning for adjoint rotation segments."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ....core.ir import Instruction
from ....simulation.native_cpu import (
    fused_cx_rotation_segment_adjoint,
    fused_rotation_segment_adjoint_,
    native_cpu_adjoint_rzz_h_fusion_available,
    native_cpu_rotation_rzz_fusion_available,
    native_cpu_rotation_segment_available,
    native_cpu_rotation_tile_wires,
    native_cpu_shared_rotation_gradient_available,
    native_cpu_terminal_adjoint_no_restore_available,
)
from .reverse_adjoint_cx import (
    apply_cpu_compact_cx_adjoint_segment,
    apply_cpu_cx_adjoint_index,
)
from .reverse_adjoint_kernels import _cpu_direct_adjoint_gate_enabled
from .reverse_observable import materialize_pending_observable_adjoint

_KIND_BY_NAME = {"rx": 0, "ry": 1, "rz": 2}


def _materialize_pending_cpu_cx(sweep: Any) -> None:
    """Apply a deferred CPU CX permutation before leaving its fusion boundary."""

    index = getattr(sweep, "pending_cpu_cx_index", None)
    images = getattr(sweep, "pending_cpu_cx_images", None)
    if index is None and images is None:
        return
    if images is not None:
        ket, adjoint = apply_cpu_compact_cx_adjoint_segment(
            sweep.reversible_state.amplitudes,
            sweep.adjoint,
            images,
            sweep.pending_cpu_cx_controls,
            sweep.pending_cpu_cx_targets,
            sweep.plan.n_qubits,
        )
    else:
        assert index is not None
        ket, adjoint = apply_cpu_cx_adjoint_index(
            sweep.reversible_state.amplitudes,
            sweep.adjoint,
            index,
        )
    sweep.reversible_state = replace(sweep.reversible_state, amplitudes=ket)
    sweep.adjoint = adjoint
    sweep.pending_cpu_cx_index = None
    sweep.pending_cpu_cx_images = None
    sweep.pending_cpu_cx_controls = ()
    sweep.pending_cpu_cx_targets = ()


def _fusable_rzz_layer(
    sweep: Any,
    cursor: int,
    real_dtype: torch.dtype,
    final_tile_wires: frozenset[int],
) -> (
    tuple[
        tuple[int, ...],
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[int, ...],
    ]
    | None
):
    """Describe an adjacent shared-parameter RZZ layer directly before rotations."""

    if cursor < 1 or not native_cpu_rotation_rzz_fusion_available():
        return None
    indices: list[int] = []
    while cursor >= 0:
        candidate = sweep.bound.instructions[cursor]
        if candidate.name != "rzz" or sweep.swaps_before.get(cursor):
            break
        indices.append(cursor)
        cursor -= 1
    active = tuple(
        sorted(sweep.slots_by_instruction.get(gate_index, ())) for gate_index in indices
    )
    if len(indices) < 2 or any(len(item) != 1 for item in active):
        return None
    if len({item[0] for item in active}) != 1:
        return None
    pairs = tuple(
        tuple(
            sweep.persistent_mapping[int(wire)]
            for wire in sweep.bound.instructions[gate_index].wires
        )
        for gate_index in indices
    )
    if any(abs(first - second) != 1 for first, second in pairs):
        return None
    if len({tuple(sorted(pair)) for pair in pairs}) != len(pairs):
        return None
    angles = torch.stack(
        tuple(
            torch.as_tensor(
                sweep.bound.instructions[gate_index].params["theta"],
                device=sweep.device,
                dtype=real_dtype,
            ).reshape(())
            for gate_index in indices
        )
    ).contiguous()
    hadamard_indices: tuple[int, ...] = ()
    if native_cpu_adjoint_rzz_h_fusion_available():
        matched: dict[int, int] = {}
        while cursor >= 0:
            if sweep.swaps_before.get(cursor) or sweep.slots_by_instruction.get(cursor):
                break
            candidate = sweep.bound.instructions[cursor]
            if candidate.name != "h" or len(candidate.wires) != 1:
                break
            wire = sweep.persistent_mapping[int(candidate.wires[0])]
            if wire in matched:
                break
            matched[wire] = cursor
            cursor -= 1
        if final_tile_wires.issubset(matched):
            hadamard_indices = tuple(matched[wire] for wire in final_tile_wires)
    return (
        tuple(indices),
        active[0][0],
        angles,
        torch.tensor([pair[0] for pair in pairs], dtype=torch.int64),
        torch.tensor([pair[1] for pair in pairs], dtype=torch.int64),
        hadamard_indices,
    )


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
        _materialize_pending_cpu_cx(sweep)
        materialize_pending_observable_adjoint(sweep)
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
        _materialize_pending_cpu_cx(sweep)
        materialize_pending_observable_adjoint(sweep)
        return False
    active_by_gate = tuple(
        sorted(sweep.slots_by_instruction.get(gate_index, ())) for gate_index in indices
    )
    if any(len(active) != 1 for active in active_by_gate):
        _materialize_pending_cpu_cx(sweep)
        materialize_pending_observable_adjoint(sweep)
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
    tile_wires = native_cpu_rotation_tile_wires()
    final_tile_start = ((len(block_wires) - 1) // tile_wires) * tile_wires
    final_tile_wires = frozenset(block_wires[final_tile_start:])
    rzz_layer = _fusable_rzz_layer(sweep, cursor, real_dtype, final_tile_wires)
    terminal_euler_layer = (
        cursor < 0
        and native_cpu_terminal_adjoint_no_restore_available()
        and len(gate_kinds) % 3 == 0
        and all(
            gate_kinds[offset : offset + 3] == [2, 1, 0]
            and mapped_wires[offset] == mapped_wires[offset + 1]
            and mapped_wires[offset] == mapped_wires[offset + 2]
            for offset in range(0, len(gate_kinds), 3)
        )
    )
    angle_tensor = torch.stack(angles).contiguous()
    kind_tensor = torch.tensor(gate_kinds, dtype=torch.int64)
    wire_tensor = torch.tensor(mapped_wires, dtype=torch.int64)
    gradients: torch.Tensor | None = None
    pending_cx_index = getattr(sweep, "pending_cpu_cx_index", None)
    pending_cx_images = getattr(sweep, "pending_cpu_cx_images", None)
    if (
        pending_cx_index is not None or pending_cx_images is not None
    ) and rzz_layer is None:
        fused = fused_cx_rotation_segment_adjoint(
            sweep.reversible_state.amplitudes,
            sweep.adjoint,
            pending_cx_index,
            angle_tensor,
            kind_tensor,
            wire_tensor,
            n_wires=sweep.plan.n_qubits,
            aggregate_shared_parameter=shared_parameter,
            observable_weights=sweep.pending_observable_weights,
            cx_images=pending_cx_images,
            restore_state=not terminal_euler_layer,
        )
        if fused is not None:
            gradients, ket, sweep.adjoint = fused
            sweep.reversible_state = replace(sweep.reversible_state, amplitudes=ket)
            sweep.pending_cpu_cx_index = None
            sweep.pending_cpu_cx_images = None
            sweep.pending_cpu_cx_controls = ()
            sweep.pending_cpu_cx_targets = ()
            sweep.evidence.peak_scratch_bytes = max(
                sweep.evidence.peak_scratch_bytes,
                2 * ket.numel() * ket.element_size(),
            )
    if gradients is None:
        _materialize_pending_cpu_cx(sweep)
        gradients = fused_rotation_segment_adjoint_(
            sweep.reversible_state.amplitudes,
            sweep.adjoint,
            angle_tensor,
            kind_tensor,
            wire_tensor,
            n_wires=sweep.plan.n_qubits,
            aggregate_shared_parameter=shared_parameter,
            rzz_angles=None if rzz_layer is None else rzz_layer[2],
            rzz_first_wires=None if rzz_layer is None else rzz_layer[3],
            rzz_second_wires=None if rzz_layer is None else rzz_layer[4],
            fuse_preceding_hadamards=bool(rzz_layer and rzz_layer[5]),
            observable_weights=sweep.pending_observable_weights,
            restore_state=not terminal_euler_layer,
        )
    if gradients is None:
        materialize_pending_observable_adjoint(sweep)
        return False
    sweep.pending_observable_weights = None
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
    if rzz_layer is not None:
        rzz_indices, parameter_index = rzz_layer[:2]
        sweep._accumulate_parameter_gradient(
            parameter_index,
            gradients[1 if shared_parameter else len(indices)].to(
                dtype=sweep.base_parameters[parameter_index].dtype
            ),
            rzz_indices[-1],
            occurrence_count=len(rzz_indices),
        )
        sweep.evidence.analytic_rotation_derivative_count += len(rzz_indices)
        sweep.evidence.fused_parameter_adjoint_count += len(rzz_indices)
        sweep.skipped_rzz_indices.update(rzz_indices)
        sweep.skipped_fixed_block_indices.update(rzz_layer[5])
    return True
