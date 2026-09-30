"""CPU CX-segment helpers for the reversible adjoint sweep."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from ....simulation.native_cpu import (
    compact_cpu_cx_adjoint_auxiliary_bytes,
    fused_compact_cx_adjoint_gather,
    fused_compact_cx_adjoint_inplace_,
    fused_cx_adjoint_gather,
    fused_cx_adjoint_inplace_,
    native_cpu_cx_rotation_adjoint_fusion_available,
    use_compact_cpu_cx_adjoint_cycles,
    use_compact_cpu_cx_mapping,
)
from ....simulation.native_cpu.permutation import compact_cx_permutation_images
from ....simulation.statevector.operations import (
    _cx_sequence_permutation_index,
)


def cpu_cx_permutation_index(
    ket: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Build the shared permutation for a local CPU CX segment."""

    return _cx_sequence_permutation_index(
        controls,
        targets,
        n_wires,
        device=ket.device,
        dtype=ket.dtype,
    )


def cpu_cx_permutation_images(
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Build one compact basis image per wire for an inverse CX mapping."""

    return compact_cx_permutation_images(controls, targets, n_wires)


def apply_cpu_cx_adjoint_index(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a precomputed CX permutation to ket and adjoint."""

    gathered = fused_cx_adjoint_gather(ket, adjoint, index)
    if gathered is not None:
        return gathered
    gathered_ket = torch.index_select(ket, 1, index.to(dtype=torch.int64))
    gathered_adjoint = torch.index_select(adjoint, 1, index.to(dtype=torch.int64))
    ket.copy_(gathered_ket)
    adjoint.copy_(gathered_adjoint)
    return ket, adjoint


def apply_cpu_cx_adjoint_segment(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one CX permutation to ket and adjoint, retaining a safe fallback."""

    index = cpu_cx_permutation_index(ket, controls, targets, n_wires)
    return apply_cpu_cx_adjoint_index(ket, adjoint, index)


def apply_cpu_compact_cx_adjoint_segment(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    images: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a compact CX mapping, retaining the existing table fallback."""

    gathered = fused_compact_cx_adjoint_gather(ket, adjoint, images)
    if gathered is not None:
        return gathered
    return apply_cpu_cx_adjoint_segment(
        ket,
        adjoint,
        controls,
        targets,
        n_wires,
    )


def apply_or_defer_cpu_cx_adjoint_segment(
    sweep: Any,
    segment_wires: Sequence[Sequence[int]],
    *,
    segment_start: int,
    index: int,
) -> bool:
    """Apply a CPU CX segment or defer it into the preceding rotation layer."""

    controls = tuple(wires[0] for wires in reversed(segment_wires))
    targets = tuple(wires[1] for wires in reversed(segment_wires))
    if sweep.policy.low_memory_cpu_cx:
        images = cpu_cx_permutation_images(
            controls,
            targets,
            sweep.plan.n_wires,
        )
        if (
            sweep.policy.compact_cpu_cx_cycles
            and use_compact_cpu_cx_adjoint_cycles(len(controls), sweep.plan.n_wires)
            and fused_compact_cx_adjoint_inplace_(
                sweep.reversible_state.amplitudes,
                sweep.adjoint,
                images,
            )
        ):
            sweep.evidence.peak_scratch_bytes = max(
                sweep.evidence.peak_scratch_bytes,
                compact_cpu_cx_adjoint_auxiliary_bytes(sweep.plan.n_wires),
            )
            sweep.skipped_cx_indices.update(range(segment_start, index))
            return True
        if fused_cx_adjoint_inplace_(
            sweep.reversible_state.amplitudes,
            sweep.adjoint,
            controls,
            targets,
            sweep.plan.n_wires,
        ):
            sweep.skipped_cx_indices.update(range(segment_start, index))
            return True
    if (
        segment_start > 0
        and sweep.bound.instructions[segment_start - 1].name in {"rx", "ry", "rz"}
        and sweep.pending_cpu_cx_index is None
        and sweep.pending_cpu_cx_images is None
        and native_cpu_cx_rotation_adjoint_fusion_available()
    ):
        if use_compact_cpu_cx_mapping(sweep.plan.n_wires):
            sweep.pending_cpu_cx_images = cpu_cx_permutation_images(
                controls,
                targets,
                sweep.plan.n_wires,
            )
            sweep.pending_cpu_cx_controls = controls
            sweep.pending_cpu_cx_targets = targets
        else:
            sweep.pending_cpu_cx_index = cpu_cx_permutation_index(
                sweep.reversible_state.amplitudes,
                controls,
                targets,
                sweep.plan.n_wires,
            )
    else:
        ket, adjoint = apply_cpu_cx_adjoint_segment(
            sweep.reversible_state.amplitudes,
            sweep.adjoint,
            controls,
            targets,
            sweep.plan.n_wires,
        )
        sweep.reversible_state = replace(sweep.reversible_state, amplitudes=ket)
        sweep.adjoint = adjoint
        sweep.evidence.peak_scratch_bytes = max(
            sweep.evidence.peak_scratch_bytes,
            2 * ket.numel() * ket.element_size(),
        )
    sweep.skipped_cx_indices.update(range(segment_start, index))
    return True
