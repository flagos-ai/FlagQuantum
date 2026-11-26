"""Pure local CPU CX adjoint and permutation numerics."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from flagquantum.simulation.native_cpu import (
    fused_compact_cx_adjoint_gather,
    fused_cx_adjoint_gather,
)
from flagquantum.simulation.native_cpu.permutation import compact_cx_permutation_images
from flagquantum.simulation.statevector.operations import (
    _cx_sequence_permutation_index,
)


def cpu_cx_permutation_index(
    ket: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    """Build the shared permutation for a local CPU CX segment."""

    return _cx_sequence_permutation_index(
        controls,
        targets,
        n_qubits,
        device=ket.device,
        dtype=ket.dtype,
    )


def cpu_cx_permutation_images(
    controls: Sequence[int],
    targets: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    """Build one compact basis image per qubit for an inverse CX mapping."""

    return compact_cx_permutation_images(controls, targets, n_qubits)


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
    n_qubits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one CX permutation to ket and adjoint, retaining a safe fallback."""

    index = cpu_cx_permutation_index(ket, controls, targets, n_qubits)
    return apply_cpu_cx_adjoint_index(ket, adjoint, index)


def apply_cpu_compact_cx_adjoint_segment(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    images: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_qubits: int,
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
        n_qubits,
    )


for _identity in (
    cpu_cx_permutation_index,
    cpu_cx_permutation_images,
    apply_cpu_cx_adjoint_index,
    apply_cpu_cx_adjoint_segment,
    apply_cpu_compact_cx_adjoint_segment,
):
    _identity.__module__ = (
        "flagquantum.runtime.executors.statevector.reverse_adjoint_cx"
    )
del _identity
