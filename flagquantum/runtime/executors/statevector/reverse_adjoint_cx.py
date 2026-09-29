"""CPU CX-segment helpers for the reversible adjoint sweep."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ....simulation.native_cpu import fused_cx_adjoint_gather
from ....simulation.statevector.operations import (
    _apply_cx_sequence_gather,
    _cx_sequence_permutation_index,
)


def apply_cpu_cx_adjoint_segment(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one CX permutation to ket and adjoint, retaining a safe fallback."""

    index = _cx_sequence_permutation_index(
        controls,
        targets,
        n_wires,
        device=ket.device,
        dtype=ket.dtype,
    )
    gathered = fused_cx_adjoint_gather(ket, adjoint, index)
    if gathered is not None:
        return gathered
    gathered_ket = _apply_cx_sequence_gather(ket, controls, targets, n_wires)
    gathered_adjoint = _apply_cx_sequence_gather(adjoint, controls, targets, n_wires)
    ket.copy_(gathered_ket)
    adjoint.copy_(gathered_adjoint)
    return ket, adjoint
