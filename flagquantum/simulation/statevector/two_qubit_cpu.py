"""CPU kernels for a dense two-qubit gate on two adjacent state bits.

The statevector is contiguous and wire ``w`` carries the bit of weight
``2 ** (n_wires - 1 - w)``, so the four amplitudes that differ in an index-adjacent
pair ``(w, w + 1)`` are always contiguous: a view of the state as
``(batch, 2 ** w, 4, 2 ** (n_wires - w - 2))`` blocks. Their order within a block
is ``bit(w) * 2 + bit(w + 1)``, which is the gate matrix's own index order for an
ascending pair, so such a gate is one bounded matrix multiplication over a view
with no statevector permutation. The descending trailing pair has the same
contiguous shape but the opposite block order, so it reorders the small matrix
instead.

These kernels live beside ``operations.py`` because they are addressed by the
state's memory layout rather than by the compiler's program steps.
"""

from __future__ import annotations

import torch


def _apply_blocked_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    stride: int,
    matrix_is_bit_reversed: bool,
) -> torch.Tensor:
    """Apply a two-qubit gate to contiguous four-amplitude blocks of ``state``."""

    bsz = state.shape[0]
    matrix = matrix.to(device=state.device, dtype=state.dtype)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).expand(bsz, -1, -1)
    elif matrix.ndim == 3 and matrix.shape[0] == 1 and bsz != 1:
        matrix = matrix.expand(bsz, -1, -1)
    if matrix.shape != (bsz, 4, 4):
        raise ValueError("two-qubit matrix must have shape [4, 4] or [batch, 4, 4]")
    if matrix_is_bit_reversed:
        matrix = (
            matrix.reshape(bsz, 2, 2, 2, 2).permute(0, 2, 1, 4, 3).reshape(bsz, 4, 4)
        )
    blocks = state.reshape(bsz, -1, 4, stride)
    return torch.matmul(matrix.unsqueeze(1), blocks).reshape(state.shape)


def _apply_reversed_trailing_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply a CPU gate whose ordered wires are the two trailing state bits."""

    return _apply_blocked_two_qubit_matrix_cpu(
        state, matrix, stride=1, matrix_is_bit_reversed=True
    )


def _apply_adjacent_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    first_wire: int,
    n_wires: int,
) -> torch.Tensor:
    """Apply a CPU gate whose ordered wires are ``(w, w + 1)``, ascending."""

    first_wire = int(first_wire)
    n_wires = int(n_wires)
    if not 0 <= first_wire < n_wires - 1:
        raise ValueError("adjacent two-qubit wires are outside the statevector")
    return _apply_blocked_two_qubit_matrix_cpu(
        state,
        matrix,
        stride=1 << (n_wires - first_wire - 2),
        matrix_is_bit_reversed=False,
    )
