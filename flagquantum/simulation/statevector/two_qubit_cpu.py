"""CPU kernels for a dense two-qubit gate on selected state-bit layouts.

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

from collections.abc import Sequence

import torch

_BLOCKED_TWO_QUBIT_MIN_HEAD_STRIDE = 1 << 12
_BLOCKED_TWO_QUBIT_MAX_MULTITHREADED_TAIL_STRIDE = 1 << 4
_BLOCKED_TWO_QUBIT_COMPLEX64_TAIL_STRIDE = 1 << 1
_BLOCKED_TWO_QUBIT_COMPLEX128_LAYOUT_STRIDE = 1 << 5
_BLOCKED_TWO_QUBIT_COST_MODEL_MIN_AMPLITUDES = 1 << 15
_BLOCKED_TWO_QUBIT_REVERSED_MIN_AMPLITUDES = 1 << 8
_BLOCKED_TWO_QUBIT_MULTITHREADED_MIN_AMPLITUDES = 1 << 20
_STRIDED_TWO_QUBIT_MIN_AMPLITUDES = 1 << 21
_STRIDED_TWO_QUBIT_MIN_BATCH = 4
_STRIDED_TWO_QUBIT_MIN_THREADS = 4
_STRIDED_TWO_QUBIT_MAX_FIRST_WIRE = 2
_STRIDED_TWO_QUBIT_MAX_WIRE_GAP = 3


def _prepare_two_qubit_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    matrix_is_bit_reversed: bool,
) -> torch.Tensor:
    """Cast, broadcast, and optionally reverse a two-qubit matrix."""

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
    return matrix


def _prefer_blocked_two_qubit_matrix_cpu(
    state: torch.Tensor,
    *,
    first_wire: int,
    n_wires: int,
    matrix_is_bit_reversed: bool = False,
) -> bool:
    """Choose the adjacent-pair shape classes that favor blocked matmul.

    The blocked kernel avoids two full-state layout copies, but its complex64
    ``[batch, outer, 4, stride]`` matmul is slower than the layout path for the
    middle stride classes once the state is large enough for those copies to
    dominate. Small ascending states keep the existing route. Newly supported
    reversed pairs start at eight wires, where avoiding layout copies repays
    the small matrix reorder. Complex128 only excludes its reproducible
    stride-32 trough.
    """

    first_wire = int(first_wire)
    n_wires = int(n_wires)
    if not 0 <= first_wire < n_wires - 1:
        raise ValueError("adjacent two-qubit wires are outside the statevector")
    stride = 1 << (n_wires - first_wire - 2)
    if matrix_is_bit_reversed and (
        state.shape[-1] < _BLOCKED_TWO_QUBIT_REVERSED_MIN_AMPLITUDES
    ):
        return False
    if (
        not matrix_is_bit_reversed
        and state.shape[-1] < _BLOCKED_TWO_QUBIT_COST_MODEL_MIN_AMPLITUDES
    ):
        return True
    if state.dtype == torch.complex128:
        return stride != _BLOCKED_TWO_QUBIT_COMPLEX128_LAYOUT_STRIDE
    if stride >= _BLOCKED_TWO_QUBIT_MIN_HEAD_STRIDE:
        return True
    if state.dtype != torch.complex64:
        return False
    if stride == _BLOCKED_TWO_QUBIT_COMPLEX64_TAIL_STRIDE:
        return True
    return (
        stride <= _BLOCKED_TWO_QUBIT_MAX_MULTITHREADED_TAIL_STRIDE
        and state.shape[-1] >= _BLOCKED_TWO_QUBIT_MULTITHREADED_MIN_AMPLITUDES
        and torch.get_num_threads() > 1
    )


def _apply_blocked_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    stride: int,
    matrix_is_bit_reversed: bool,
) -> torch.Tensor:
    """Apply a two-qubit gate to contiguous four-amplitude blocks of ``state``."""

    bsz = state.shape[0]
    matrix = _prepare_two_qubit_matrix(
        state, matrix, matrix_is_bit_reversed=matrix_is_bit_reversed
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


def _apply_reversed_adjacent_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    first_wire: int,
    n_wires: int,
) -> torch.Tensor:
    """Apply a CPU gate whose ordered wires are ``(w + 1, w)``."""

    first_wire = int(first_wire)
    n_wires = int(n_wires)
    if not 0 <= first_wire < n_wires - 1:
        raise ValueError("adjacent two-qubit wires are outside the statevector")
    return _apply_blocked_two_qubit_matrix_cpu(
        state,
        matrix,
        stride=1 << (n_wires - first_wire - 2),
        matrix_is_bit_reversed=True,
    )


def _apply_preferred_adjacent_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor | None:
    """Apply an adjacent-pair kernel when its measured shape class favors it."""

    wires = tuple(wires)
    if len(wires) != 2:
        return None
    if wires[1] == wires[0] + 1 and wires[1] < int(n_wires):
        if _prefer_blocked_two_qubit_matrix_cpu(
            state, first_wire=wires[0], n_wires=n_wires
        ):
            return _apply_adjacent_two_qubit_matrix_cpu(
                state, matrix, first_wire=wires[0], n_wires=n_wires
            )
        return None
    if (
        wires[0] == wires[1] + 1
        and state.shape[-1] >= _BLOCKED_TWO_QUBIT_REVERSED_MIN_AMPLITUDES
        and _prefer_blocked_two_qubit_matrix_cpu(
            state,
            first_wire=wires[1],
            n_wires=n_wires,
            matrix_is_bit_reversed=True,
        )
    ):
        return _apply_reversed_adjacent_two_qubit_matrix_cpu(
            state, matrix, first_wire=wires[1], n_wires=n_wires
        )
    return None


def _prefer_strided_two_qubit_matrix_cpu(
    state: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> bool:
    """Choose the measured large-batch shape class for the strided kernel."""

    wires = tuple(int(wire) for wire in wires)
    n_wires = int(n_wires)
    if len(wires) != 2 or len(set(wires)) != 2:
        return False
    low, high = sorted(wires)
    if low < 0 or high >= n_wires or high - low == 1:
        return False
    return (
        state.dtype == torch.complex128
        and state.shape[0] >= _STRIDED_TWO_QUBIT_MIN_BATCH
        and state.shape[-1] >= _STRIDED_TWO_QUBIT_MIN_AMPLITUDES
        and torch.get_num_threads() >= _STRIDED_TWO_QUBIT_MIN_THREADS
        and low <= _STRIDED_TWO_QUBIT_MAX_FIRST_WIRE
        and high - low <= _STRIDED_TWO_QUBIT_MAX_WIRE_GAP
    )


def _apply_strided_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply a nonadjacent gate directly to its four strided amplitude views."""

    wires = tuple(int(wire) for wire in wires)
    n_wires = int(n_wires)
    if len(wires) != 2 or len(set(wires)) != 2:
        raise ValueError("two distinct wires are required")
    low, high = sorted(wires)
    if low < 0 or high >= n_wires or high - low == 1:
        raise ValueError("nonadjacent two-qubit wires are outside the statevector")
    bsz = state.shape[0]
    matrix = _prepare_two_qubit_matrix(
        state, matrix, matrix_is_bit_reversed=wires[0] > wires[1]
    )
    values = state.reshape(
        bsz,
        1 << low,
        2,
        1 << (high - low - 1),
        2,
        1 << (n_wires - high - 1),
    )
    columns = (
        values[:, :, 0, :, 0, :],
        values[:, :, 0, :, 1, :],
        values[:, :, 1, :, 0, :],
        values[:, :, 1, :, 1, :],
    )
    coefficient_shape = (bsz, 1, 1, 1)
    rows = []
    for row in range(4):
        result = columns[0] * matrix[:, row, 0].reshape(coefficient_shape)
        for column in range(1, 4):
            result = result + columns[column] * matrix[:, row, column].reshape(
                coefficient_shape
            )
        rows.append(result)
    low_zero = torch.stack((rows[0], rows[1]), dim=3)
    low_one = torch.stack((rows[2], rows[3]), dim=3)
    return torch.stack((low_zero, low_one), dim=2).reshape(state.shape)


def _apply_preferred_nonadjacent_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor | None:
    """Apply the strided kernel only for its measured profitable shape class."""

    if not _prefer_strided_two_qubit_matrix_cpu(state, wires, n_wires):
        return None
    return _apply_strided_two_qubit_matrix_cpu(state, matrix, wires, n_wires)


def _apply_preferred_two_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor | None:
    """Route a two-qubit CPU gate to a profitable memory-layout kernel."""

    wires = tuple(int(wire) for wire in wires)
    n_wires = int(n_wires)
    if len(wires) != 2:
        return None
    if wires == (n_wires - 1, n_wires - 2):
        return _apply_reversed_trailing_two_qubit_matrix_cpu(state, matrix)
    if abs(wires[0] - wires[1]) != 1:
        return _apply_preferred_nonadjacent_two_qubit_matrix_cpu(
            state, matrix, wires, n_wires
        )
    ascending = wires[1] == wires[0] + 1
    if ascending and state.shape[-1] < _BLOCKED_TWO_QUBIT_COST_MODEL_MIN_AMPLITUDES:
        return _apply_adjacent_two_qubit_matrix_cpu(
            state, matrix, first_wire=wires[0], n_wires=n_wires
        )
    if not ascending and state.shape[-1] < _BLOCKED_TWO_QUBIT_REVERSED_MIN_AMPLITUDES:
        return None
    return _apply_preferred_adjacent_two_qubit_matrix_cpu(state, matrix, wires, n_wires)
