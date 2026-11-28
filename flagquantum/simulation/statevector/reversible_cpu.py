"""Exact local Toffoli permutations with explicit state-buffer ownership."""

from collections.abc import Sequence

import torch


def _apply_toffoli_cpu(
    state: torch.Tensor,
    qubits: Sequence[int],
    n_qubits: int,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Swap target amplitudes, reusing only owned, non-differentiable storage."""
    first, second, target = qubits
    # Direct indexing remains cheaper for small states and the large streaming regime.
    if not 12 <= n_qubits < 20:
        tensor = state.reshape((state.shape[0],) + (2,) * n_qubits)
        selection: list[int | slice] = [slice(None)] * (n_qubits + 1)
        selection[first + 1] = selection[second + 1] = 1
        selected = tuple(selection)
        target_axis = target + 1 - int(first < target) - int(second < target)
        output = tensor if inplace and not state.requires_grad else tensor.clone()
        output[selected] = tensor[selected].flip(target_axis)
        return output.reshape(state.shape)

    # Group untouched axes so indexing the two target halves has bounded rank.
    low, middle, high = ordered = sorted((first, second, target))
    tensor = state.reshape(
        state.shape[0],
        1 << low,
        2,
        1 << (middle - low - 1),
        2,
        1 << (high - middle - 1),
        2,
        1 << (n_qubits - high - 1),
    )
    right = (slice(None), slice(None), 1, slice(None), 1, slice(None), 1, slice(None))
    left: list[int | slice] = list(right)
    left[2 + 2 * ordered.index(target)] = 0
    selected_left = tuple(left)
    output = tensor if inplace and not state.requires_grad else tensor.clone()
    if output is tensor:
        left_view, right_view = tensor[selected_left], tensor[right]
        temporary = left_view.clone()
        left_view.copy_(right_view)
        right_view.copy_(temporary)
    else:
        output[selected_left].copy_(tensor[right])
        output[right].copy_(tensor[selected_left])
    return output.reshape(state.shape)
