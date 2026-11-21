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
    tensor = state.reshape((state.shape[0],) + (2,) * n_qubits)
    selection: list[int | slice] = [slice(None)] * (n_qubits + 1)
    selection[first + 1] = selection[second + 1] = 1
    selected = tuple(selection)
    target_axis = target + 1 - int(first < target) - int(second < target)
    output = tensor if inplace and not state.requires_grad else tensor.clone()
    output[selected] = tensor[selected].flip(target_axis)
    return output.reshape(state.shape)
