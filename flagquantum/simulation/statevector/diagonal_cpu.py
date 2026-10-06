"""CPU tensor kernels for fused statevector diagonal regions."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def apply_diagonal_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    *,
    inplace: bool = False,
) -> torch.Tensor | None:
    """Apply a diagonal matrix directly when the state uses the CPU layout."""

    if state.device.type != "cpu" or not state.is_contiguous():
        return None
    bsz = state.shape[0]
    matrix = matrix.to(device=state.device, dtype=state.dtype)
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    if diagonal.ndim == 1:
        diagonal = diagonal.unsqueeze(0).expand(bsz, -1)
    elif diagonal.shape[0] == 1 and bsz != 1:
        diagonal = diagonal.expand(bsz, -1)
    dim = 2 ** len(wires)
    if diagonal.shape != (bsz, dim):
        raise ValueError(
            "diagonal matrix must have shape [dim, dim] or [batch, dim, dim]"
        )
    ordered_wires = tuple(sorted(wires))
    wire_order = tuple(wires.index(wire) for wire in ordered_wires)
    factors = diagonal.reshape((bsz,) + (2,) * len(wires)).permute(
        (0,) + tuple(index + 1 for index in wire_order)
    )
    factor_shape = [bsz] + [1] * n_wires
    for wire in ordered_wires:
        factor_shape[wire + 1] = 2
    tensor = state.reshape((bsz,) + (2,) * n_wires)
    factors = factors.reshape(factor_shape)
    if inplace and not state.requires_grad and not matrix.requires_grad:
        tensor.mul_(factors)
        return state
    return (tensor * factors).reshape(state.shape)


def _apply_cross_wire_diagonal_cpu(
    state: torch.Tensor,
    diagonals: Sequence[torch.Tensor],
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply several single-qubit diagonals with one pass over a CPU statevector."""

    return _apply_disjoint_diagonal_regions_cpu(
        state,
        diagonals,
        tuple((int(wire),) for wire in wires),
        n_wires,
    )


def _apply_disjoint_diagonal_regions_cpu(
    state: torch.Tensor,
    diagonals: Sequence[torch.Tensor],
    wire_groups: Sequence[Sequence[int]],
    n_wires: int,
) -> torch.Tensor:
    """Apply qubit-disjoint one- or two-qubit diagonals in one statevector pass."""

    normalized_groups = tuple(
        tuple(int(wire) for wire in wires) for wires in wire_groups
    )
    if len(diagonals) != len(normalized_groups):
        raise ValueError("one diagonal is required for each wire group")
    flat_wires = tuple(wire for wires in normalized_groups for wire in wires)
    if len(set(flat_wires)) != len(flat_wires):
        raise ValueError("diagonal fusion requires wire-disjoint regions")
    if any(not 0 <= wire < int(n_wires) for wire in flat_wires):
        raise ValueError("wire is outside the statevector")
    if any(len(wires) not in {1, 2} for wires in normalized_groups):
        raise ValueError("diagonal fusion supports one- or two-wire regions")

    bsz = state.shape[0]
    combined = state.new_ones((bsz, 1))
    factor_wires: list[int] = []
    for wires, diagonal in zip(normalized_groups, diagonals, strict=True):
        diagonal = diagonal.to(device=state.device, dtype=state.dtype)
        dim = 2 ** len(wires)
        if diagonal.ndim == 1:
            diagonal = diagonal.unsqueeze(0).expand(bsz, -1)
        elif diagonal.ndim == 2 and diagonal.shape[0] == 1 and bsz != 1:
            diagonal = diagonal.expand(bsz, -1)
        if diagonal.shape != (bsz, dim):
            raise ValueError(
                "diagonal must have shape [2**len(wires)] or [batch, 2**len(wires)]"
            )
        combined = (combined.unsqueeze(-1) * diagonal.unsqueeze(1)).reshape(bsz, -1)
        factor_wires.extend(wires)

    wire_order = tuple(sorted(range(len(factor_wires)), key=factor_wires.__getitem__))
    if wire_order != tuple(range(len(factor_wires))):
        combined = combined.reshape((bsz,) + (2,) * len(factor_wires)).permute(
            (0,) + tuple(index + 1 for index in wire_order)
        )

    factor_shape = [bsz] + [1] * int(n_wires)
    for wire in sorted(flat_wires):
        factor_shape[wire + 1] = 2
    tensor = state.reshape((bsz,) + (2,) * int(n_wires))
    return (tensor * combined.reshape(factor_shape)).reshape(state.shape)
