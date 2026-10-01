"""BMM lowering and future provider boundary for MPS canonical transfers."""

from __future__ import annotations

import torch


def absorb_left_canonical_transfer(
    transfer: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Absorb a right-going transfer through canonical batched matmul."""

    if transfer.ndim != 3 or right.ndim != 4:
        raise ValueError(
            "left canonical transfer absorption expects rank-3/rank-4 inputs"
        )
    batch, bond, physical_dim, right_dim = right.shape
    if transfer.shape[0] != batch or transfer.shape[2] != bond:
        raise ValueError("left canonical transfer dimensions do not align")
    matrix = right.reshape(batch, bond, physical_dim * right_dim)
    result = torch.bmm(transfer, matrix)
    return result.reshape(batch, transfer.shape[1], physical_dim, right_dim)


def absorb_right_canonical_transfer(
    left: torch.Tensor,
    transfer: torch.Tensor,
) -> torch.Tensor:
    """Absorb a left-going transfer through canonical batched matmul."""

    if left.ndim != 4 or transfer.ndim != 3:
        raise ValueError(
            "right canonical transfer absorption expects rank-4/rank-3 inputs"
        )
    batch, left_dim, physical_dim, bond = left.shape
    if transfer.shape[0] != batch or transfer.shape[1] != bond:
        raise ValueError("right canonical transfer dimensions do not align")
    matrix = left.reshape(batch, left_dim * physical_dim, bond)
    result = torch.bmm(matrix, transfer)
    return result.reshape(batch, left_dim, physical_dim, transfer.shape[2])


__all__ = (
    "absorb_left_canonical_transfer",
    "absorb_right_canonical_transfer",
)
