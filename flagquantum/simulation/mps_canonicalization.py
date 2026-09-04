"""Pure rank-local MPS canonicalization numerics."""

from __future__ import annotations

from typing import Mapping

import torch


def deterministic_mps_qr(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """QR with a non-negative real diagonal convention for stable gauges."""

    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    magnitude = diagonal.abs()
    phase = torch.where(magnitude > 0, diagonal / magnitude, torch.ones_like(diagonal))
    return q * phase.unsqueeze(-2), phase.conj().unsqueeze(-1) * r


def factor_left_canonical_site(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a left-canonical site and its right-going transfer matrix."""

    bsz, left_dim, physical_dim, right_dim = tensor.shape
    q, transfer = deterministic_mps_qr(
        tensor.reshape(bsz, left_dim * physical_dim, right_dim)
    )
    return q.reshape(bsz, left_dim, physical_dim, q.shape[-1]), transfer


def absorb_left_canonical_transfer(
    transfer: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Absorb a right-going canonicalization transfer into the next site."""

    return torch.einsum("bij,bjsk->bisk", transfer, right)


def factor_right_canonical_site(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a left-going transfer matrix and a right-canonical site."""

    bsz, left_dim, physical_dim, right_dim = tensor.shape
    matrix_t = tensor.reshape(bsz, left_dim, physical_dim * right_dim).transpose(-2, -1)
    q, r = deterministic_mps_qr(matrix_t)
    right = q.transpose(-2, -1)
    transfer = r.transpose(-2, -1)
    return transfer, right.reshape(bsz, right.shape[1], physical_dim, right_dim)


def absorb_right_canonical_transfer(
    left: torch.Tensor,
    transfer: torch.Tensor,
) -> torch.Tensor:
    """Absorb a left-going canonicalization transfer into the previous site."""

    return torch.einsum("blpa,bac->blpc", left, transfer)


def local_mixed_canonical_residual(
    tensors: Mapping[int, torch.Tensor],
    center: int,
) -> torch.Tensor:
    """Return the maximum owner-local mixed-canonical residual."""

    if not tensors:
        raise ValueError("MPS canonical residual requires at least one local tensor")
    reference = next(iter(tensors.values()))
    residual = torch.zeros((), dtype=torch.float64, device=reference.device)
    for wire, tensor in tensors.items():
        if wire == center:
            continue
        for batch_tensor in tensor:
            if wire < center:
                matrix = batch_tensor.reshape(-1, batch_tensor.shape[-1])
                gram = matrix.mH @ matrix
            else:
                matrix = batch_tensor.reshape(batch_tensor.shape[0], -1)
                gram = matrix @ matrix.mH
            eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            residual = torch.maximum(
                residual, torch.max(torch.abs(gram - eye)).double()
            )
    return residual


def mps_center_norms(tensor: torch.Tensor) -> torch.Tensor:
    """Return squared state norms represented by one canonical center tensor."""

    return torch.sum(torch.abs(tensor.reshape(tensor.shape[0], -1)) ** 2, dim=1)


__all__ = (
    "absorb_left_canonical_transfer",
    "absorb_right_canonical_transfer",
    "deterministic_mps_qr",
    "factor_left_canonical_site",
    "factor_right_canonical_site",
    "local_mixed_canonical_residual",
    "mps_center_norms",
)
