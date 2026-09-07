"""Fixed-rank MPS factorization primitives.

The reference implementation defines the numerical contract for a future
Triton kernel that computes the range projection without materializing the
full two-site matrix.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _cpu_projection(rows: int, columns: int, rank: int) -> torch.Tensor:
    """Return a deterministic complex Gaussian range projection."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0xF1A6 + 131 * rows + 17 * columns + rank)
    real = torch.randn(columns, rank, generator=generator)
    imag = torch.randn(columns, rank, generator=generator)
    return torch.complex(real, imag) / float(max(columns, 1)) ** 0.5


def fixed_rank_range_qr(
    matrix: torch.Tensor,
    rank: int,
    *,
    power_iterations: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate ``matrix`` as ``Q @ B`` with a fixed rank.

    This avoids SVD and has a stable reduced-QR backward when the sampled
    range has full column rank.  It is approximate even when cutoff is zero.
    """

    if matrix.ndim != 3:
        raise ValueError("matrix must have shape [batch, rows, columns]")
    rows, columns = int(matrix.shape[-2]), int(matrix.shape[-1])
    rank = int(rank)
    if rank < 1 or rank > min(rows, columns):
        raise ValueError("rank must be in [1, min(rows, columns)]")
    if power_iterations < 0:
        raise ValueError("power_iterations must be nonnegative")
    projection = _cpu_projection(rows, columns, rank).to(
        device=matrix.device, dtype=matrix.dtype
    )
    sampled = torch.matmul(matrix, projection)
    for _ in range(int(power_iterations)):
        sampled = torch.matmul(
            matrix, torch.matmul(torch.conj(matrix).transpose(-2, -1), sampled)
        )
    basis, _ = torch.linalg.qr(sampled, mode="reduced")
    reduced = torch.matmul(torch.conj(basis).transpose(-2, -1), matrix)
    return basis, reduced


def fixed_rank_two_site_range_qr(
    left: torch.Tensor,
    gate: torch.Tensor,
    right: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor a gated two-site tensor without materializing its full matrix."""

    batch, left_dim, _, _ = left.shape
    right_dim = int(right.shape[-1])
    projection = (
        _cpu_projection(2 * left_dim, 2 * right_dim, int(rank))
        .to(device=left.device, dtype=left.dtype)
        .reshape(2, right_dim, int(rank))
    )
    gate_tensor = (
        gate.reshape(batch, 2, 2, 2, 2) if gate.ndim == 3 else gate.reshape(2, 2, 2, 2)
    )
    projected_right = torch.einsum("bmtr,qrk->bmtqk", right, projection)
    equation = "bpqst,blsm,bmtqk->blpk" if gate.ndim == 3 else "pqst,blsm,bmtqk->blpk"
    sampled = torch.einsum(equation, gate_tensor, left, projected_right).reshape(
        batch, 2 * left_dim, int(rank)
    )
    basis, _ = torch.linalg.qr(sampled, mode="reduced")
    basis_tensor = basis.reshape(batch, left_dim, 2, int(rank))
    reduced_equation = (
        "blpk,bpqst,blsm,bmtr->bkqr" if gate.ndim == 3 else "blpk,pqst,blsm,bmtr->bkqr"
    )
    reduced = torch.einsum(
        reduced_equation, torch.conj(basis_tensor), gate_tensor, left, right
    )
    return basis_tensor, reduced


__all__ = ["fixed_rank_range_qr", "fixed_rank_two_site_range_qr"]
