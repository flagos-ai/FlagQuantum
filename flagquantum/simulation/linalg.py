"""Native linear algebra utilities backed by PyTorch."""

from __future__ import annotations

import torch


def expm(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_exp(matrix)


def svd(matrix: torch.Tensor, full_matrices: bool = True):
    return torch.linalg.svd(matrix, full_matrices=full_matrices)


def qr(matrix: torch.Tensor, mode: str = "reduced"):
    return torch.linalg.qr(matrix, mode=mode)


def eigh(matrix: torch.Tensor):
    return torch.linalg.eigh(matrix)


def random_unitary(n: int, *, device=None, dtype=torch.complex64) -> torch.Tensor:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    z = torch.randn(n, n, device=device, dtype=real_dtype) + 1j * torch.randn(
        n, n, device=device, dtype=real_dtype
    )
    q, r = torch.linalg.qr(z.to(dtype))
    phases = torch.diagonal(r) / torch.abs(torch.diagonal(r)).clamp_min(1e-12)
    return q * phases.conj()


__all__ = ["expm", "svd", "qr", "eigh", "random_unitary"]
