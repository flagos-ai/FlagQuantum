"""Provide an independent complete finite-resolution spectrum reference."""

import math

import torch

from .model import COMPLEX, DTYPE, PHASE_WORDS


def spectral_state(
    values: torch.Tensor,
    eigenvectors: torch.Tensor,
    rhs: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Compute the finite-resolution QPE and inverse directly in each eigenmode."""
    steps = torch.arange(PHASE_WORDS, dtype=DTYPE)
    inverse_dft = torch.exp(
        -2j * math.pi * torch.outer(steps, steps) / PHASE_WORDS
    ) / math.sqrt(PHASE_WORDS)
    evolved = torch.exp(2j * math.pi * values[:, None] * steps[None, :])
    forward = evolved @ inverse_dft.T / math.sqrt(PHASE_WORDS)
    weighted = torch.stack(
        (torch.cos(angles / 2) * forward, torch.sin(angles / 2) * forward)
    ).to(COMPLEX)
    restored = (weighted @ inverse_dft.conj()) * evolved.conj() / math.sqrt(PHASE_WORDS)
    # The final H tensor product applies the Walsh transform to each phase row.
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=COMPLEX) / math.sqrt(2)
    walsh = hadamard
    for _ in range(5):
        walsh = torch.kron(walsh, hadamard)
    restored = restored @ walsh * math.sqrt(PHASE_WORDS)
    projected = eigenvectors.T @ rhs
    return eigenvectors.to(COMPLEX) @ (projected[None, :, None] * restored)


def reference_swap(logical: torch.Tensor, compared: torch.Tensor) -> torch.Tensor:
    """Predict all four outcomes by contracting the phase-entangled overlap."""
    normalized = compared.to(COMPLEX) / compared.norm()
    overlap = (normalized.conj() @ logical).abs().square().sum(dim=-1)
    weight = logical.abs().square().sum(dim=(1, 2))
    return torch.stack(((weight + overlap) / 2, (weight - overlap) / 2))
