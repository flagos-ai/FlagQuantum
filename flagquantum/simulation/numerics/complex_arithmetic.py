"""Portable complex arithmetic built from real PyTorch operations."""

from __future__ import annotations

import torch


def complex_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply complex tensors without requiring complex ``aten::mul`` backward."""

    if not left.is_complex() or not right.is_complex():
        return left * right
    real = left.real * right.real - left.imag * right.imag
    imag = left.real * right.imag + left.imag * right.real
    return torch.complex(real, imag)


def complex_conj(value: torch.Tensor) -> torch.Tensor:
    """Conjugate without requiring a backend complex-conjugate kernel."""

    if not value.is_complex():
        return value
    return torch.complex(value.real, -value.imag)


__all__ = ("complex_conj", "complex_mul")
