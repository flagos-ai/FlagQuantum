"""Numerical kernels for exact density-matrix evolution."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ....noise import KrausChannel
from ....ops.matrices import get_global_precision


def _complex_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    return dtype or get_global_precision()


def density_matrix(circuit_or_state: Any) -> torch.Tensor:
    """Return a batched density matrix from a circuit or statevector."""

    state = (
        circuit_or_state.state()
        if hasattr(circuit_or_state, "state")
        else circuit_or_state
    )
    state = torch.as_tensor(state)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    return state.unsqueeze(-1) * torch.conj(state).unsqueeze(-2)


def _basis_bits(index: int, n_wires: int) -> list[int]:
    return [(index >> (n_wires - 1 - wire)) & 1 for wire in range(n_wires)]


def _bits_to_index(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def expand_operator(
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Expand a k-wire operator into the full Hilbert space."""

    wires = tuple(int(wire) for wire in wires)
    matrix = torch.as_tensor(matrix, dtype=_complex_dtype(dtype), device=device)
    if matrix.ndim == 3:
        return torch.stack(
            [
                expand_operator(
                    item, wires, n_wires, dtype=matrix.dtype, device=matrix.device
                )
                for item in matrix
            ]
        )
    dim = 2**n_wires
    full = torch.zeros(dim, dim, dtype=matrix.dtype, device=matrix.device)
    for col in range(dim):
        bits = _basis_bits(col, n_wires)
        sub_col = _bits_to_index(tuple(bits[wire] for wire in wires))
        for sub_row in range(2 ** len(wires)):
            row_bits = list(bits)
            replacement = _basis_bits(sub_row, len(wires))
            for offset, wire in enumerate(wires):
                row_bits[wire] = replacement[offset]
            row = _bits_to_index(row_bits)
            full[row, col] = matrix[sub_row, sub_col]
    return full


def apply_unitary_density(
    rho: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply a unitary matrix to a batched density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    full = expand_operator(matrix, wires, n_wires, dtype=rho.dtype, device=rho.device)
    if full.ndim == 2:
        full = full.expand(rho.shape[0], -1, -1)
    return torch.bmm(torch.bmm(full, rho), torch.conj(full).transpose(-1, -2))


def apply_kraus_density(
    rho: torch.Tensor,
    kraus: KrausChannel | Sequence[torch.Tensor],
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply Kraus operators to a batched density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    ops = kraus.kraus if isinstance(kraus, KrausChannel) else tuple(kraus)
    out = torch.zeros_like(rho)
    for op in ops:
        full = expand_operator(op, wires, n_wires, dtype=rho.dtype, device=rho.device)
        if full.ndim == 2:
            full = full.expand(rho.shape[0], -1, -1)
        out = out + torch.bmm(torch.bmm(full, rho), torch.conj(full).transpose(-1, -2))
    return out


__all__ = (
    "apply_kraus_density",
    "apply_unitary_density",
    "density_matrix",
    "expand_operator",
)
