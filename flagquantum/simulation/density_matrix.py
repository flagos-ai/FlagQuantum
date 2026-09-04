"""Exact local density-matrix simulation."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch

from ..core.ir import CircuitIR
from ..noise import KrausChannel
from ..ops.gate_matrix import gate_matrix
from ..ops.matrices import get_global_precision


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


def density_matrix_from_ir(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute unitary and channel IR instructions as a density matrix."""

    if hasattr(circuit_or_ir, "to_ir"):
        circuit = circuit_or_ir
        ir = circuit.to_ir()
        state = circuit.initial_state()
    elif isinstance(circuit_or_ir, CircuitIR):
        ir = circuit_or_ir
        out_dtype = dtype or get_global_precision()
        state = torch.zeros(bsz, 2**ir.n_wires, dtype=out_dtype, device=device)
        state[:, 0] = 1
    else:
        raise TypeError("density_matrix_from_ir expects a Circuit or CircuitIR.")

    rho = density_matrix(state)
    for instruction in ir:
        if instruction.metadata.get("is_channel"):
            rho = apply_kraus_density(
                rho,
                instruction.matrix,
                instruction.wires,
                ir.n_wires,
            )
            continue
        matrix = gate_matrix(
            instruction,
            bsz=rho.shape[0],
            device=rho.device,
            dtype=rho.dtype,
        )
        rho = apply_unitary_density(rho, matrix, instruction.wires, ir.n_wires)
    return rho


def expectation_z_density(
    rho: torch.Tensor,
    wires: Iterable[int] | int | None = None,
) -> torch.Tensor:
    """Compute Z expectations from a density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    n_wires = int(torch.log2(torch.tensor(rho.shape[-1], dtype=torch.float32)).item())
    if wires is None:
        target_wires = tuple(range(n_wires))
    elif isinstance(wires, int):
        target_wires = (wires,)
    else:
        target_wires = tuple(int(wire) for wire in wires)

    probs = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1))
    shaped = probs.reshape((rho.shape[0],) + (2,) * n_wires)
    values = []
    for wire in target_wires:
        axes = tuple(axis for axis in range(1, n_wires + 1) if axis != wire + 1)
        marginal = shaped.sum(dim=axes) if axes else shaped
        values.append(marginal[:, 0] - marginal[:, 1])
    return torch.stack(values, dim=-1)


__all__ = (
    "apply_kraus_density",
    "apply_unitary_density",
    "density_matrix",
    "density_matrix_from_ir",
    "expectation_z_density",
    "expand_operator",
)
