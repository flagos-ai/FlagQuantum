"""Exact local density-matrix simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch

from ..core.ir import CircuitIR
from ..core.runtime_config import get_runtime_config
from ..noise import KrausChannel
from .gate_matrix import gate_matrix


def _complex_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    return dtype or getattr(torch, get_runtime_config().complex_dtype)


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


def _basis_bits(index: int, n_qubits: int) -> list[int]:
    return [(index >> (n_qubits - 1 - qubit)) & 1 for qubit in range(n_qubits)]


def _bits_to_index(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _validate_qubits(qubits: Sequence[int], n_qubits: int) -> tuple[int, ...]:
    """Refuse a qubit the Hilbert space does not have, and normalize the rest.

    Every function here addresses qubits by indexing a size-``2**n_qubits`` axis,
    and a negative index is a valid Python index, so ``-1`` silently addressed
    the last qubit: ``apply_unitary_density`` returned a density matrix identical
    to the one for the qubit the caller did not name. An index past the last qubit
    raised a bare ``IndexError`` from the indexing expression instead of naming
    the offending qubit.
    """
    normalized = tuple(int(qubit) for qubit in qubits)
    if any(qubit < 0 or qubit >= n_qubits for qubit in normalized):
        raise ValueError("qubit index out of range")
    return normalized


def expand_operator(
    matrix: torch.Tensor,
    qubits: Sequence[int],
    n_qubits: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Expand a k-qubit operator into the full Hilbert space."""

    qubits = _validate_qubits(qubits, n_qubits)
    matrix = torch.as_tensor(matrix, dtype=_complex_dtype(dtype), device=device)
    if matrix.ndim == 3:
        return torch.stack(
            [
                expand_operator(
                    item, qubits, n_qubits, dtype=matrix.dtype, device=matrix.device
                )
                for item in matrix
            ]
        )
    dim = 2**n_qubits
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator matrix must be square")
    if matrix.shape[0] != 2 ** len(qubits):
        raise ValueError("operator dimension does not match the target qubits")
    full = torch.zeros(dim, dim, dtype=matrix.dtype, device=matrix.device)
    for col in range(dim):
        bits = _basis_bits(col, n_qubits)
        sub_col = _bits_to_index(tuple(bits[qubit] for qubit in qubits))
        for sub_row in range(2 ** len(qubits)):
            row_bits = list(bits)
            replacement = _basis_bits(sub_row, len(qubits))
            for offset, qubit in enumerate(qubits):
                row_bits[qubit] = replacement[offset]
            row = _bits_to_index(row_bits)
            full[row, col] = matrix[sub_row, sub_col]
    return full


def apply_unitary_density(
    rho: torch.Tensor,
    matrix: torch.Tensor,
    qubits: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    """Apply a unitary matrix to a batched density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    full = expand_operator(matrix, qubits, n_qubits, dtype=rho.dtype, device=rho.device)
    if full.ndim == 2:
        full = full.expand(rho.shape[0], -1, -1)
    return torch.bmm(torch.bmm(full, rho), torch.conj(full).transpose(-1, -2))


def apply_kraus_density(
    rho: torch.Tensor,
    kraus: KrausChannel | Sequence[torch.Tensor],
    qubits: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    """Apply Kraus operators to a batched density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    ops = kraus.kraus if isinstance(kraus, KrausChannel) else tuple(kraus)
    if not ops:
        # `out` starts at zero, so an empty channel returned the zero map: a
        # trace-0 "state" that silently zeroed every later expectation.
        raise ValueError("Kraus channel must contain at least one operator")
    out = torch.zeros_like(rho)
    for op in ops:
        full = expand_operator(op, qubits, n_qubits, dtype=rho.dtype, device=rho.device)
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
        out_dtype = dtype or getattr(torch, get_runtime_config().complex_dtype)
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
    qubits: Iterable[int] | int | None = None,
) -> torch.Tensor:
    """Compute Z expectations from a density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    n_qubits = int(torch.log2(torch.tensor(rho.shape[-1], dtype=torch.float32)).item())
    if qubits is None:
        target_qubits = tuple(range(n_qubits))
    elif isinstance(qubits, int):
        target_qubits = (qubits,)
    else:
        target_qubits = tuple(int(qubit) for qubit in qubits)

    probs = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1))
    shaped = probs.reshape((rho.shape[0],) + (2,) * n_qubits)
    values = []
    for qubit in _validate_qubits(target_qubits, n_qubits):
        axes = tuple(axis for axis in range(1, n_qubits + 1) if axis != qubit + 1)
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
