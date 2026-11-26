"""Finite qubit-operator matrix embedding, without state evolution."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from flagquantum.core.runtime_config import get_runtime_config


def _complex_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    return dtype or getattr(torch, get_runtime_config().complex_dtype)


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


for _identity_export in (
    _complex_dtype,
    _basis_bits,
    _bits_to_index,
    _validate_qubits,
    expand_operator,
):
    _identity_export.__module__ = "flagquantum.simulation.density_matrix"
del _identity_export
