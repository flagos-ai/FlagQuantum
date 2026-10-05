"""Dense Pauli-product numerics shared by algorithm workflows."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .matrices import GATE_MAT_DICT
from .statevector.operations import _apply_matrix

PauliProduct = Sequence[tuple[int, str]]


def infer_n_qubits_from_dense_state(state: torch.Tensor) -> int:
    """Infer a qubit count from a dense statevector or density matrix."""

    dimension = int(state.shape[-1])
    n_qubits = dimension.bit_length() - 1
    if dimension <= 0 or 2**n_qubits != dimension:
        raise ValueError("State dimension must be a power of two.")
    return n_qubits


def pauli_product_operator(
    operators: PauliProduct,
    n_qubits: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Materialize one dense Pauli-product operator."""

    by_qubit = {int(qubit): str(name).lower() for qubit, name in operators}
    if any(qubit < 0 or qubit >= n_qubits for qubit in by_qubit):
        raise ValueError("Pauli operator qubit index out of range")
    result = torch.ones(1, 1, dtype=dtype, device=device)
    for qubit in range(int(n_qubits)):
        matrix = GATE_MAT_DICT[by_qubit.get(qubit, "i")]
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Pauli products require fixed gate matrices.")
        matrix = matrix.to(
            device=device,
            dtype=dtype,
        )
        result = torch.kron(result, matrix)
    return result


def pauli_product_statevector_expectation(
    state: torch.Tensor,
    operators: PauliProduct,
    n_qubits: int,
) -> torch.Tensor:
    """Evaluate one Pauli product on a dense statevector batch."""

    batch = state.reshape(1, -1) if state.ndim == 1 else state
    transformed = batch
    for qubit, name in operators:
        matrix = GATE_MAT_DICT[str(name).lower()]
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Pauli products require fixed gate matrices.")
        matrix = matrix.to(
            device=batch.device,
            dtype=batch.dtype,
        )
        transformed = _apply_matrix(transformed, matrix, (int(qubit),), n_qubits)
    return torch.real((torch.conj(batch) * transformed).sum(dim=-1))


def pauli_product_density_expectation(
    density: torch.Tensor,
    operators: PauliProduct,
    n_qubits: int,
) -> torch.Tensor:
    """Evaluate one Pauli product on a dense density-matrix batch."""

    batch = density.reshape(1, *density.shape) if density.ndim == 2 else density
    operator = pauli_product_operator(
        operators,
        n_qubits,
        dtype=batch.dtype,
        device=batch.device,
    )
    values = torch.einsum("bij,ji->b", batch, operator)
    return torch.real(values)


__all__ = (
    "infer_n_qubits_from_dense_state",
    "pauli_product_density_expectation",
    "pauli_product_operator",
    "pauli_product_statevector_expectation",
)
