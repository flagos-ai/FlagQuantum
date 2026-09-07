"""Dense Pauli-product numerics shared by algorithm workflows."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .matrices import GATE_MAT_DICT
from .statevector.operations import _apply_matrix

PauliProduct = Sequence[tuple[int, str]]


def infer_n_wires_from_dense_state(state: torch.Tensor) -> int:
    """Infer a qubit count from a dense statevector or density matrix."""

    dimension = int(state.shape[-1])
    n_wires = dimension.bit_length() - 1
    if dimension <= 0 or 2**n_wires != dimension:
        raise ValueError("State dimension must be a power of two.")
    return n_wires


def pauli_product_operator(
    operators: PauliProduct,
    n_wires: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Materialize one dense Pauli-product operator."""

    by_wire = {int(wire): str(name).lower() for wire, name in operators}
    result = torch.ones(1, 1, dtype=dtype, device=device)
    for wire in range(int(n_wires)):
        matrix = GATE_MAT_DICT[by_wire.get(wire, "i")].to(
            device=device,
            dtype=dtype,
        )
        result = torch.kron(result, matrix)
    return result


def pauli_product_statevector_expectation(
    state: torch.Tensor,
    operators: PauliProduct,
    n_wires: int,
) -> torch.Tensor:
    """Evaluate one Pauli product on a dense statevector batch."""

    batch = state.reshape(1, -1) if state.ndim == 1 else state
    transformed = batch
    for wire, name in operators:
        matrix = GATE_MAT_DICT[str(name).lower()].to(
            device=batch.device,
            dtype=batch.dtype,
        )
        transformed = _apply_matrix(transformed, matrix, (int(wire),), n_wires)
    return torch.real((torch.conj(batch) * transformed).sum(dim=-1))


def pauli_product_density_expectation(
    density: torch.Tensor,
    operators: PauliProduct,
    n_wires: int,
) -> torch.Tensor:
    """Evaluate one Pauli product on a dense density-matrix batch."""

    batch = density.reshape(1, *density.shape) if density.ndim == 2 else density
    operator = pauli_product_operator(
        operators,
        n_wires,
        dtype=batch.dtype,
        device=batch.device,
    )
    values = torch.diagonal(torch.matmul(batch, operator), dim1=-2, dim2=-1).sum(dim=-1)
    return torch.real(values)


__all__ = (
    "infer_n_wires_from_dense_state",
    "pauli_product_density_expectation",
    "pauli_product_operator",
    "pauli_product_statevector_expectation",
)
