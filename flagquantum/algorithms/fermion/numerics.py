"""Bounded dense inspection helpers, separate from symbolic algebra."""

from __future__ import annotations

import torch

from ...core.algebra.pauli import PauliOperator, count


def pauli_matrix(
    operator: PauliOperator, *, n_qubits: int, max_qubits: int = 10
) -> torch.Tensor:
    """Materialize a small complex128 matrix in FlagQuantum MSB order.

    Dense storage costs 16 * 4**n_qubits bytes. The explicit guard prevents this
    inspection helper from silently becoming a large-system execution path.
    """
    width = count(n_qubits, "n_qubits")
    limit = count(max_qubits, "max_qubits")
    if width > limit:
        raise ValueError("dense matrix exceeds the explicit max_qubits budget")
    axes = {
        "i": torch.eye(2, dtype=torch.complex128),
        "x": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        "y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
        "z": torch.diag(torch.tensor([1, -1], dtype=torch.complex128)),
    }
    result = torch.zeros((1 << width, 1 << width), dtype=torch.complex128)
    for term in operator.terms:
        factors = dict(term.factors)
        if any(q >= width for q in factors):
            raise ValueError("operator acts outside the declared register")
        matrix = torch.ones((1, 1), dtype=torch.complex128)
        for qubit in range(width):
            matrix = torch.kron(matrix, axes[factors.get(qubit, "i")])
        result += term.coefficient * matrix
    return result


def static_tensor(value: torch.Tensor, name: str, *, ndim: int) -> torch.Tensor:
    """Own a finite CPU complex128 snapshot; refuse a detached gradient path."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.requires_grad:
        raise ValueError(
            f"{name} is static preprocessing; autograd inputs are unsupported"
        )
    if value.ndim != ndim or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with {ndim} dimensions")
    if value.dtype == torch.bool:
        raise TypeError(f"{name} must contain numeric matrix entries")
    return value.to(device="cpu", dtype=torch.complex128).clone()
