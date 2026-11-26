"""QAOA circuit construction and objective evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from flagquantum.algorithms.primitives.hamiltonian import Hamiltonian
from flagquantum.circuit import Circuit
from flagquantum.core.runtime_config import get_runtime_config


def qaoa_circuit(
    n_qubits: int,
    edges: Iterable[tuple[int, int] | tuple[int, int, float]],
    gammas: torch.Tensor | Sequence[float],
    betas: torch.Tensor | Sequence[float],
    *,
    device: torch.device | str = "cpu",
) -> Circuit:
    """Build a QAOA circuit for weighted ZZ cost edges."""

    gamma_values = torch.as_tensor(gammas, dtype=torch.float32, device=device).reshape(
        -1
    )
    beta_values = torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(-1)
    if gamma_values.numel() != beta_values.numel():
        raise ValueError("gammas and betas must have the same number of layers.")

    edge_tuple = tuple(edges)
    circuit = Circuit(
        n_qubits,
        device=device,
        dtype=getattr(torch, get_runtime_config().complex_dtype),
    )
    for qubit in range(n_qubits):
        circuit.gate("h", qubit)

    for gamma, beta in zip(gamma_values, beta_values, strict=True):
        for edge in edge_tuple:
            if len(edge) == 2:
                src, dst = edge
                weight = 1.0
            else:
                src, dst, weight = edge
            circuit.gate("rzz", (int(src), int(dst)), theta=2.0 * gamma * float(weight))
        for qubit in range(n_qubits):
            circuit.gate("rx", qubit, theta=2.0 * beta)
    return circuit


def qaoa_loss(
    n_qubits: int,
    edges: Iterable[tuple[int, int] | tuple[int, int, float]],
    gammas: torch.Tensor | Sequence[float],
    betas: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    reduction: str = "mean",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Evaluate a differentiable QAOA objective."""

    circuit = qaoa_circuit(n_qubits, edges, gammas, betas, device=device)
    values = hamiltonian.expectation(circuit)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")


__all__ = ("qaoa_circuit", "qaoa_loss")

# Preserve the public algorithm and serialized result identities.
for _identity_export in (
    qaoa_circuit,
    qaoa_loss,
):
    _identity_export.__module__ = "flagquantum.algorithms.core"
del _identity_export
