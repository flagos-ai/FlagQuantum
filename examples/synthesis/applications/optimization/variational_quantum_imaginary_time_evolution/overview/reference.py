"""Build the same ordered ansatz independently as dense Circuit products."""

import math

import torch

import flagquantum as fq

from .model import CutModel


def reference_circuit(model: CutModel, angles: tuple[float, ...]) -> fq.Circuit:
    """Use physical Circuit axes and explicitly ordered dense Pauli exponentials."""
    dtype = torch.complex128
    single_identity = torch.eye(2, dtype=dtype)
    identity = torch.eye(16, dtype=dtype)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    z = torch.diag(torch.tensor([1, -1], dtype=dtype))
    circuit = fq.Circuit(4, dtype=dtype)
    for wire in range(4):
        circuit.h(wire)
    for (z_node, y_node), angle in zip(model.generators, angles, strict=True):
        axes = [single_identity] * 4
        axes[3 - z_node], axes[3 - y_node] = z, y
        matrix = torch.kron(torch.kron(axes[0], axes[1]), torch.kron(axes[2], axes[3]))
        exponential = math.cos(angle) * identity - 1j * math.sin(angle) * matrix
        circuit.any(0, 1, 2, 3, matrix=exponential)
    return circuit


def direct_cut(model: CutModel, word: int) -> float:
    """Compute the positive cut weight directly from the edges."""
    return sum(
        weight for u, v, weight in model.edges if ((word >> u) ^ (word >> v)) & 1
    )
