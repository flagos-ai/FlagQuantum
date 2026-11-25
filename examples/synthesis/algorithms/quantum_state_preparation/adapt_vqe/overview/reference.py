"""Check native Pauli phases through independently assembled dense gates."""

import math

import torch

import flagquantum as fq

from .model import DTYPE, POOL, dense_word


def reference_circuit(
    indices: tuple[int, ...], angles: tuple[float, ...]
) -> fq.Circuit:
    """Apply each source Pauli exponential as an explicit 16 by 16 matrix."""
    circuit = fq.Circuit(4, dtype=DTYPE)
    identity = torch.eye(16, dtype=DTYPE)
    for index, angle in zip(indices, angles, strict=True):
        matrix = math.cos(angle) * identity - 1j * math.sin(angle) * dense_word(
            POOL[index]
        )
        circuit.any(0, 1, 2, 3, matrix=matrix)
    return circuit


def finite_difference_gradients(
    matrix: torch.Tensor,
    indices: tuple[int, ...],
    angles: tuple[float, ...],
    delta: float = 1e-5,
) -> tuple[float, ...]:
    """Differentiate the entire pool independently by signed dense evolution."""
    state: torch.Tensor = (
        fq.run(reference_circuit(indices, angles)).statevector().flatten()
    )
    values: list[float] = []
    for word in POOL:
        change = -1j * math.sin(delta) * (dense_word(word) @ state)
        even = math.cos(delta) * state
        positive, negative = even + change, even - change
        top = torch.vdot(positive, matrix @ positive).real
        bottom = torch.vdot(negative, matrix @ negative).real
        values.append(float((top - bottom) / (2 * delta)))
    return tuple(values)
