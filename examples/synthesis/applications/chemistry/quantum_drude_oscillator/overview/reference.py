"""Independent positional Circuit and occupation-basis matrix references."""

import math
from collections.abc import Sequence
from itertools import product

import numpy as np
import torch

import flagquantum as fq

from .model import RealArray


def reference_circuit(
    oscillator_count: int, angles: Sequence[float] | torch.Tensor
) -> fq.Circuit:
    """Expand both source pair schedules directly into primitive gates."""
    firsts = (0, 2, 1) if oscillator_count == 2 else (0, 2, 4, 1, 3) * 2
    width = 2 * oscillator_count
    circuit = fq.Circuit(width, dtype=torch.complex128)
    quarter = math.pi / 2
    for layer, first in enumerate(firsts):
        a, b = width - first - 1, width - first - 2
        p0, p1, p2, p3 = angles[4 * layer : 4 * layer + 4]
        circuit.rz(a, theta=quarter).rz(b, theta=quarter).ry(b, theta=quarter)
        circuit.cx(b, a)
        circuit.ry(a, theta=p0).rz(a, theta=p1)
        circuit.ry(b, theta=p2).rz(b, theta=p3)
        circuit.cx(b, a).ry(b, theta=-quarter)
        circuit.rz(a, theta=-quarter).rz(b, theta=-quarter)
    return circuit


def transition_matrix(
    oscillator_count: int, dimension: int, distance: float
) -> RealArray:
    """Enumerate every occupation and two-oscillator adjacency independently."""
    states = [
        levels[::-1] for levels in product(range(dimension), repeat=oscillator_count)
    ]
    words = {levels: word for word, levels in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=np.float64)
    for word, levels in enumerate(states):
        matrix[word, word] = oscillator_count + 2 * sum(levels)
        for first in range(oscillator_count):
            for second in range(first + 1, oscillator_count):
                for step_a, step_b in product((-1, 1), repeat=2):
                    a, b = levels[first] + step_a, levels[second] + step_b
                    if not (0 <= a < dimension and 0 <= b < dimension):
                        continue
                    destination = list(levels)
                    destination[first], destination[second] = a, b
                    weight = math.sqrt(max(a, levels[first]) * max(b, levels[second]))
                    matrix[words[tuple(destination)], word] = (
                        -29 * weight / ((second - first) * distance) ** 3
                    )
    return matrix


def analytical_position_product(bits: int) -> RealArray:
    """Construct the original one- and two-bit analytic Pauli expansions."""
    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.diag((1, -1)).astype(np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    if bits == 1:
        return np.asarray((np.kron(x, x) / 2).real)
    if bits != 2:
        raise ValueError("the analytical checks use one or two bits per oscillator")
    # Position expansion on (high bit, low bit) prior to multiplying oscillators.
    root_three = math.sqrt(3)
    local = (
        (root_three + 1) * np.kron(identity, x)
        + (1 - root_three) * np.kron(z, x)
        + math.sqrt(2) * (np.kron(x, x) + np.kron(y, y))
    ) / (2 * math.sqrt(2))
    return np.asarray(np.kron(local, local).real)
