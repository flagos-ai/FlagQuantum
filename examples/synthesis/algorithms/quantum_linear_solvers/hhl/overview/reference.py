"""Independent dense controlled evolutions and reciprocal rotation table."""

import math

import torch

import flagquantum as fq
from examples.synthesis.hhl_reference import householder

from .ansatz import Route
from .physics import AMPLITUDES, ANGLES, MATRIX, PAULI_TERMS, PAULIS, REPETITIONS


def reference_circuit(route: Route) -> fq.Circuit:
    if route not in ("exact", "trotter"):
        raise ValueError("use exact or trotter")
    identity = torch.eye(4, dtype=torch.complex128)
    circuit = fq.Circuit(7, dtype=torch.complex128).any(
        0, 1, matrix=householder(AMPLITUDES)
    )
    powers = []
    for bit, repetitions in enumerate(REPETITIONS):
        power = 1 << bit
        if route == "exact":
            evolution = torch.matrix_exp(2j * math.pi * power * MATRIX)
        else:
            step = identity.clone()
            for word, coefficient in PAULI_TERMS:
                pauli = torch.kron(PAULIS[word[0]], PAULIS[word[1]])
                angle = 2 * math.pi * power * coefficient / repetitions
                step = (
                    math.cos(angle) * identity + 1j * math.sin(angle) * pauli
                ) @ step
            evolution = torch.linalg.matrix_power(step, repetitions)
        controlled = torch.eye(8, dtype=torch.complex128)
        controlled[4:, 4:] = evolution
        wire = 5 - bit
        circuit.h(wire).any(wire, 0, 1, matrix=controlled)
        powers.append((wire, controlled))
    axis = torch.arange(16, dtype=torch.float64)
    inverse = torch.exp(-2j * math.pi * torch.outer(axis, axis) / 16) / 4
    circuit.any(2, 3, 4, 5, matrix=inverse)
    table = torch.zeros((32, 32), dtype=torch.complex128)
    for selector_word, angle in enumerate(ANGLES):
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        table[
            2 * selector_word : 2 * selector_word + 2,
            2 * selector_word : 2 * selector_word + 2,
        ] = torch.tensor([[c, -s], [s, c]], dtype=torch.complex128)
    circuit.any(2, 3, 4, 5, 6, matrix=table)
    circuit.any(2, 3, 4, 5, matrix=inverse.mH)
    for wire, controlled in reversed(powers):
        circuit.any(wire, 0, 1, matrix=controlled.mH)
    for wire in (2, 3, 4, 5):
        circuit.h(wire)
    return circuit
