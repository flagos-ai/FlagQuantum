"""Independent spectral matrix powers, Householder preparation and dense inverse DFT."""

import math

import torch

import flagquantum as fq

from .physics import MATRIX, RAW_B


def reference_circuit(precision: int) -> fq.Circuit:
    eigenvalues, eigenvectors = torch.linalg.eigh(MATRIX)
    vector = RAW_B.to(torch.complex128) / RAW_B.norm()
    identity = torch.eye(8, dtype=torch.complex128)
    delta = identity[:, 0] - vector
    prep = identity - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    count = 1 << precision
    words = torch.arange(count, dtype=torch.float64)
    fourier = torch.exp(2j * math.pi * torch.outer(words, words) / count) / math.sqrt(
        count
    )
    circuit = fq.Circuit(precision + 4, dtype=torch.complex128).any(
        0, 1, 2, matrix=prep
    )
    for bit in range(precision):
        circuit.h(3 + bit)
    for bit in range(precision):
        power = (
            eigenvectors.to(torch.complex128)
            * torch.exp(2j * math.pi * (1 << bit) * eigenvalues)
        ) @ eigenvectors.mT.to(torch.complex128)
        controlled = torch.eye(16, dtype=torch.complex128)
        controlled[8:, 8:] = power
        circuit.any(3 + precision - 1 - bit, 0, 1, 2, matrix=controlled)
    circuit.any(*range(3, 3 + precision), matrix=fourier.mH)
    rotations = torch.zeros((2 * count, 2 * count), dtype=torch.complex128)
    for word in range(count):
        amplitude = 0.0 if word == 0 else 1 / word
        cosine = math.sqrt(1 - amplitude**2)
        rotations[2 * word : 2 * word + 2, 2 * word : 2 * word + 2] = torch.tensor(
            [[cosine, -amplitude], [amplitude, cosine]], dtype=torch.complex128
        )
    circuit.any(*range(3, 4 + precision), matrix=rotations).any(
        *range(3, 3 + precision), matrix=fourier
    )
    for bit in reversed(range(precision)):
        power = (
            eigenvectors.to(torch.complex128)
            * torch.exp(-2j * math.pi * (1 << bit) * eigenvalues)
        ) @ eigenvectors.mT.to(torch.complex128)
        controlled = torch.eye(16, dtype=torch.complex128)
        controlled[8:, 8:] = power
        circuit.any(3 + precision - 1 - bit, 0, 1, 2, matrix=controlled)
    for bit in range(precision):
        circuit.h(3 + bit)
    return circuit
