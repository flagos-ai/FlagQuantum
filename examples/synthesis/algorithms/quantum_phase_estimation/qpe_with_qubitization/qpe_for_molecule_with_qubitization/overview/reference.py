"""Verify both phase-estimation routes with dense local Circuit references."""

import math

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.physics import (
    PAULI_WORDS,
)

from .model import Model
from .quantum import BLOCK, DATA, LOGICAL_WIDTH, PHASE


def walk_matrix(model: Model) -> torch.Tensor:
    size = 16
    dtype = torch.complex128
    paulis = {
        "I": torch.eye(2, dtype=dtype),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=dtype),
    }
    blocks = []
    for word, coefficient in zip(
        PAULI_WORDS, (*model.coefficients, model.coefficients[-1]), strict=True
    ):
        matrix = torch.ones(1, 1, dtype=dtype)
        for axis in reversed(word):
            matrix = torch.kron(matrix, paulis[axis])
        blocks.append(math.copysign(1, coefficient) * matrix)
    selector = torch.zeros((size**2, size**2), dtype=dtype)
    for i, selected in enumerate(blocks):
        first = size * i
        selector[first : first + size, first : first + size] = selected
    preparation = torch.eye(size, dtype=dtype)
    for level in range(4):
        stride = size >> level
        for first in range(0, size, stride):
            middle = first + stride // 2
            last = first + stride
            left = math.fsum(model.probabilities[first:middle])
            right = math.fsum(model.probabilities[middle:last])
            if left + right == 0:
                continue
            cosine = math.sqrt(left / (left + right))
            sine = math.sqrt(right / (left + right))
            a = preparation[first:middle].clone()
            b = preparation[middle:last].clone()
            preparation[first:middle] = cosine * a - sine * b
            preparation[middle:last] = sine * a + cosine * b
    distribution = torch.kron(preparation, torch.eye(size, dtype=dtype))
    reflection = torch.diag(
        torch.tensor([1] * size + [-1] * (size**2 - size), dtype=dtype)
    )
    walk: torch.Tensor = reflection @ distribution.mH @ selector @ distribution
    return walk


def circuit(model: Model, optimized: bool) -> fq.Circuit:
    dtype = torch.complex128
    circuit = fq.Circuit(LOGICAL_WIDTH, dtype=dtype).x(DATA[3]).x(DATA[1])
    for wire in PHASE:
        circuit.h(wire)
    walk = walk_matrix(model)
    identity = torch.eye(256, dtype=dtype)
    mirror_sign = torch.ones(32, dtype=dtype)
    mirror_sign[0] = -1
    controlled_reflection = torch.diag(mirror_sign)
    for bit, wire in enumerate(reversed(PHASE)):
        operator = torch.linalg.matrix_power(
            walk, 2 ** (bit - 1 if optimized and bit else bit)
        )
        if optimized and bit:
            circuit.any(wire, *BLOCK, matrix=controlled_reflection)
            circuit.any(*BLOCK, *DATA, matrix=operator)
            circuit.any(wire, *BLOCK, matrix=controlled_reflection)
        else:
            controlled = torch.zeros((512, 512), dtype=dtype)
            controlled[:256, :256] = identity
            controlled[256:, 256:] = operator
            circuit.any(wire, *BLOCK, *DATA, matrix=controlled)
    frequencies = torch.arange(32, dtype=torch.float64)
    inverse = torch.exp(
        -2j * math.pi * torch.outer(frequencies, frequencies) / 32
    ) / math.sqrt(32)
    return circuit.any(*PHASE, matrix=inverse)
