"""Independent dense matrix powers and inverse Fourier for small QPE examples."""

import math

import torch

import flagquantum as fq


def dense_qpe(matrix: torch.Tensor, preparation: torch.Tensor, bits: int) -> fq.Circuit:
    """Retain data instead of assuming dropped registers are clean."""
    dimension = matrix.shape[0]
    width = dimension.bit_length() - 1
    data = tuple(range(bits, bits + width))
    circuit = fq.Circuit(bits + width, dtype=torch.complex128)
    circuit.any(*data, matrix=preparation)
    for bit in range(bits):
        controlled = torch.eye(2 * dimension, dtype=torch.complex128)
        controlled[dimension:, dimension:] = torch.linalg.matrix_power(matrix, 1 << bit)
        circuit.h(bits - 1 - bit).any(bits - 1 - bit, *data, matrix=controlled)
    axis = torch.arange(1 << bits, dtype=torch.float64)
    inverse = torch.exp(-2j * math.pi * torch.outer(axis, axis) / (1 << bits))
    return circuit.any(*range(bits), matrix=inverse / math.sqrt(1 << bits))


def probability_preparation(probabilities: tuple[float, ...]) -> torch.Tensor:
    """Construct prefix-conditioned rotations, including all nonzero columns."""
    dimension = len(probabilities)
    preparation = torch.eye(dimension, dtype=torch.complex128)
    stride = dimension // 2
    while stride:
        layer = torch.eye(dimension, dtype=torch.complex128)
        for group in range(0, dimension, 2 * stride):
            left = math.sqrt(sum(probabilities[group : group + stride]))
            right = math.sqrt(sum(probabilities[group + stride : group + 2 * stride]))
            angle = math.atan2(right, left)
            for offset in range(stride):
                a, b = group + offset, group + offset + stride
                layer[a, a] = layer[b, b] = math.cos(angle)
                layer[a, b], layer[b, a] = -math.sin(angle), math.sin(angle)
        preparation = layer @ preparation
        stride //= 2
    return preparation
