"""Compute the original workshop data with the existing PyTorch dependency."""

import math

import torch

MATRIX = torch.tensor(
    [
        [0.28, -0.01, 0.02, -0.1],
        [-0.01, 0.5, -0.22, -0.07],
        [0.02, -0.22, 0.43, -0.05],
        [-0.1, -0.07, -0.05, 0.42],
    ],
    dtype=torch.float64,
)
RAW_B = torch.tensor([1.0, 2.0, 4.0, 3.0], dtype=torch.float64)
B = RAW_B / torch.linalg.vector_norm(RAW_B)
QPE_SIZE = 4
RECIPROCAL_SCALE = 1 / 2**QPE_SIZE
UNITARY = tuple(
    tuple(complex(v) for v in row)
    for row in torch.matrix_exp(2j * math.pi * MATRIX).tolist()
)
ANGLES = tuple(
    0.0 if word == 0 else 2 * math.asin(1 / word) for word in range(2**QPE_SIZE)
)
AMPLITUDES = tuple(float(v) for v in B)
