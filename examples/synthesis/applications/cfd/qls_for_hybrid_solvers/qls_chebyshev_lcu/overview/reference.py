"""Check the complete controlled walk without the binary-control circuit."""

import math

import torch

from .encoding import CTYPE, DTYPE


def preparation_matrix(probabilities: torch.Tensor, uniform_bits: int) -> torch.Tensor:
    """Build the probability tree and its uniform tail directly in basis order."""
    group = 1 << uniform_bits
    reduced = probabilities.reshape(-1, group).sum(dim=1)
    size = reduced.numel()
    high = torch.eye(size, dtype=CTYPE)
    stride = size
    while stride > 1:
        half = stride // 2
        for first in range(0, size, stride):
            mid = first + half
            end = first + stride
            left = float(reduced[first:mid].sum())
            right = float(reduced[mid:end].sum())
            if left + right == 0:
                continue
            cosine = math.sqrt(left / (left + right))
            sine = math.sqrt(right / (left + right))
            a, b = high[first:mid].clone(), high[mid:end].clone()
            high[first:mid] = cosine * a - sine * b
            high[mid:end] = sine * a + cosine * b
        stride = half
    low = torch.eye(1, dtype=CTYPE)
    h = torch.tensor(((1, 1), (1, -1)), dtype=CTYPE) / math.sqrt(2)
    for _ in range(uniform_bits):
        low = torch.kron(low, h)
    return torch.kron(high, low)


def full_state(
    block_encoding: torch.Tensor,
    rhs: torch.Tensor,
    probabilities: torch.Tensor,
    uniform_bits: int,
) -> torch.Tensor:
    prep = preparation_matrix(probabilities, uniform_bits)
    block_size = block_encoding.size(0)
    signs = -torch.ones(block_size, dtype=DTYPE)
    signs[: rhs.numel()] = 1
    state = torch.zeros(block_size, dtype=CTYPE)
    state[: rhs.numel()] = rhs / rhs.norm()
    rows = torch.empty((probabilities.numel(), block_size), dtype=CTYPE)
    inverse = block_encoding.mH
    for row in range(probabilities.numel()):
        rows[row] = signs * (inverse @ (signs * state)) * prep[row, 0] * (-1) ** row
        state = block_encoding @ (signs * (inverse @ (signs * state)))
    output: torch.Tensor = prep.mH @ rows
    return output.flatten()
