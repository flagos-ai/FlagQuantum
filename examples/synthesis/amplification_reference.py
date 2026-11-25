"""Independent dense amplitude amplification on small initialized registers."""

import torch

import flagquantum as fq


def amplified_circuit(preparation: torch.Tensor, k: int) -> fq.Circuit:
    """Apply A once, then (-A R0 A† Z_indicator)^k without phase adjustment."""
    dimension = preparation.shape[0]
    zero = torch.eye(dimension, dtype=torch.complex128)
    zero[0, 0] = -1
    oracle = torch.diag(
        torch.tensor(
            [1] * (dimension // 2) + [-1] * (dimension // 2), dtype=torch.complex128
        )
    )
    grover = -preparation @ zero @ preparation.mH @ oracle
    matrix = torch.linalg.matrix_power(grover, k) @ preparation
    width = dimension.bit_length() - 1
    return fq.Circuit(width, dtype=torch.complex128).any(*range(width), matrix=matrix)
