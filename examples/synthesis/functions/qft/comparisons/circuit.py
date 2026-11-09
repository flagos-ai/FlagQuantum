"""Use the Fourier definition rather than the compiler's gate decomposition."""

import math

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qft.main import build_program


def fourier_matrix(width: int) -> torch.Tensor:
    size = 1 << width
    indices = torch.arange(size, dtype=torch.float64)
    return torch.exp(2j * math.pi / size * torch.outer(indices, indices)) / math.sqrt(
        size
    )


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(4, dtype=torch.complex128).any(
        *range(4), matrix=fourier_matrix(4)
    )


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x",))


if __name__ == "__main__":
    run_example()
