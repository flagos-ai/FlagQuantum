"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.state_preparation.example_prepare_amplitudes.main import (
    main,
)
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    amplitudes = torch.tensor(
        [
            -0.540061724867322,
            -0.38575837490523,
            -0.231455024943138,
            -0.077151674981046,
            0.077151674981046,
            0.231455024943138,
            0.38575837490523,
            0.540061724867322,
        ],
        dtype=torch.float64,
    )
    identity = torch.eye(8, dtype=torch.complex128)
    difference = identity[:, 0] - amplitudes
    matrix = (
        identity
        - 2 * torch.outer(difference, difference.conj()) / difference.norm().square()
    )
    return fq.Circuit(3, dtype=torch.complex128).any(0, 1, 2, matrix=matrix)


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        reference_circuit(),
        ("x",),
    )


if __name__ == "__main__":
    run_example()
