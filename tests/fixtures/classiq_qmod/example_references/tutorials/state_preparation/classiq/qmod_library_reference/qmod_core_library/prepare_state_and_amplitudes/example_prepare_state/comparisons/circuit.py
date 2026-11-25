"""Compare the exact original Qmod, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.functions.state_preparation.example_prepare_state.main import (
    main,
)
from flagquantum import synth
from tools.validate_native_source_states import compare_states


def reference_circuit() -> fq.Circuit:
    amplitudes = torch.tensor(
        [0.05, 0.11, 0.13, 0.23, 0.27, 0.12, 0.03, 0.06], dtype=torch.float64
    )
    amplitudes = amplitudes.sqrt()
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
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("x",),
    )


if __name__ == "__main__":
    run_example()
