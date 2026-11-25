"""Compare native FlagQuantum and independent Circuit amplitudes."""

import cmath

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.standard_gates.phase.main import main
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(1, dtype=torch.complex128).any(
        0, matrix=torch.diag(torch.tensor([1, cmath.exp(1j)], dtype=torch.complex128))
    )


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        reference_circuit(),
        ("q",),
    )


if __name__ == "__main__":
    run_example()
