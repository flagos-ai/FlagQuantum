"""Compare native FlagQuantum and independent Circuit amplitudes."""

import cmath
import math

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.standard_gates.u.main import main
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    (cosine, sine) = (math.cos(0.5), math.sin(0.5))
    matrix = cmath.exp(1.1j) * torch.tensor(
        [
            [cosine, -cmath.exp(1.5j) * sine],
            [cmath.exp(2j) * sine, cmath.exp(3.5j) * cosine],
        ],
        dtype=torch.complex128,
    )
    return fq.Circuit(1, dtype=torch.complex128).any(0, matrix=matrix)


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        reference_circuit(),
        ("q",),
    )


if __name__ == "__main__":
    run_example()
