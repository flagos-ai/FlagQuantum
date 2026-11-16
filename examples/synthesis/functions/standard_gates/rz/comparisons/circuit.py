"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.standard_gates.rz.main import main
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(1, dtype=torch.complex128).rz(0, theta=1.9)


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        reference_circuit(),
        ("q",),
    )


if __name__ == "__main__":
    run_example()
