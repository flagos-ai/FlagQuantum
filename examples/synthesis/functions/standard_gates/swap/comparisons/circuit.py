"""Compare native FlagQuantum and independent Circuit amplitudes."""

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.standard_gates.swap.main import main
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(2, dtype=torch.complex128).swap(0, 1)


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        reference_circuit(),
        ("q1", "q2"),
    )


if __name__ == "__main__":
    run_example()
