"""Compare the exact original Qmod, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.arithmetic.classiq.function_usage_examples.arithmetic.multiplication.multiplication_2vars_example.main import (
    main,
)
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(12, dtype=torch.complex128).x(0).x(3).x(5).x(7).x(9)


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("a", "b", "res"),
    )


if __name__ == "__main__":
    run_example()
