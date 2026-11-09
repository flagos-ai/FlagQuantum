"""Compare the exact original Qmod, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.classiq.the_classiq_tutorial.execution_tutorial.main import (
    main,
)
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    matrix = torch.zeros((512, 512), dtype=torch.complex128)
    for column in range(512):
        x = column >> 6
        matrix[column ^ (x * x + 1), column] = 1
    return (
        fq.Circuit(9, dtype=torch.complex128)
        .h(0)
        .h(1)
        .h(2)
        .any(*range(9), matrix=matrix)
    )


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("x", "y"),
    )


if __name__ == "__main__":
    run_example()
