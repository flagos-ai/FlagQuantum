"""Compare the unchanged original, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.library.classiq.qmod_library_reference.classiq_open_library.hadamard_transform.hadamard_transform.main import (
    main,
)
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2)


def run_example() -> None:
    compare_states(
        synth.synthesize(main),
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("x",),
    )


if __name__ == "__main__":
    run_example()
