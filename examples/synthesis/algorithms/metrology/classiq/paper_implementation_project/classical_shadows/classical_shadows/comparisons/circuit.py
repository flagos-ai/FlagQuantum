"""Compare the unchanged original, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.algorithms.metrology.classiq.paper_implementation_project.classical_shadows.classical_shadows.main import (
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(2, dtype=torch.complex128).h(0).z(0).cx(0, 1)


def run_example() -> None:
    compare_states(
        synth.synthesize(main),
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("qarr",),
    )


if __name__ == "__main__":
    run_example()
