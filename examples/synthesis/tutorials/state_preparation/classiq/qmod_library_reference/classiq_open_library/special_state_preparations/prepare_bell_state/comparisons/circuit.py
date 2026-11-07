"""Compare the exact original Qmod, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.state_preparation.classiq.qmod_library_reference.classiq_open_library.special_state_preparations.prepare_bell_state.main import (
    main,
)
from flagquantum import synth


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1).x(1)


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("x",),
    )


if __name__ == "__main__":
    run_example()
