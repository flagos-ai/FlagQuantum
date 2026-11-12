"""Compare the exact original Qmod, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.functions.standard_gates.x.main import (
    main,
)
from flagquantum import synth
from tools.validate_native_source_states import compare_states


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(1, dtype=torch.complex128).x(0)


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("q",),
    )


if __name__ == "__main__":
    run_example()
