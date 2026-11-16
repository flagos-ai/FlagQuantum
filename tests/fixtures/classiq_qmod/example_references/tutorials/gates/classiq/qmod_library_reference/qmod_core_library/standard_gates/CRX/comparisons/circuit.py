"""Compare the exact original Qmod, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.functions.standard_gates.crx.main import (
    main,
)
from flagquantum import synth
from tools.validate_native_source_states import compare_states


def reference_circuit() -> fq.Circuit:
    return fq.Circuit(2, dtype=torch.complex128).crx(1, 0, theta=1)


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("q_target", "q_control"),
    )


if __name__ == "__main__":
    run_example()
