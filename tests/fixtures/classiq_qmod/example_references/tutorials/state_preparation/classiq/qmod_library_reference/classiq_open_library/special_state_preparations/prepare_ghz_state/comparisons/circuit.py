"""Compare the exact original Qmod, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.functions.state_preparation.prepare_ghz_state.main import (
    main,
)
from flagquantum import synth
from tools.validate_native_source_states import compare_states


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(5, dtype=torch.complex128).h(0)
    for target in range(1, 5):
        circuit.cx(0, target)
    return circuit


def run_example() -> None:
    compare_states(
        synth.synthesize(main, memory_limit_bytes=512 * 1024**2),
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("x",),
    )


if __name__ == "__main__":
    run_example()
