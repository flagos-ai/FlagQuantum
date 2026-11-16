"""Compare the unchanged original, native program and independent Circuit."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.entanglement.main import (
    main,
)
from flagquantum import synth
from tools.validate_native_source_states import compare_states


def reference_circuit() -> fq.Circuit:
    circuit = fq.Circuit(6, dtype=torch.complex128).h(5)
    for target in range(4, -1, -1):
        circuit.cx(target + 1, target)
    return circuit


def run_example() -> None:
    compare_states(
        synth.synthesize(main),
        Path(__file__).with_name("source.qmod"),
        reference_circuit(),
        ("reg",),
    )


if __name__ == "__main__":
    run_example()
