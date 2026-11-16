"""Validate the frozen oracle-workshop source against its dense reference."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

QMOD = Path(__file__).with_name("source.qmod")


def reference_circuit() -> fq.Circuit:
    diagonal = torch.tensor(
        [-1 if basis // 4 - basis % 4 == 2 else 1 for basis in range(16)],
        dtype=torch.complex128,
    )
    reflection = torch.full((16, 16), 2 / 16, dtype=torch.complex128) - torch.eye(
        16, dtype=torch.complex128
    )
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        circuit.h(wire)
    for _ in range(2):
        circuit.any(0, 1, 2, 3, matrix=torch.diag(diagonal))
        circuit.any(0, 1, 2, 3, matrix=reflection)
    return circuit


def run_example() -> None:
    compare_states(
        synth.synthesize(QMOD, source_format="qmod"),
        reference_circuit(),
        ("vars.x", "vars.y"),
    )


if __name__ == "__main__":
    run_example()
