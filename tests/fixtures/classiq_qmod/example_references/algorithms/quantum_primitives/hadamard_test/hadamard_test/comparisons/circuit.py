"""Validate the exact source against an independent complete joint state."""

from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.hadamard_test.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(Path(__file__).with_name("source.qmod"), max_workspace=0)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(),
        fq.run(reference_circuit()).statevector(),
        atol=1e-12,
        rtol=0,
    )
    probabilities = dict(
        program.probabilities(result, registers=("expectation_value",))
    )
    assert abs(probabilities[(0,)] - 5 / 8) < 1e-12
    assert abs(probabilities[(1,)] - 3 / 8) < 1e-12


if __name__ == "__main__":
    run_example()
