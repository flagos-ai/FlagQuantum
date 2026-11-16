"""Compare native FlagQuantum and independent Circuit amplitudes."""

import math

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification.amplification_iterations.main import (
    one,
    two,
    zero,
)
from flagquantum import synth


def run_example() -> None:
    a = torch.tensor(
        [[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]],
        dtype=torch.complex128,
    ) / math.sqrt(2)
    oracle = torch.diag(torch.tensor([1, 1, -1, -1], dtype=torch.complex128))
    reflection = torch.diag(torch.tensor([1, -1, -1, -1], dtype=torch.complex128))
    grover = a @ reflection @ a.conj().T @ oracle
    for k, function in enumerate((zero, one, two)):
        circuit = fq.Circuit(2, dtype=torch.complex128).any(0, 1, matrix=a)
        for _ in range(k):
            circuit.any(0, 1, matrix=grover)
        expected = fq.run(circuit).statevector().reshape(-1)
        for p in (synth.synthesize(function),):
            result = fq.run(p)
            state = result.statevector().reshape(4, -1)
            torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
            assert float(state[:, 1:].abs().square().sum()) < 1e-24
            print(k, dict(p.probabilities(result, registers=("flag",))))
            print(p.report["resources"])
    print(
        "Native qfunc and Circuit agree, including absolute phase and clean workspace."
    )


if __name__ == "__main__":
    run_example()
